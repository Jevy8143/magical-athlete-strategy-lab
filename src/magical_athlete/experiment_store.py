"""Transactional, purpose-scoped raw logs and content-addressed replay artifacts.

SQLite is the index and transaction boundary. Immutable artifacts can exist
before a commit, but only committed game rows are eligible for any dataset.
No trainer or learned policy is implemented here.
"""

import hashlib
import json
import os
import platform
import random
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .engine import replay_trace
from .experiment_records import build_game_records
from .experiment_spec import OBJECTIVE, GameSeeds, PolicySnapshot, Purpose, RunSpec, canonical_json, content_hash
from .game import GameEngine, replay_game_trace
from .policy_runtime_router import annotate_behavior, require_implemented, snapshot_artifacts

LOG_SCHEMA_VERSION = 1
TABLE_KEYS = {
    "races": "race_index", "decisions": "decision_seq", "events": "sequence",
    "score_entries": "sequence", "transitions": "decision_seq",
}


class ExperimentError(ValueError):
    """An invalid, incompatible or corrupted experiment record."""


def infrastructure_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("experiment_spec.py", "experiment_records.py", "experiment_store.py", "experiments.py",
                 "policy_runtime.py", "model_a_features.py", "model_a.py", "model_a_deployment.py",
                 "policy_runtime_router.py", "model_a_repetition_guard_v1.py", "model_a_repetition_guard_v2.py", "model_a_repetition_guard_v3.py", "model_a_repetition_guard_v4.py", "first_legal_memory_v1.py",
                 "training_a.py", "evaluation_a.py", "benchmark_a.py", "campaign_a.py", "star_baby_cycle.py",
                 "source_compatibility.py", "romantic_queue_cycle.py", "preparing_copy_romantic_cycle.py", "romantic_anchor_cycle.py",
                 "romantic_baby_star_cycle.py", "static_inheritance.py", "fish_baby_star_cycle.py", "romantic_baby_arrow_cycle.py",
                 "romantic_duel_copy_cycle.py", "egg_copy_scoocher_cycle.py", "romantic_scoocher_anchor_cycle.py",
                 "romantic_baby_baba_cycle.py", "copy_fish_baby_cycle.py", "centaur_copy_baby_star_cycle.py",
                 "fish_baby_arrow11_cycle.py", "romantic_baby_shared12_cycle.py", "scoocher_baby_duel22_cycle.py", "fish_baby_baba4_cycle.py", "fish_baby_arrow11_scoocher_cycle.py", "duel_romantic_scoocher22_cycle.py", "copy_baby_duel_scoocher_cycle.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_identity() -> dict:
    return {"python": sys.version, "implementation": sys.implementation.name,
            "platform": platform.platform(), "sqlite": sqlite3.sqlite_version}


def _checked_json(payload: str, digest: str):
    if hashlib.sha256(payload.encode("utf-8")).hexdigest() != digest:
        raise ExperimentError("stored JSON checksum mismatch")
    return json.loads(payload)


def _write_immutable(path: Path, payload: bytes) -> None:
    """Replace only a newly prepared file; an existing different value is an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ExperimentError(f"immutable artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Linking the already flushed temporary file creates the destination
        # atomically and refuses to overwrite a competing writer's result.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ExperimentError(f"immutable artifact differs: {path}")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ExperimentStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "experiments.sqlite3"
        connection = self._connect()
        try:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
                previous = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                if previous is not None and previous[0] != str(LOG_SCHEMA_VERSION):
                    raise ExperimentError("unsupported experiment database schema")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY, payload TEXT NOT NULL, checksum TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    sha256 TEXT PRIMARY KEY, byte_count INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, study_id TEXT NOT NULL, purpose TEXT NOT NULL,
                    spec_hash TEXT NOT NULL, plan_hash TEXT NOT NULL, manifest TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL, infrastructure_hash TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS planned_games (
                    run_id TEXT NOT NULL REFERENCES runs(run_id), game_index INTEGER NOT NULL,
                    seeds TEXT NOT NULL, seeds_hash TEXT NOT NULL,
                    PRIMARY KEY (run_id, game_index));
                CREATE TABLE IF NOT EXISTS seed_claims (
                    seed INTEGER PRIMARY KEY, purpose TEXT NOT NULL, study_id TEXT NOT NULL,
                    game_index INTEGER NOT NULL, stream TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, game_index INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('completed','diagnostic','error')),
                    replay_verified INTEGER NOT NULL, trace_hash TEXT NOT NULL REFERENCES artifacts(sha256),
                    records_hash TEXT NOT NULL, summary TEXT NOT NULL, summary_hash TEXT NOT NULL,
                    failure TEXT, recorded_at TEXT NOT NULL,
                    UNIQUE (run_id, game_index),
                    FOREIGN KEY (run_id, game_index) REFERENCES planned_games(run_id, game_index));
                CREATE TABLE IF NOT EXISTS races (
                    game_id TEXT NOT NULL REFERENCES games(game_id), race_index INTEGER NOT NULL,
                    payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    PRIMARY KEY (game_id, race_index));
                CREATE TABLE IF NOT EXISTS decisions (
                    game_id TEXT NOT NULL REFERENCES games(game_id), decision_seq INTEGER NOT NULL,
                    player_id INTEGER NOT NULL CHECK(player_id BETWEEN 0 AND 3),
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                    payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    PRIMARY KEY (game_id, decision_seq));
                CREATE TABLE IF NOT EXISTS events (
                    game_id TEXT NOT NULL REFERENCES games(game_id), race_index INTEGER NOT NULL,
                    sequence INTEGER NOT NULL, payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    PRIMARY KEY (game_id, race_index, sequence));
                CREATE TABLE IF NOT EXISTS score_entries (
                    game_id TEXT NOT NULL REFERENCES games(game_id), sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    PRIMARY KEY (game_id, sequence));
                CREATE TABLE IF NOT EXISTS transitions (
                    game_id TEXT NOT NULL, decision_seq INTEGER NOT NULL, player_id INTEGER NOT NULL,
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
                    payload TEXT NOT NULL, checksum TEXT NOT NULL,
                    PRIMARY KEY (game_id, decision_seq),
                    FOREIGN KEY (game_id, decision_seq) REFERENCES decisions(game_id, decision_seq));
                CREATE INDEX IF NOT EXISTS games_by_run_status ON games(run_id, status);
                CREATE INDEX IF NOT EXISTS transitions_by_snapshot ON transitions(snapshot_id, game_id);
                CREATE TABLE IF NOT EXISTS quarantined_games (
                    run_id TEXT NOT NULL, game_index INTEGER NOT NULL,
                    artifact_hash TEXT NOT NULL REFERENCES artifacts(sha256), recorded_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, game_index),
                    FOREIGN KEY (run_id, game_index) REFERENCES planned_games(run_id, game_index));
            """)
            connection.execute("INSERT OR IGNORE INTO metadata VALUES ('schema_version', ?)", (str(LOG_SCHEMA_VERSION),))
            version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()[0]
            if version != str(LOG_SCHEMA_VERSION):
                raise ExperimentError("unsupported experiment database schema")
        finally:
            connection.close()

    def _connect(self):
        connection = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _transaction(self):
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _rows(self, sql, parameters=()):
        connection = self._connect()
        try:
            return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def _artifact_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ExperimentError("invalid artifact SHA256")
        return self.root / "artifacts" / digest[:2] / digest

    def put_artifact(self, payload: bytes) -> str:
        if not isinstance(payload, bytes):
            raise ExperimentError("artifact must be bytes")
        digest = hashlib.sha256(payload).hexdigest()
        _write_immutable(self._artifact_path(digest), payload)
        with self._transaction() as connection:
            connection.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?)", (digest, len(payload)))
        return digest

    def read_artifact(self, digest: str) -> bytes:
        rows = self._rows("SELECT byte_count FROM artifacts WHERE sha256=?", (digest,))
        if not rows:
            raise ExperimentError("unregistered artifact")
        try:
            payload = self._artifact_path(digest).read_bytes()
        except OSError as exc:
            raise ExperimentError("missing artifact") from exc
        if len(payload) != rows[0][0] or hashlib.sha256(payload).hexdigest() != digest:
            raise ExperimentError("artifact checksum mismatch")
        return payload

    def register_snapshot(self, snapshot: PolicySnapshot, artifact: bytes | None = None) -> str:
        if snapshot.artifact_sha256 is not None:
            if artifact is not None:
                if hashlib.sha256(artifact).hexdigest() != snapshot.artifact_sha256:
                    raise ExperimentError("model artifact does not match the frozen snapshot")
                self.put_artifact(artifact)
            self.read_artifact(snapshot.artifact_sha256)
        elif artifact is not None:
            raise ExperimentError("this snapshot does not declare an artifact")
        payload = canonical_json(snapshot.to_dict())
        with self._transaction() as connection:
            row = connection.execute("SELECT payload,checksum FROM snapshots WHERE snapshot_id=?", (snapshot.snapshot_id,)).fetchone()
            if row is not None:
                stored = _checked_json(row["payload"], row["checksum"])
                if stored != snapshot.to_dict():
                    raise ExperimentError("snapshot identity collision")
            else:
                connection.execute("INSERT INTO snapshots VALUES (?,?,?)", (snapshot.snapshot_id, payload, snapshot.snapshot_id))
        return snapshot.snapshot_id

    def read_snapshot(self, snapshot_id: str) -> PolicySnapshot:
        rows = self._rows("SELECT payload,checksum FROM snapshots WHERE snapshot_id=?", (snapshot_id,))
        if not rows:
            raise ExperimentError("unknown snapshot")
        data = _checked_json(rows[0]["payload"], rows[0]["checksum"])
        snapshot = PolicySnapshot.from_dict(data)
        if snapshot.snapshot_id != snapshot_id:
            raise ExperimentError("snapshot identity mismatch")
        if snapshot.artifact_sha256:
            self.read_artifact(snapshot.artifact_sha256)
        return snapshot

    def create_run(self, spec: RunSpec, game_indices) -> dict:
        indices = tuple(game_indices)
        if not indices or any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != len(indices):
            raise ExperimentError("a run requires distinct nonnegative game indices")
        indices = tuple(sorted(indices))
        for policy in spec.policies:
            self.register_snapshot(policy)
        manifest = {
            "schema_version": LOG_SCHEMA_VERSION, "spec": spec.to_dict(),
            "spec_hash": spec.spec_id, "game_indices": list(indices),
            "infrastructure_fingerprint": infrastructure_fingerprint(),
            "runtime": runtime_identity(),
            "data_versions": spec.data_versions,
            "seed_plan_hash": content_hash(spec.seed_plan.to_dict()),
            "pairing_unit": ["seed_plan_hash", "game_index"],
            "independent_sample_unit": "game_id",
            "champion_counting": "Each tied-highest player is a co-winner; winner indicators need not sum to one",
        }
        manifest_json = canonical_json(manifest)
        plan_hash = content_hash(manifest)
        with self._transaction() as connection:
            row = connection.execute("SELECT plan_hash,manifest,manifest_hash FROM runs WHERE run_id=?", (spec.run_id,)).fetchone()
            if row is not None:
                _checked_json(row["manifest"], row["manifest_hash"])
                if row["plan_hash"] != plan_hash:
                    raise ExperimentError("run_id is immutable; use a new run_id for a different plan/version")
                return manifest
            connection.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?)", (
                spec.run_id, spec.study_id, spec.purpose.value, spec.spec_id, plan_hash,
                manifest_json, plan_hash, manifest["infrastructure_fingerprint"], datetime.now(timezone.utc).isoformat(),
            ))
            for index in indices:
                seeds = spec.seed_plan.derive(index)
                payload = canonical_json(seeds.to_dict())
                connection.execute("INSERT INTO planned_games VALUES (?,?,?,?)", (spec.run_id, index, payload, content_hash(seeds.to_dict())))
                labels = ("game", "dice", *(f"game_policy:{seat}" for seat in range(4)), *(f"race_policy:{seat}" for seat in range(4)))
                for label, seed in zip(labels, seeds.all_seeds):
                    claim = (spec.purpose.value, spec.study_id, index, label)
                    existing = connection.execute("SELECT purpose,study_id,game_index,stream FROM seed_claims WHERE seed=?", (seed,)).fetchone()
                    # Same study/purpose/index/stream is an explicit paired run.
                    if existing is not None and tuple(existing) != claim:
                        raise ExperimentError("seed stream already belongs to another purpose/study/game/stream")
                    connection.execute("INSERT OR IGNORE INTO seed_claims VALUES (?,?,?,?,?)", (seed, *claim))
        path = self.root / "runs" / spec.purpose.value / spec.spec_id / "manifest.json"
        _write_immutable(path, manifest_json.encode("utf-8"))
        return manifest

    def manifest(self, run_id: str) -> dict:
        rows = self._rows("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if not rows:
            raise ExperimentError("unknown run_id")
        row = rows[0]
        manifest = _checked_json(row["manifest"], row["manifest_hash"])
        if content_hash(manifest) != row["plan_hash"] or content_hash(manifest["spec"]) != row["spec_hash"]:
            raise ExperimentError("run manifest identity mismatch")
        if (manifest["spec"]["run_id"], manifest["spec"]["study_id"], manifest["spec"]["purpose"]) != (run_id, row["study_id"], row["purpose"]):
            raise ExperimentError("run partition metadata mismatch")
        for data in manifest["spec"]["policies"]:
            snapshot_id = content_hash(data)
            if self.read_snapshot(snapshot_id).to_dict() != data:
                raise ExperimentError("run policy differs from its snapshot")
        copy = self.root / "runs" / row["purpose"] / row["spec_hash"] / "manifest.json"
        if copy.exists() and copy.read_bytes() != row["manifest"].encode("utf-8"):
            raise ExperimentError("manifest artifact differs from the transactional index")
        return manifest

    def run_spec(self, run_id: str) -> RunSpec:
        manifest = self.manifest(run_id)
        # Old manifests remain inspectable via manifest(); execution/export is
        # deliberately strict until an explicit version migration is supplied.
        spec = RunSpec.from_dict(manifest["spec"])
        if manifest["infrastructure_fingerprint"] != infrastructure_fingerprint():
            raise ExperimentError("experiment infrastructure version differs; use a new run")
        if manifest["runtime"] != runtime_identity():
            raise ExperimentError("experiment runtime differs; use a new run")
        return spec

    def seeds_for(self, run_id: str, game_index: int) -> GameSeeds:
        rows = self._rows("SELECT seeds,seeds_hash FROM planned_games WHERE run_id=? AND game_index=?", (run_id, game_index))
        if not rows:
            raise ExperimentError("game index is not in this immutable run plan")
        return GameSeeds.from_dict(_checked_json(rows[0]["seeds"], rows[0]["seeds_hash"]))

    def completed_indices(self, run_id: str) -> set[int]:
        """Every committed attempt is preserved, including diagnostics/errors."""
        return {row[0] for row in self._rows(
            "SELECT game_index FROM games WHERE run_id=? UNION SELECT game_index FROM quarantined_games WHERE run_id=?",
            (run_id, run_id),
        )}

    def _validate_trace_identity(self, spec: RunSpec, seeds: GameSeeds, trace: dict) -> None:
        for policy in spec.policies:
            require_implemented(policy)
        if type(trace["seed"]) is not int or type(trace["dice_seed"]) is not int:
            raise ExperimentError("actual shuffle/dice seeds must be integers")
        if trace["seed"] != seeds.game_seed or trace["dice_seed"] != seeds.dice_seed:
            raise ExperimentError("actual shuffle/dice seeds differ from the reserved streams")
        if trace["code_fingerprint"] != spec.engine_fingerprint or trace["rule_version"] != spec.rule_version:
            raise ExperimentError("trace engine/rule version differs from its run")
        if canonical_json(trace["config"]) != canonical_json(spec.game_config):
            raise ExperimentError("trace game configuration differs from its run")
        if trace["run_kind"] != ("official_game" if spec.game_config.official else "synthetic_or_restricted_game_fixture"):
            raise ExperimentError("trace official/fixture classification differs")
        for seat, snapshot in enumerate(spec.policies):
            expected = {"policy_id": snapshot.game_policy_id,
                        "seed": seeds.game_policy_seeds[seat] if snapshot.policy_kind == "random" or snapshot.model_type == "A" else None}
            if trace["game_policies"].get(str(seat)) != expected:
                raise ExperimentError("game behavior policy/seed differs from its snapshot")
        race_traces = list(trace["race_traces"])
        active = trace.get("race_diagnostic") or trace.get("active_race")
        if active:
            race_traces.append(active)
        for race in race_traces:
            for seat, snapshot in enumerate(spec.policies):
                expected = {"policy_id": snapshot.race_policy_id,
                            "seed": seeds.race_policy_seeds[seat] if snapshot.policy_kind == "random" or snapshot.model_type == "A" else None}
                if race["policies"].get(str(seat)) != expected:
                    raise ExperimentError("race behavior policy/seed differs from its snapshot")
        dice = random.Random(seeds.dice_seed)
        for value in trace["dice_rolls"]:
            if type(value) is not int or value != dice.randint(1, 6):
                raise ExperimentError("actual dice sequence differs from its reserved seed")

    def _records(self, trace: dict, spec: RunSpec, seeds: GameSeeds) -> dict:
        return annotate_behavior(build_game_records(trace), spec, seeds, snapshot_artifacts(self, spec))

    @staticmethod
    def _summary(records, spec, seeds, game_index, status, failure=None):
        game_id = f"{spec.spec_id}:{game_index}"
        return {**records["game"], "game_id": game_id, "game_index": game_index,
                "run_id": spec.run_id, "purpose": spec.purpose.value, "matchup": spec.matchup,
                "seat_snapshot_ids": list(spec.seat_snapshot_ids), "seeds": seeds.to_dict(),
                "seat_policies": [{"player_id": seat, "snapshot_id": policy.snapshot_id,
                                   "model_type": policy.model_type, "model_version": policy.model_version,
                                   "game_policy_id": policy.game_policy_id, "race_policy_id": policy.race_policy_id}
                                  for seat, policy in enumerate(spec.policies)],
                "pairing_key": f"{content_hash(spec.seed_plan.to_dict())}:{game_index}",
                "status": status, "replay_verified": status == "completed",
                "failure": failure,
                "independent_sample_unit": "game_id"}

    def record_game(self, run_id: str, game_index: int, trace: dict, *, failure: dict | None = None) -> str:
        spec = self.run_spec(run_id)
        seeds = self.seeds_for(run_id, game_index)
        if seeds != spec.seed_plan.derive(game_index):
            raise ExperimentError("reserved seed plan differs from the run specification")
        trace = json.loads(canonical_json(trace))
        self._validate_trace_identity(spec, seeds, trace)
        complete = trace["completed"] is True
        if complete and (failure is not None or trace.get("diagnostic_failure")):
            raise ExperimentError("a failed game cannot be labelled as a completed sample")
        if complete:
            replay_game_trace(trace)
            status = "completed"
        else:
            if failure is None:
                raise ExperimentError("incomplete game requires an explicit diagnostic/error record")
            if failure.get("kind") not in {"diagnostic", "error"}:
                raise ExperimentError("unknown failure kind")
            status = failure["kind"]
            for race in trace["race_traces"]:
                replay_trace(race)
            partial = trace.get("race_diagnostic")
            if partial:
                replay_trace(partial)
        records = self._records(trace, spec, seeds)
        if not complete and records["transitions"]:
            raise ExperimentError("incomplete games must not produce ordinary transitions")
        trace_hash = self.put_artifact(canonical_json(trace).encode("utf-8"))
        game_id = f"{spec.spec_id}:{game_index}"
        summary = self._summary(records, spec, seeds, game_index, status, failure)
        records["game"] = summary
        records_hash = content_hash(records)
        payload = canonical_json(summary)
        failure_json = canonical_json(failure) if failure else None
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM quarantined_games WHERE run_id=? AND game_index=?", (run_id, game_index)).fetchone():
                raise ExperimentError("game index is already quarantined; preserve it and create a new run")
            existing = connection.execute("SELECT game_id,trace_hash,records_hash,failure FROM games WHERE run_id=? AND game_index=?", (run_id, game_index)).fetchone()
            if existing is not None:
                if (existing["trace_hash"], existing["records_hash"], existing["failure"]) != (trace_hash, records_hash, failure_json):
                    raise ExperimentError("game index already committed with different results; preserve it and create a new run")
                return existing["game_id"]
            connection.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                game_id, run_id, game_index, status, int(complete), trace_hash, records_hash,
                payload, content_hash(summary), failure_json, datetime.now(timezone.utc).isoformat(),
            ))
            for table, values in (("races", records["races"]), ("decisions", records["decisions"]),
                                  ("events", records["events"]), ("score_entries", records["scores"]),
                                  ("transitions", records["transitions"])):
                for item in values:
                    packed, checksum = canonical_json(item), content_hash(item)
                    if table == "events":
                        connection.execute("INSERT INTO events VALUES (?,?,?,?,?)", (game_id, item["race_index"], item["sequence"], packed, checksum))
                    elif table in {"decisions", "transitions"}:
                        seat = item["player_id"]
                        connection.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?)", (game_id, item["decision_seq"], seat, spec.policies[seat].snapshot_id, packed, checksum))
                    else:
                        connection.execute(f"INSERT INTO {table} VALUES (?,?,?,?)", (game_id, item[TABLE_KEYS[table]], packed, checksum))
        return game_id

    def quarantine_game(self, run_id: str, game_index: int, trace, failure: dict) -> None:
        """Preserve even an invalid/unfinished trace without declaring a sample."""
        spec = self.run_spec(run_id)
        self.seeds_for(run_id, game_index)
        data = {"schema_version": LOG_SCHEMA_VERSION, "run_id": run_id, "game_index": game_index,
                "purpose": spec.purpose.value, "trace": trace, "failure": failure,
                "eligible_for_training_or_evaluation": False}
        digest = self.put_artifact(canonical_json(data).encode("utf-8"))
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM games WHERE run_id=? AND game_index=?", (run_id, game_index)).fetchone():
                raise ExperimentError("cannot quarantine or replace an already committed game")
            existing = connection.execute("SELECT artifact_hash FROM quarantined_games WHERE run_id=? AND game_index=?", (run_id, game_index)).fetchone()
            if existing is not None and existing[0] != digest:
                raise ExperimentError("quarantined attempt is immutable; use a new run")
            connection.execute("INSERT OR IGNORE INTO quarantined_games VALUES (?,?,?,?)", (
                run_id, game_index, digest, datetime.now(timezone.utc).isoformat(),
            ))

    def _game_rows(self, run_id: str, purpose: Purpose):
        manifest = self.manifest(run_id)
        if manifest["spec"]["purpose"] != Purpose(purpose).value:
            raise ExperimentError("run purpose mismatch; cross-purpose access is forbidden")
        return self._rows("SELECT * FROM games WHERE run_id=? ORDER BY game_index", (run_id,))

    def read_game_trace(self, run_id: str, game_index: int, *, purpose: Purpose) -> dict:
        rows = self._game_rows(run_id, purpose)
        row = next((row for row in rows if row["game_index"] == game_index), None)
        if row is None:
            raise ExperimentError("game has no committed record")
        return json.loads(self.read_artifact(row["trace_hash"]))

    def summary(self, run_id: str) -> dict:
        manifest = self.manifest(run_id)
        rows = self._game_rows(run_id, Purpose(manifest["spec"]["purpose"]))
        for row in rows:
            stored = _checked_json(row["summary"], row["summary_hash"])
            if (stored["status"], stored["replay_verified"]) != (row["status"], bool(row["replay_verified"])):
                raise ExperimentError("game eligibility metadata mismatch")
        counts = {status: sum(row["status"] == status for row in rows) for status in ("completed", "diagnostic", "error")}
        counts["quarantined"] = self._rows("SELECT COUNT(*) FROM quarantined_games WHERE run_id=?", (run_id,))[0][0]
        planned = len(manifest["game_indices"])
        missing = planned - len(rows) - counts["quarantined"]
        return {
            "run_id": run_id, "purpose": manifest["spec"]["purpose"],
            "planned_games": planned, **counts, "uncommitted_games": missing,
            "status": "incomplete" if missing else "completed" if counts["completed"] == planned else "finished_with_failures",
            "independent_completed_games": counts["completed"],
            "player_outcomes_are_correlated_within_game": True,
            "spec_hash": manifest["spec_hash"],
            "champion_counting": manifest["champion_counting"],
        }

    def failures(self, run_id: str) -> list[dict]:
        manifest = self.manifest(run_id)
        output = []
        for row in self._game_rows(run_id, Purpose(manifest["spec"]["purpose"])):
            if row["status"] != "completed":
                self.read_artifact(row["trace_hash"])
                records = self._load_records(row)
                output.append({"game_index": row["game_index"], "status": row["status"],
                               "failure": records["game"]["failure"], "trace_hash": row["trace_hash"]})
        for row in self._rows("SELECT game_index,artifact_hash FROM quarantined_games WHERE run_id=?", (run_id,)):
            data = json.loads(self.read_artifact(row["artifact_hash"]))
            if (data["run_id"], data["game_index"], data["purpose"]) != (run_id, row["game_index"], manifest["spec"]["purpose"]):
                raise ExperimentError("quarantined failure index differs from its artifact")
            output.append({"game_index": row["game_index"], "status": "quarantined",
                           "failure": data["failure"], "artifact_hash": row["artifact_hash"]})
        return sorted(output, key=lambda item: item["game_index"])

    def _load_records(self, row):
        records = {"game": _checked_json(row["summary"], row["summary_hash"])}
        summary = records["game"]
        if (summary["game_id"], summary["run_id"], summary["game_index"], summary["status"], summary["replay_verified"]) != (
            row["game_id"], row["run_id"], row["game_index"], row["status"], bool(row["replay_verified"]),
        ):
            raise ExperimentError("game index metadata differs from its payload")
        failure = json.loads(row["failure"]) if row["failure"] is not None else None
        if failure != summary["failure"]:
            raise ExperimentError("failure evidence differs from its checksummed payload")
        for table in TABLE_KEYS:
            order = "race_index,sequence" if table == "events" else TABLE_KEYS[table]
            values = self._rows(f"SELECT * FROM {table} WHERE game_id=? ORDER BY {order}", (row["game_id"],))
            items = []
            for value in values:
                item = _checked_json(value["payload"], value["checksum"])
                keys = ("race_index", "sequence") if table == "events" else (TABLE_KEYS[table],)
                if any(value[key] != item[key] for key in keys):
                    raise ExperimentError("record index metadata differs from its payload")
                if table in {"decisions", "transitions"}:
                    seat = item["player_id"]
                    if (value["player_id"] != seat or value["snapshot_id"] != item["snapshot_id"]
                            or value["snapshot_id"] != summary["seat_snapshot_ids"][seat]):
                        raise ExperimentError("record player/snapshot index metadata differs from its payload")
                items.append(item)
            records["scores" if table == "score_entries" else table] = items
        if content_hash(records) != row["records_hash"]:
            raise ExperimentError("normalized game records checksum mismatch")
        return records

    def audit_run(self, run_id: str, *, replay=True) -> dict:
        spec = self.run_spec(run_id)
        checked = 0
        for row in self._game_rows(run_id, spec.purpose):
            records = self._load_records(row)
            trace = json.loads(self.read_artifact(row["trace_hash"]))
            self._validate_trace_identity(spec, self.seeds_for(run_id, row["game_index"]), trace)
            seeds = self.seeds_for(run_id, row["game_index"])
            if seeds != spec.seed_plan.derive(row["game_index"]):
                raise ExperimentError("stored seed plan differs from the run")
            rebuilt = self._records(trace, spec, seeds)
            rebuilt["game"] = self._summary(rebuilt, spec, seeds, row["game_index"], row["status"], records["game"]["failure"])
            if content_hash(rebuilt) != row["records_hash"]:
                raise ExperimentError("normalized records differ from the source trace")
            if replay:
                if row["status"] == "completed":
                    if not row["replay_verified"] or not trace["completed"]:
                        raise ExperimentError("completed eligibility metadata differs")
                    replay_game_trace(trace)
                else:
                    if trace["completed"] or records["transitions"]:
                        raise ExperimentError("failed game is marked as an eligible sample")
                    for race in trace["race_traces"]:
                        replay_trace(race)
                    if trace.get("race_diagnostic"):
                        replay_trace(trace["race_diagnostic"])
            checked += 1
        for row in self._rows("SELECT game_index,artifact_hash FROM quarantined_games WHERE run_id=?", (run_id,)):
            data = json.loads(self.read_artifact(row["artifact_hash"]))
            if (data["run_id"] != run_id or data["game_index"] != row["game_index"]
                    or data["purpose"] != spec.purpose.value or data["eligible_for_training_or_evaluation"] is not False):
                raise ExperimentError("quarantine identity/eligibility mismatch")
        return {**self.summary(run_id), "records_checked": checked, "replay_checked": replay,
                "infrastructure_fingerprint": infrastructure_fingerprint()}

    def training_transitions(self, run_id: str, *, model_type: str, snapshot_id: str) -> list[dict]:
        spec = self.run_spec(run_id)
        if spec.purpose != Purpose.TRAIN:
            raise ExperimentError("only train runs may supply training samples")
        if model_type not in {"A", "B"} or {policy.model_type for policy in spec.policies} != {model_type}:
            raise ExperimentError("training model family differs from this 4A/4B run")
        if snapshot_id not in spec.seat_snapshot_ids:
            raise ExperimentError("snapshot is not a behavior policy in this run")
        if model_type == "A":
            from .policy_runtime_router import require_trainable_snapshot
            # A frozen reference may be an opponent in a TRAIN game, but its
            # transferred estimates are never the learner's on-policy source.
            for policy in spec.policies:
                if policy.snapshot_id == snapshot_id:
                    require_trainable_snapshot(policy)
        self.audit_run(run_id, replay=False)
        output = []
        for row in self._game_rows(run_id, Purpose.TRAIN):
            if row["status"] != "completed" or not row["replay_verified"]:
                continue
            for transition in self._load_records(row)["transitions"]:
                seat = transition["player_id"]
                if spec.policies[seat].snapshot_id == snapshot_id:
                    if "model_a_deployment" in transition:
                        raise ExperimentError("fixed-context deployment trajectories are not trainable")
                    output.append({**transition, "game_id": row["game_id"], "run_id": run_id,
                                   "purpose": "train", "model_type": model_type,
                                   "snapshot_id": snapshot_id, "seat_snapshot_ids": list(spec.seat_snapshot_ids),
                                   "observation_version": spec.observation_version})
        return output

    def export(self, run_id: str, destination: str | Path, *, purpose: Purpose,
               model_type: str | None = None, snapshot_id: str | None = None) -> dict:
        spec = self.run_spec(run_id)
        purpose = Purpose(purpose)
        if spec.purpose != purpose:
            raise ExperimentError("export purpose differs from the immutable run")
        if purpose == Purpose.TRAIN:
            records = self.training_transitions(run_id, model_type=model_type, snapshot_id=snapshot_id)
            dataset_kind = "same_player_training_transitions"
        elif purpose in {Purpose.EVALUATION, Purpose.CROSSPLAY}:
            if model_type is not None or snapshot_id is not None:
                raise ExperimentError("model/snapshot training selectors cannot relabel evaluation exports")
            self.audit_run(run_id, replay=False)
            records = [self._load_records(row)["game"] for row in self._game_rows(run_id, purpose)
                       if row["status"] == "completed" and row["replay_verified"]]
            dataset_kind = "frozen_policy_complete_game_outcomes"
        else:
            raise ExperimentError("diagnostic logs cannot be exported as a training/evaluation dataset")
        payload = "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")
        manifest = {
            "schema_version": LOG_SCHEMA_VERSION, "purpose": purpose.value, "dataset_kind": dataset_kind,
            "run_id": run_id, "spec_hash": spec.spec_id, "model_type": model_type,
            "snapshot_id": snapshot_id, "seat_snapshot_ids": list(spec.seat_snapshot_ids),
            "objective": spec.objective, "record_count": len(records),
            "independent_game_count": len({record["game_id"] for record in records}),
            "independent_sample_unit": "game_id", "player_outcomes_are_correlated_within_game": True,
            "sha256": hashlib.sha256(payload).hexdigest(), "source_summary": self.summary(run_id),
            "run_manifest": self.manifest(run_id),
        }
        manifest["plan_hash"] = content_hash(manifest["run_manifest"])
        manifest["manifest_sha256"] = content_hash(manifest)
        destination = Path(destination)
        _write_immutable(destination, payload)
        _write_immutable(destination.with_name(destination.name + ".manifest.json"), canonical_json(manifest).encode("utf-8"))
        return manifest


def verify_export(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text(encoding="utf-8"))
    checksum = manifest.get("manifest_sha256")
    if checksum != content_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise ExperimentError("export manifest checksum mismatch")
    payload = path.read_bytes()
    if manifest["schema_version"] != LOG_SCHEMA_VERSION or hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ExperimentError("export checksum/schema mismatch")
    purpose = Purpose(manifest["purpose"])
    expected_kind = "same_player_training_transitions" if purpose == Purpose.TRAIN else "frozen_policy_complete_game_outcomes"
    if purpose == Purpose.DIAGNOSTIC or manifest["dataset_kind"] != expected_kind:
        raise ExperimentError("export purpose/dataset kind mismatch")
    run = manifest["run_manifest"]
    spec = run["spec"]
    snapshots = [content_hash(policy) for policy in spec["policies"]]
    if (
        manifest["objective"] != OBJECTIVE or spec["objective"] != OBJECTIVE
        or manifest["plan_hash"] != content_hash(run)
        or manifest["spec_hash"] != content_hash(spec) or run["spec_hash"] != manifest["spec_hash"]
        or spec["purpose"] != purpose.value or spec["run_id"] != manifest["run_id"]
        or snapshots != manifest["seat_snapshot_ids"]
    ):
        raise ExperimentError("export objective/run/policy lineage mismatch")
    if purpose == Purpose.TRAIN:
        if ({policy["model_type"] for policy in spec["policies"]} != {manifest["model_type"]}
                or manifest["model_type"] not in {"A", "B"} or manifest["snapshot_id"] not in snapshots):
            raise ExperimentError("training export model/snapshot partition mismatch")
    elif manifest["model_type"] is not None or manifest["snapshot_id"] is not None:
        raise ExperimentError("evaluation export has training selectors")
    records = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(records) != manifest["record_count"] or len({record["game_id"] for record in records}) != manifest["independent_game_count"]:
        raise ExperimentError("export record/game count mismatch")
    if any(record["purpose"] != manifest["purpose"] or record["run_id"] != manifest["run_id"] for record in records):
        raise ExperimentError("export crosses run/purpose partitions")
    if any(record["seat_snapshot_ids"] != snapshots or not record["game_id"].startswith(manifest["spec_hash"] + ":") for record in records):
        raise ExperimentError("export game/policy lineage mismatch")
    return manifest
