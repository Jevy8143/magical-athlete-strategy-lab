"""Small, fully logged model-A control iterations and independent paired checks.

Each game has one current A and three frozen initial A opponents. Rotating the
focus seat keeps the actual opponent context fixed across checkpoint updates.
All four players' records are retained; only the declared focal trajectory is
consumed by this learner. No evaluation game is used to update the model.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from .experiment_spec import PolicySnapshot, Purpose, RunSpec, SeedPlan, canonical_json, content_hash
from .experiment_store import ExperimentError, ExperimentStore, _checked_json, _write_immutable, infrastructure_fingerprint, runtime_identity
from .experiments import run_experiment
from .game import GameConfig, GameEngine, official_config, replay_game_trace
from .model_a import ALGORITHM, AConfig, FrozenAModel
from .model_a_features import FEATURE_VERSION
from .policy_runtime_router import load_model, opponent_context_id, require_implemented, require_trainable_snapshot

PILOT_PROTOCOL = "focal-4A-fixed-initial-opponents-v2"


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Invocation safety margins; these are not rewards or model parameters."""

    minimum_project_free_bytes: int = 30 * 1024**3
    minimum_system_free_bytes: int = 5 * 1024**3
    bytes_per_planned_record: int = 5_000_000
    temporary_reserve_bytes: int = 512 * 1024**2

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("resource limits must be nonnegative integer byte counts")
        if self.bytes_per_planned_record == 0:
            raise ValueError("planned record storage allowance must be positive")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class ResourceBudgetError(ExperimentError):
    """No new games should start until local capacity is available again."""


def check_resources(store, limits, planned_records, phase, *, expected_source=None):
    """Persist observed capacity outside deterministic game/model evidence."""
    if type(planned_records) is not int or planned_records < 0:
        raise ValueError("planned_records must be a nonnegative integer")
    actual_source = (GameEngine._code_fingerprint(), infrastructure_fingerprint())
    source_ok = expected_source is None or tuple(expected_source) == actual_source
    record = {"kind": "model-a-invocation-resource-check-v1", "phase": phase,
              "planned_records": planned_records, "game_fingerprint": actual_source[0],
              "infrastructure_fingerprint": actual_source[1], "source_unchanged": source_ok,
              "observed_at_unix_seconds": time.time(), "limits": limits.to_dict() if limits else None}
    allowed = True
    if limits is not None:
        project = store.root.resolve()
        system = Path(os.environ.get("SystemDrive", Path.home().anchor))
        if system.drive and not system.is_absolute():
            system = Path(str(system) + "\\")
        project_usage, system_usage = shutil.disk_usage(project), shutil.disk_usage(system)
        allocation = planned_records * limits.bytes_per_planned_record + limits.temporary_reserve_bytes
        required_project = limits.minimum_project_free_bytes + allocation
        required_system = limits.minimum_system_free_bytes
        # On a single-volume installation the project reservation also covers
        # the system volume. Never count the same free space twice.
        if project.anchor == system.anchor:
            required_project = max(limits.minimum_project_free_bytes, required_system) + allocation
        allowed = project_usage.free >= required_project and system_usage.free >= required_system
        record.update(project_path=str(project), system_path=str(system),
                      project_free_bytes=project_usage.free, system_free_bytes=system_usage.free,
                      required_project_free_bytes=required_project,
                      required_system_free_bytes=required_system, capacity_available=allowed)
    record["allowed_to_start_new_games"] = bool(source_ok and allowed)
    path = store.root / "resource-checks" / (uuid.uuid4().hex + ".json")
    _write_immutable(path, canonical_json(record).encode("utf-8"))
    if not source_ok:
        raise ExperimentError(f"source changed during model-A invocation; no further update/games allowed; evidence={path}")
    if not allowed:
        raise ResourceBudgetError(f"insufficient project/system free space; original games retained; evidence={path}")
    return record


@contextmanager
def local_work_environment(store):
    """Keep child worker temporary files on the project volume, then restore."""
    root = store.root / "worker-cache"
    paths = {"TEMP": root / "temp", "TMP": root / "temp", "TMPDIR": root / "temp",
             "PIP_CACHE_DIR": root / "pip", "PYTHONPYCACHEPREFIX": root / "pycache"}
    for path in set(paths.values()):
        path.mkdir(parents=True, exist_ok=True)
    previous = {name: os.environ.get(name) for name in paths}
    previous_tempdir = tempfile.tempdir
    try:
        for name, path in paths.items():
            os.environ[name] = str(path.resolve())
        tempfile.tempdir = str(paths["TEMP"].resolve())
        yield
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def pilot_config():
    # Safety budgets produce retained diagnostics, never invented game scores.
    # Keep pathological queue growth cheap enough to investigate promptly.
    return replace(official_config(), max_events_per_race=10_000, max_turns_per_race=2_000)


@dataclass(frozen=True, slots=True)
class PilotPlan:
    pilot_id: str
    master_seed: int = 20261029
    batches: int = 3
    games_per_seat: int = 8
    evaluation_games_per_seat: int = 8
    model_config: AConfig = field(default_factory=AConfig)
    game_config: GameConfig = field(default_factory=pilot_config)
    parent_artifact_sha256: str | None = None
    opponent_snapshot_id: str | None = None
    strict_training_health: bool = False
    source_semantics_certificate_sha256: str | None = None
    repetition_guard: str | None = None
    source_revision_reference_pilot_id: str | None = None
    source_revision_reference_plan_sha256: str | None = None
    source_revision_start_batch_index: int | None = None
    source_revision_ultimate_pilot_id: str | None = None
    source_revision_preserve_learned_parent: bool = False
    source_revision_preserve_learned_prefix: bool = False
    evaluation_guard_scheme: str | None = None

    def __post_init__(self):
        if not isinstance(self.pilot_id, str) or not self.pilot_id or len(self.pilot_id) > 48:
            raise ValueError("pilot_id must be a nonempty portable identifier of at most 48 characters")
        SeedPlan(self.master_seed, Purpose.TRAIN, self.pilot_id)
        for value in (self.batches, self.games_per_seat, self.evaluation_games_per_seat):
            if type(value) is not int or value < 1:
                raise ValueError("pilot batch and game counts must be positive integers")
        if not isinstance(self.model_config, AConfig) or not isinstance(self.game_config, GameConfig) or not self.game_config.official:
            raise ValueError("model A pilot requires an explicit A configuration and verified official game")
        if (self.parent_artifact_sha256 is None) != (self.opponent_snapshot_id is None):
            raise ValueError("parent_artifact_sha256 and opponent_snapshot_id must be supplied together")
        if self.repetition_guard not in (None, "all-legal-exhaustion-release-v1", "public-turn-local-duel-keep-sweep-v2",
                                         "public-turn-local-duel-fixed-keep-uniform-retry-v3"):
            raise ValueError("unsupported pilot repetition guard version")
        if self.evaluation_guard_scheme is not None:
            if type(self.evaluation_guard_scheme) is not str or not self.evaluation_guard_scheme:
                raise ValueError("evaluation guard must be an explicit supported scheme")
            from .policy_runtime_router import repetition_guard_binding
            repetition_guard_binding(self.evaluation_guard_scheme)
        if type(self.strict_training_health) is not bool:
            raise ValueError("strict_training_health must be boolean")
        if self.source_semantics_certificate_sha256 is not None and self.parent_artifact_sha256 is None:
            raise ValueError("source semantics certification requires an explicit continuation parent")
        for digest in (self.parent_artifact_sha256, self.opponent_snapshot_id, self.source_semantics_certificate_sha256):
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(character not in "0123456789abcdef" for character in digest)):
                raise ValueError("continuation parent and opponent IDs must be SHA256 digests")
        reference = self.source_revision_reference_pilot_id
        pin = self.source_revision_reference_plan_sha256
        if type(self.source_revision_preserve_learned_prefix) is not bool:
            raise ValueError("learned-prefix source revision opt-in must be boolean")
        if self.source_revision_preserve_learned_prefix and (
                reference is None or self.source_semantics_certificate_sha256 is None
                or self.source_revision_start_batch_index is None
                or self.source_revision_ultimate_pilot_id != reference
                or self.source_revision_preserve_learned_parent):
            raise ValueError("certified learned-prefix tail requires direct ordinary reference, current exact-prefix certificate and absolute tail")
        if type(self.source_revision_preserve_learned_parent) is not bool:
            raise ValueError("source revision learned-parent opt-in must be boolean")
        if self.source_revision_preserve_learned_parent and (
                reference is None or self.source_semantics_certificate_sha256 is None
                or self.source_revision_start_batch_index is not None
                or self.source_revision_ultimate_pilot_id is not None):
            raise ValueError("learned-parent source revision requires a reference, current certificate and whole unconsumed cohort")
        if (reference is None) != (pin is None):
            raise ValueError("source revision reference pilot and plan SHA256 must be supplied together")
        if reference is not None:
            if (not isinstance(reference, str) or not reference or len(reference) > 48
                    or reference == self.pilot_id):
                raise ValueError("source revision requires a distinct portable reference pilot")
            SeedPlan(self.master_seed, Purpose.TRAIN, reference)
            if (type(pin) is not str or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin)):
                raise ValueError("source revision reference plan must be an immutable SHA256")
            if self.parent_artifact_sha256 is None or not self.strict_training_health:
                raise ValueError("source revision requires an explicit empty parent/opponent and strict complete batches")
            if (self.source_semantics_certificate_sha256 is not None and self.source_revision_start_batch_index is None
                    and not self.source_revision_preserve_learned_parent):
                raise ValueError("unconsumed source revision does not inherit a learned-source certificate")
        start = self.source_revision_start_batch_index
        ultimate = self.source_revision_ultimate_pilot_id
        if (start is None) != (ultimate is None):
            raise ValueError("tail source revision requires both absolute start batch and ultimate pilot")
        if start is not None:
            if (reference is None or type(start) is not int or not 0 < start < self.batches
                    or not isinstance(ultimate, str) or not ultimate or len(ultimate) > 48
                    or ultimate == self.pilot_id
                    or (ultimate == reference and not self.source_revision_preserve_learned_prefix)):
                raise ValueError("tail source revision requires a remaining absolute batch and distinct ultimate pilot")
            SeedPlan(self.master_seed, Purpose.TRAIN, ultimate)
            if self.source_semantics_certificate_sha256 is None:
                raise ValueError("tail recovery requires current-source certification of its actual learned parent")

    def to_dict(self):
        result = {"protocol": PILOT_PROTOCOL, "pilot_id": self.pilot_id, "master_seed": self.master_seed,
                "retention_policy": "all_games_full_trace_and_summary",
                "batches": self.batches, "games_per_seat": self.games_per_seat,
                "evaluation_games_per_seat": self.evaluation_games_per_seat,
                "model_config": self.model_config.to_dict(),
                "game_config": json.loads(canonical_json(self.game_config)),
                "engine_fingerprint": GameEngine._code_fingerprint(),
                "infrastructure_fingerprint": infrastructure_fingerprint(), "runtime": runtime_identity()}
        if self.repetition_guard is not None:
            from .policy_runtime_router import repetition_guard_binding
            result["repetition_guard"] = repetition_guard_binding(self.repetition_guard)
        # Keep the original non-continuation plan shape and hash convention.
        if self.parent_artifact_sha256 is not None:
            result.update(parent_artifact_sha256=self.parent_artifact_sha256,
                          opponent_snapshot_id=self.opponent_snapshot_id)
        if self.strict_training_health:
            result["strict_training_health"] = True
        if self.source_semantics_certificate_sha256 is not None:
            result.update(protocol=PILOT_PROTOCOL + "-source-semantics-v1",
                          source_semantics_certificate_sha256=self.source_semantics_certificate_sha256,
                          historical_source_policy="exact complete-source current-engine equivalence and byte-identical MC reconstruction")
        if self.source_revision_reference_pilot_id is not None:
            result.update(protocol=PILOT_PROTOCOL + "-source-revision-v1",
                          source_revision_reference_pilot_id=self.source_revision_reference_pilot_id,
                          source_revision_reference_plan_sha256=self.source_revision_reference_plan_sha256,
                          source_revision_policy="new run identities; original studies, frozen policies and exact ten-stream coordinates; consume complete recovery cohort once")
        if self.source_revision_start_batch_index is not None:
            result.update(protocol=PILOT_PROTOCOL + "-source-revision-tail-v1",
                          source_revision_start_batch_index=self.source_revision_start_batch_index,
                          source_revision_ultimate_pilot_id=self.source_revision_ultimate_pilot_id,
                          source_revision_policy="certified actual learned prefix; new remaining-run identities at original absolute coordinates; no repeated prefix consumption")
        if self.source_revision_preserve_learned_parent:
            result.update(protocol=PILOT_PROTOCOL + "-source-revision-certified-parent-v1",
                          source_revision_preserve_learned_parent=True,
                          source_revision_policy="same certified learned parent and original fixed opponents; new run identities at original coordinates; consume the entire previously unconsumed cohort once")
        if self.source_revision_preserve_learned_prefix:
            result.update(protocol=PILOT_PROTOCOL + "-source-revision-certified-prefix-tail-v1",
                          source_revision_preserve_learned_prefix=True,
                          source_revision_policy="exact inherited learned initial plus original committed prefix; only remaining absolute coordinates under new run identities")
        if self.evaluation_guard_scheme is not None:
            from .policy_runtime_router import repetition_guard_binding
            result.update(evaluation_guard_scheme=self.evaluation_guard_scheme,
                          evaluation_guard_binding=repetition_guard_binding(self.evaluation_guard_scheme),
                          evaluation_guard_policy="mean EVAL focal snapshots use the explicit guard; TRAIN and fixed opponents retain the base guard")
        return result


def model_snapshot(store, model, version, mode="thompson"):
    digest = store.put_artifact(model.to_bytes())
    snapshot = PolicySnapshot(
        "A", "model_artifact", version,
        canonical_json({"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": mode}),
        frozen=True, artifact_sha256=digest, training_games=model.training_games,
    )
    store.register_snapshot(snapshot)
    return snapshot


def _pilot_snapshot(store, plan, model, version, mode="thompson"):
    if plan.source_revision_reference_pilot_id is not None:
        if not version.startswith(plan.pilot_id + "-"):
            raise ExperimentError("source revision snapshot version must belong to its recovery pilot")
        version = (plan.source_revision_ultimate_pilot_id or plan.source_revision_reference_pilot_id) + version[len(plan.pilot_id):]
    scheme = (plan.evaluation_guard_scheme if mode == "mean" and plan.evaluation_guard_scheme is not None
              else plan.repetition_guard)
    if scheme is None:
        return model_snapshot(store, model, version, mode)
    from .policy_runtime_router import guard_snapshot
    return guard_snapshot(store, model, version, mode, guard_scheme=scheme)


def _ensure_tables(store):
    with store._transaction() as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS a_pilots (
            pilot_id TEXT PRIMARY KEY, plan TEXT NOT NULL, plan_hash TEXT NOT NULL,
            initial_artifact TEXT NOT NULL REFERENCES artifacts(sha256))""")
        connection.execute("""CREATE TABLE IF NOT EXISTS a_updates (
            pilot_id TEXT NOT NULL REFERENCES a_pilots(pilot_id), batch_index INTEGER NOT NULL,
            parent_artifact TEXT NOT NULL REFERENCES artifacts(sha256),
            child_artifact TEXT NOT NULL REFERENCES artifacts(sha256),
            evidence TEXT NOT NULL, evidence_hash TEXT NOT NULL,
            PRIMARY KEY (pilot_id,batch_index))""")


def _archive_sources(store):
    """Retain exact source bytes before a pilot, alongside version fingerprints."""
    engine_before, infrastructure_before = GameEngine._code_fingerprint(), infrastructure_fingerprint()
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = sorted(package.glob("*.py"))
    paths.extend([root / "pyproject.toml"])
    files = {}
    for path in paths:
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(), "utf8": raw.decode("utf-8"),
        }
    if (engine_before != GameEngine._code_fingerprint() or infrastructure_before != infrastructure_fingerprint()
            or any(hashlib.sha256((root / name).read_bytes()).hexdigest() != record["sha256"]
                   for name, record in files.items())):
        raise ExperimentError("source changed while archiving; no pilot started")
    bundle = {"kind": "magical-athlete-source-v2", "files": files,
              "engine_fingerprint": engine_before,
              "infrastructure_fingerprint": infrastructure_before}
    digest = store.put_artifact(canonical_json(bundle).encode("utf-8"))
    index = {"artifact_sha256": digest, "engine_fingerprint": bundle["engine_fingerprint"],
             "infrastructure_fingerprint": bundle["infrastructure_fingerprint"]}
    _write_immutable(store.root / "source-archives" / (digest + ".json"), canonical_json(index).encode("utf-8"))
    return digest


def _run_spec(plan, purpose, stage, seat, current, initial):
    policies = [initial] * 4
    policies[seat] = current
    reference = plan.source_revision_ultimate_pilot_id or plan.source_revision_reference_pilot_id or plan.pilot_id
    return RunSpec(
        f"{plan.pilot_id}-{stage}-s{seat}", f"{reference}-{'train' if purpose == Purpose.TRAIN else 'holdout'}",
        purpose, tuple(policies),
        SeedPlan(plan.master_seed, purpose, f"{reference}-{'train' if purpose == Purpose.TRAIN else 'holdout'}"),
        plan.game_config, learning=purpose == Purpose.TRAIN,
    )


def _validate_source_revision_reference(store, plan):
    """Authenticate an unconsumed cohort, without relabelling old attempts."""
    if plan.source_revision_preserve_learned_prefix:
        return _validate_source_revision_certified_prefix_tail_reference(store, plan)
    if plan.source_revision_start_batch_index is not None:
        return _validate_source_revision_tail_reference(store, plan)
    if plan.source_revision_preserve_learned_parent:
        return _validate_source_revision_certified_parent_reference(store, plan)
    reference = plan.source_revision_reference_pilot_id
    rows = store._rows("SELECT plan,plan_hash,initial_artifact FROM a_pilots WHERE pilot_id=?", (reference,))
    if len(rows) != 1:
        raise ExperimentError("source revision reference pilot is not registered")
    row = rows[0]
    original = _checked_json(row["plan"], row["plan_hash"])
    if row["plan_hash"] != plan.source_revision_reference_plan_sha256 or original["pilot_id"] != reference:
        raise ExperimentError("source revision reference plan hash/identity differs")
    current = plan.to_dict()
    fields = ("master_seed", "batches", "games_per_seat", "evaluation_games_per_seat", "model_config", "game_config")
    if (any(canonical_json(original[key]) != canonical_json(current[key]) for key in fields)
            or original.get("repetition_guard") != current.get("repetition_guard")
            or original.get("strict_training_health", False) is not True
            or original.get("protocol") != PILOT_PROTOCOL
            or "source_revision_reference_pilot_id" in original
            or "source_semantics_certificate_sha256" in original):
        raise ExperimentError("source revision reference model, seeds, counts, configuration or guard differs")
    empty = FrozenAModel.empty(plan.model_config)
    if (row["initial_artifact"] != plan.parent_artifact_sha256
            or plan.parent_artifact_sha256 != empty.artifact_sha256
            or store.read_artifact(plan.parent_artifact_sha256) != empty.to_bytes()
            or original.get("parent_artifact_sha256", empty.artifact_sha256) != empty.artifact_sha256):
        raise ExperimentError("source revision requires the exact original untrained parent")
    from .policy_runtime_router import repetition_guard_binding
    parameters = {"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": "thompson"}
    if plan.repetition_guard is not None:
        parameters["repetition_guard"] = repetition_guard_binding(plan.repetition_guard)
    expected = PolicySnapshot("A", "model_artifact", f"{reference}-c0", canonical_json(parameters),
                              frozen=True, artifact_sha256=empty.artifact_sha256, training_games=0)
    opponent = store.read_snapshot(plan.opponent_snapshot_id)
    if (opponent.to_dict() != expected.to_dict()
            or original.get("opponent_snapshot_id", expected.snapshot_id) != expected.snapshot_id):
        raise ExperimentError("source revision fixed original A0 snapshot differs")
    if store._rows("SELECT 1 FROM a_updates WHERE pilot_id=?", (reference,)):
        raise ExperimentError("source revision reference pilot already consumed TRAIN samples")
    studies = {}
    for update in store._rows("SELECT pilot_id,evidence,evidence_hash FROM a_updates"):
        evidence = _checked_json(update["evidence"], update["evidence_hash"])
        if update["pilot_id"] == plan.pilot_id:
            continue  # An exact invocation resume reconstructs its own committed update.
        for source in evidence["source_games"]:
            run_id = source["run_id"]
            if run_id not in studies:
                run = store.manifest(run_id)["spec"]
                studies[run_id] = (run["purpose"], run["study_id"])
            if studies[run_id] == (Purpose.TRAIN.value, reference + "-train"):
                raise ExperimentError("source revision original TRAIN cohort was already consumed by another pilot")


def _validate_source_revision_certified_parent_reference(store, plan):
    """Keep an ordinary pilot's certified learned parent, not its failed attempts.

    The reference certificate authenticates its historical plan only. The new
    current-source certificate is verified by _starting_model_and_opponent.
    """
    reference = plan.source_revision_reference_pilot_id
    rows = store._rows("SELECT plan,plan_hash,initial_artifact FROM a_pilots WHERE pilot_id=?", (reference,))
    if len(rows) != 1 or rows[0]["plan_hash"] != plan.source_revision_reference_plan_sha256:
        raise ExperimentError("learned-parent source revision reference plan hash/identity differs")
    registered = rows[0]
    original = _checked_json(registered["plan"], registered["plan_hash"])
    current = plan.to_dict()
    fields = ("master_seed", "batches", "games_per_seat", "evaluation_games_per_seat", "model_config", "game_config")
    if (original.get("pilot_id") != reference
            or original.get("protocol") != PILOT_PROTOCOL + "-source-semantics-v1"
            or any(key.startswith("source_revision_") for key in original)
            or original.get("strict_training_health") is not True
            or any(canonical_json(original[key]) != canonical_json(current[key]) for key in fields)
            or canonical_json(original.get("repetition_guard")) != canonical_json(current.get("repetition_guard"))):
        raise ExperimentError("learned-parent source revision parameters, original seeds or guard differ")
    parent = FrozenAModel.from_bytes(store.read_artifact(plan.parent_artifact_sha256))
    if (registered["initial_artifact"] != original.get("parent_artifact_sha256")
            or original.get("parent_artifact_sha256") != plan.parent_artifact_sha256
            or parent.training_games <= 0 or parent.config != plan.model_config
            or original.get("opponent_snapshot_id") != plan.opponent_snapshot_id):
        raise ExperimentError("learned-parent source revision exact initial parent/fixed opponent differs")
    original_certificate = json.loads(store.read_artifact(original["source_semantics_certificate_sha256"]))
    from .source_compatibility import CERTIFICATE_KIND
    binding = original_certificate.get("bindings", {})
    source = binding.get("current_source", {})
    probe = _run_spec(plan, Purpose.TRAIN, "historical-context-check", 0,
                      store.read_snapshot(plan.opponent_snapshot_id), store.read_snapshot(plan.opponent_snapshot_id))
    if (original_certificate.get("kind") != CERTIFICATE_KIND
            or type(original_certificate.get("schema_version")) is not int or original_certificate["schema_version"] != 1
            or type(original_certificate.get("new_training_samples")) is not int or original_certificate["new_training_samples"] != 0
            or binding.get("parent_artifact_sha256") != parent.artifact_sha256
            or binding.get("parent_training_games") != parent.training_games
            or binding.get("parent_source_batches_sha256") != content_hash(list(parent.source_batches))
            or binding.get("original_source_game_ids") != sorted(parent.training_game_ids)
            or canonical_json(binding.get("game_config")) != canonical_json(plan.game_config)
            or binding.get("context_id") != opponent_context_id(probe, 0)
            or binding.get("opponent_snapshot_ids") != [plan.opponent_snapshot_id] * 3
            or source.get("game_fingerprint") != original["engine_fingerprint"]
            or source.get("infrastructure_fingerprint") != original["infrastructure_fingerprint"]):
        raise ExperimentError("learned-parent source revision historical certificate binding differs")
    if store._rows("SELECT 1 FROM a_updates WHERE pilot_id=?", (reference,)):
        raise ExperimentError("learned-parent source revision reference cohort already consumed")
    studies = {}
    for update in store._rows("SELECT pilot_id,evidence,evidence_hash FROM a_updates"):
        evidence = _checked_json(update["evidence"], update["evidence_hash"])
        if update["pilot_id"] == plan.pilot_id:
            continue  # Exact immutable resume reconstructs its own committed update.
        for item in evidence["source_games"]:
            run_id = item["run_id"]
            if run_id not in studies:
                spec = store.manifest(run_id)["spec"]
                studies[run_id] = (spec["purpose"], spec["study_id"])
            if studies[run_id] == (Purpose.TRAIN.value, reference + "-train"):
                raise ExperimentError("learned-parent source revision cohort already consumed by another pilot")


def _validate_source_revision_certified_prefix_tail_reference(store, plan):
    """Continue only the unconsumed tail of an ordinary learned-parent pilot.

    Its historical certificate may authenticate a strict ancestor of the
    ordinary initial model. The new certificate must bind the entire actual
    latest committed prefix; _starting_model_and_opponent verifies it normally.
    """
    reference = plan.source_revision_reference_pilot_id
    rows = store._rows("SELECT plan,plan_hash,initial_artifact FROM a_pilots WHERE pilot_id=?", (reference,))
    if len(rows) != 1 or rows[0]["plan_hash"] != plan.source_revision_reference_plan_sha256:
        raise ExperimentError("certified-prefix tail actual ordinary reference hash differs")
    registered = rows[0]
    original = _checked_json(registered["plan"], registered["plan_hash"])
    current = plan.to_dict()
    fields = ("master_seed", "batches", "games_per_seat", "evaluation_games_per_seat", "model_config", "game_config")
    if (original.get("pilot_id") != reference or plan.source_revision_ultimate_pilot_id != reference
            or original.get("protocol") != PILOT_PROTOCOL + "-source-semantics-v1"
            or any(key.startswith("source_revision_") for key in original)
            or original.get("strict_training_health") is not True
            or any(canonical_json(original[key]) != canonical_json(current[key]) for key in fields)
            or canonical_json(original.get("repetition_guard")) != canonical_json(current.get("repetition_guard"))
            or original.get("opponent_snapshot_id") != plan.opponent_snapshot_id
            or registered["initial_artifact"] != original.get("parent_artifact_sha256")):
        raise ExperimentError("certified-prefix tail ordinary parameters, original seeds or fixed opponent differ")
    base = FrozenAModel.from_bytes(store.read_artifact(registered["initial_artifact"]))
    parent = FrozenAModel.from_bytes(store.read_artifact(plan.parent_artifact_sha256))
    start = plan.source_revision_start_batch_index
    size = 4 * plan.games_per_seat
    offset = len(base.source_batches)
    if (base.training_games <= 0 or base.config != plan.model_config or parent.config != base.config
            or parent.training_games != base.training_games + start * size
            or len(parent.source_batches) != offset + start
            or parent.source_batches[:offset] != base.source_batches
            or not set(base.training_game_ids).issubset(parent.training_game_ids)):
        raise ExperimentError("certified-prefix tail exact inherited initial or committed prefix differs")
    from .source_compatibility import CERTIFICATE_KIND
    opponent = store.read_snapshot(plan.opponent_snapshot_id)
    probe = _run_spec(plan, Purpose.TRAIN, "historical-context-check", 0, opponent, opponent)
    context = opponent_context_id(probe, 0)
    history = json.loads(store.read_artifact(original["source_semantics_certificate_sha256"]))
    binding = history.get("bindings", {})
    anchor = FrozenAModel.from_bytes(store.read_artifact(binding.get("parent_artifact_sha256", "")))
    source = binding.get("current_source", {})
    if (history.get("kind") != CERTIFICATE_KIND or type(history.get("schema_version")) is not int
            or history["schema_version"] != 1 or type(history.get("new_training_samples")) is not int
            or history["new_training_samples"] != 0 or anchor.training_games <= 0
            or anchor.config != base.config or anchor.training_games > base.training_games
            or base.source_batches[:len(anchor.source_batches)] != anchor.source_batches
            or not set(anchor.training_game_ids).issubset(base.training_game_ids)
            or binding.get("parent_artifact_sha256") != anchor.artifact_sha256
            or binding.get("parent_training_games") != anchor.training_games
            or binding.get("parent_source_batches_sha256") != content_hash(list(anchor.source_batches))
            or binding.get("original_source_game_ids") != sorted(anchor.training_game_ids)
            or canonical_json(binding.get("game_config")) != canonical_json(plan.game_config)
            or binding.get("context_id") != context or binding.get("opponent_snapshot_ids") != [plan.opponent_snapshot_id] * 3
            or source.get("game_fingerprint") != original["engine_fingerprint"]
            or source.get("infrastructure_fingerprint") != original["infrastructure_fingerprint"]
            or any(row.context_id != context for row in base.rows)):
        raise ExperimentError("certified-prefix tail historical ancestor certificate binding differs")
    certificate = json.loads(store.read_artifact(plan.source_semantics_certificate_sha256))
    actual = certificate.get("bindings", {})
    if (certificate.get("kind") != CERTIFICATE_KIND or type(certificate.get("schema_version")) is not int
            or certificate["schema_version"] != 1 or type(certificate.get("new_training_samples")) is not int
            or certificate["new_training_samples"] != 0
            or actual.get("parent_artifact_sha256") != parent.artifact_sha256
            or actual.get("parent_training_games") != parent.training_games
            or actual.get("parent_source_batches_sha256") != content_hash(list(parent.source_batches))
            or actual.get("original_source_game_ids") != sorted(parent.training_game_ids)
            or actual.get("context_id") != context or actual.get("opponent_snapshot_ids") != [plan.opponent_snapshot_id] * 3
            or canonical_json(actual.get("game_config")) != canonical_json(plan.game_config)):
        raise ExperimentError("certified-prefix tail needs the exact full current-prefix certificate")
    updates = store._rows("SELECT * FROM a_updates WHERE pilot_id=? ORDER BY batch_index", (reference,))
    if [row["batch_index"] for row in updates] != list(range(start)):
        raise ExperimentError("certified-prefix tail requires exactly the original committed absolute prefix")
    allowed, coordinates, ids = set(), set(), set(base.training_game_ids)
    previous = base.artifact_sha256
    for index, (raw_batch, row) in enumerate(zip(parent.source_batches[offset:], updates, strict=True)):
        batch = json.loads(raw_batch)
        evidence = _checked_json(row["evidence"], row["evidence_hash"])
        checkpoint = FrozenAModel.from_bytes(store.read_artifact(row["child_artifact"]))
        if (row["parent_artifact"] != previous or batch["parent_artifact_sha256"] != previous
                or batch["batch_id"] != f"{reference}-batch-{index}"
                or evidence.get("pilot_id") != reference or evidence.get("batch_index") != index
                or evidence.get("parent_artifact") != previous or evidence.get("child_artifact") != row["child_artifact"]
                or evidence.get("consumed_independent_games") != size or len(batch["games"]) != size
                or checkpoint.config != base.config
                or checkpoint.source_batches != parent.source_batches[:offset + index + 1]
                or checkpoint.training_games != base.training_games + (index + 1) * size
                or [game["game_id"] for game in batch["games"]] != [game["game_id"] for game in evidence["source_games"]]):
            raise ExperimentError("certified-prefix tail original update/model lineage differs")
        allowed.add((reference, index))
        for source_game in batch["games"]:
            games = store._rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (source_game["game_id"], source_game["run_id"]))
            if (len(games) != 1 or games[0]["status"] != "completed" or games[0]["replay_verified"] != 1
                    or games[0]["failure"] is not None):
                raise ExperimentError("certified-prefix tail original source is not complete verified TRAIN")
            spec = store.manifest(source_game["run_id"])["spec"]
            focus, game_index = source_game["focus_player_id"], games[0]["game_index"]
            if (type(focus) is not int or focus not in range(4) or type(game_index) is not int
                    or spec["purpose"] != "train" or spec["learning"] is not True
                    or spec["study_id"] != reference + "-train"
                    or spec["seed_plan"] != SeedPlan(plan.master_seed, Purpose.TRAIN, reference + "-train").to_dict()
                    or canonical_json(spec["game_config"]) != canonical_json(current["game_config"])
                    or not index * size <= game_index < (index + 1) * size
                    or focus != (game_index // plan.games_per_seat) % 4
                    or source_game["context_id"] != context or source_game["purpose"] != "train"
                    or source_game["opponent_snapshot_ids"] != [plan.opponent_snapshot_id] * 3
                    or source_game["game_id"] in ids or (game_index, focus) in coordinates):
                raise ExperimentError("certified-prefix tail original absolute coordinate/context differs")
            behavior = spec["policies"][focus]
            if (behavior["artifact_sha256"] != previous
                    or behavior["training_games"] != base.training_games + index * size
                    or content_hash(behavior) != source_game["behavior_snapshot_id"]
                    or json.loads(behavior["parameters_json"]).get("mode") != "thompson"
                    or json.loads(behavior["parameters_json"]).get("repetition_guard") != current.get("repetition_guard")
                    or "deployment" in json.loads(behavior["parameters_json"])
                    or any(content_hash(spec["policies"][seat]) != plan.opponent_snapshot_id for seat in range(4) if seat != focus)):
                raise ExperimentError("certified-prefix tail original frozen behavior/opponents differ")
            coordinates.add((game_index, focus))
            ids.add(source_game["game_id"])
        previous = row["child_artifact"]
    if previous != parent.artifact_sha256 or ids != set(parent.training_game_ids) or len(coordinates) != start * size:
        raise ExperimentError("certified-prefix tail parent is not exactly the latest registered learned prefix")
    studies = {}
    for row in store._rows("SELECT pilot_id,batch_index,evidence,evidence_hash FROM a_updates"):
        evidence = _checked_json(row["evidence"], row["evidence_hash"])
        if row["pilot_id"] == plan.pilot_id:
            continue  # Immutable exact resume recomputes and compares its own update.
        for game in evidence["source_games"]:
            run_id = game["run_id"]
            if run_id not in studies:
                spec = store.manifest(run_id)["spec"]
                studies[run_id] = (spec["purpose"], spec["study_id"])
            if studies[run_id] == ("train", reference + "-train") and (row["pilot_id"], row["batch_index"]) not in allowed:
                raise ExperimentError("certified-prefix tail already consumed outside its original committed prefix")


def _validate_source_revision_tail_reference(store, plan):
    """Authenticate the actual recovered prefix and an entirely unconsumed tail."""
    reference = plan.source_revision_reference_pilot_id
    rows = store._rows("SELECT * FROM a_pilots WHERE pilot_id=?", (reference,))
    if len(rows) != 1:
        raise ExperimentError("tail reference pilot is not registered")
    registered = rows[0]
    original = _checked_json(registered["plan"], registered["plan_hash"])
    if registered["plan_hash"] != plan.source_revision_reference_plan_sha256:
        raise ExperimentError("tail actual reference plan hash differs")
    current = plan.to_dict()
    ultimate = plan.source_revision_ultimate_pilot_id
    cursor, chain = original, set()
    while "source_revision_reference_pilot_id" in cursor:
        previous = cursor["source_revision_reference_pilot_id"]
        if previous in chain or len(chain) >= 16:
            raise ExperimentError("tail source reference chain is cyclic or excessive")
        chain.add(previous)
        ancestor = store._rows("SELECT plan,plan_hash,initial_artifact FROM a_pilots WHERE pilot_id=?", (previous,))
        if len(ancestor) != 1 or ancestor[0]["plan_hash"] != cursor["source_revision_reference_plan_sha256"]:
            raise ExperimentError("tail source reference ancestor hash differs")
        cursor = _checked_json(ancestor[0]["plan"], ancestor[0]["plan_hash"])
    if cursor["pilot_id"] != ultimate or ultimate == reference or cursor["protocol"] != PILOT_PROTOCOL:
        raise ExperimentError("tail ultimate namespace differs from its actual reference chain")
    fields = ("master_seed", "batches", "games_per_seat", "evaluation_games_per_seat", "model_config", "game_config")
    for source in (original, cursor):
        if (any(canonical_json(source[key]) != canonical_json(current[key]) for key in fields)
                or source.get("repetition_guard") != current.get("repetition_guard")
                or source.get("strict_training_health") is not True):
            raise ExperimentError("tail reference parameters, original seeds, counts or guard differ")
    empty = FrozenAModel.empty(plan.model_config)
    if registered["initial_artifact"] != empty.artifact_sha256:
        raise ExperimentError("tail reference must be the actual recovered-from-zero pilot")
    parent = FrozenAModel.from_bytes(store.read_artifact(plan.parent_artifact_sha256))
    start = plan.source_revision_start_batch_index
    batch_size = 4 * plan.games_per_seat
    if (parent.config != plan.model_config or parent.training_games != start * batch_size
            or len(parent.source_batches) != start):
        raise ExperimentError("tail learned parent does not match the complete absolute prefix")
    from .policy_runtime_router import repetition_guard_binding
    parameters = {"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": "thompson"}
    if plan.repetition_guard is not None:
        parameters["repetition_guard"] = repetition_guard_binding(plan.repetition_guard)
    expected = PolicySnapshot("A", "model_artifact", ultimate + "-c0", canonical_json(parameters),
                              frozen=True, artifact_sha256=empty.artifact_sha256, training_games=0)
    if store.read_snapshot(plan.opponent_snapshot_id).to_dict() != expected.to_dict():
        raise ExperimentError("tail original fixed A0 differs")
    prefix_updates = store._rows("SELECT * FROM a_updates WHERE pilot_id=? ORDER BY batch_index", (reference,))
    if [row["batch_index"] for row in prefix_updates] != list(range(start)):
        raise ExperimentError("tail reference must have exactly the certified committed prefix")
    allowed_updates, coordinates, prefix_ids = set(), set(), set()
    previous_artifact = empty.artifact_sha256
    for index, (raw_batch, row) in enumerate(zip(parent.source_batches, prefix_updates, strict=True)):
        batch = json.loads(raw_batch)
        evidence = _checked_json(row["evidence"], row["evidence_hash"])
        checkpoint = FrozenAModel.from_bytes(store.read_artifact(row["child_artifact"]))
        if (row["parent_artifact"] != previous_artifact or batch["parent_artifact_sha256"] != previous_artifact
                or batch["batch_id"] != f"{reference}-batch-{index}"
                or checkpoint.source_batches != parent.source_batches[:index + 1]
                or checkpoint.training_games != (index + 1) * batch_size
                or [game["game_id"] for game in batch["games"]] != [game["game_id"] for game in evidence["source_games"]]):
            raise ExperimentError("tail reference update/model prefix lineage differs")
        previous_artifact = row["child_artifact"]
        allowed_updates.add((reference, index))
        for source in batch["games"]:
            game_rows = store._rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (source["game_id"], source["run_id"]))
            if len(game_rows) != 1 or game_rows[0]["status"] != "completed" or game_rows[0]["replay_verified"] != 1:
                raise ExperimentError("tail prefix requires actual complete verified TRAIN sources")
            spec = store.manifest(source["run_id"])["spec"]
            focus, game_index = source["focus_player_id"], game_rows[0]["game_index"]
            if (spec["purpose"] != "train" or spec["learning"] is not True
                    or spec["study_id"] != ultimate + "-train"
                    or spec["seed_plan"] != SeedPlan(plan.master_seed, Purpose.TRAIN, ultimate + "-train").to_dict()
                    or canonical_json(spec["game_config"]) != canonical_json(current["game_config"])
                    or not index * batch_size <= game_index < (index + 1) * batch_size
                    or focus != (game_index // plan.games_per_seat) % 4):
                raise ExperimentError("tail prefix original absolute TRAIN coordinate differs")
            coordinate = (game_index, focus)
            if coordinate in coordinates or source["game_id"] in prefix_ids:
                raise ExperimentError("tail prefix repeats an original source coordinate")
            coordinates.add(coordinate)
            prefix_ids.add(source["game_id"])
    if previous_artifact != parent.artifact_sha256 or prefix_ids != set(parent.training_game_ids):
        raise ExperimentError("tail parent is not the actual latest committed reference checkpoint")
    studies = {}
    for row in store._rows("SELECT pilot_id,batch_index,evidence,evidence_hash FROM a_updates"):
        evidence = _checked_json(row["evidence"], row["evidence_hash"])
        if row["pilot_id"] == plan.pilot_id:
            continue  # Exact resumed tail recomputes the same update, never adds it twice.
        for source in evidence["source_games"]:
            run_id = source["run_id"]
            if run_id not in studies:
                spec = store.manifest(run_id)["spec"]
                studies[run_id] = (spec["purpose"], spec["study_id"])
            if studies[run_id] == ("train", ultimate + "-train") and (row["pilot_id"], row["batch_index"]) not in allowed_updates:
                raise ExperimentError("tail cohort already consumed outside its certified learned prefix")


def _starting_model_and_opponent(store, plan):
    if plan.source_revision_reference_pilot_id is not None:
        _validate_source_revision_reference(store, plan)
    if plan.parent_artifact_sha256 is None:
        model = FrozenAModel.empty(plan.model_config)
        return model, _pilot_snapshot(store, plan, model, f"{plan.pilot_id}-c0")

    parent_bytes = store.read_artifact(plan.parent_artifact_sha256)
    parent = FrozenAModel.from_bytes(parent_bytes)
    if parent.config != plan.model_config:
        raise ExperimentError("continuation model configuration differs from its parent")
    opponent = store.read_snapshot(plan.opponent_snapshot_id)
    require_implemented(opponent)
    if (opponent.model_type != "A" or not opponent.frozen
            or json.loads(opponent.parameters_json).get("mode") != "thompson"):
        raise ExperimentError("continuation opponent must be a frozen current-algorithm Thompson model A")
    declared_guard = json.loads(opponent.parameters_json).get("repetition_guard")
    if ((declared_guard is None) != (plan.repetition_guard is None)
            or declared_guard is not None and declared_guard["scheme"] != plan.repetition_guard):
        raise ExperimentError("pilot policy guard differs from its frozen opponent")
    opponent_model = load_model(opponent, {opponent.artifact_sha256: store.read_artifact(opponent.artifact_sha256)})
    if opponent_model.config != parent.config:
        raise ExperimentError("continuation opponent configuration differs from its parent")

    probe = _run_spec(plan, Purpose.TRAIN, "context-check", 0, opponent, opponent)
    context = opponent_context_id(probe, 0)
    if any(row.context_id != context for row in parent.rows):
        raise ExperimentError("continuation parent contains a different exact opponent context or game configuration")
    if plan.source_semantics_certificate_sha256 is not None:
        from .source_compatibility import certified_source_conditions

        conditions, audit = certified_source_conditions(store, (parent,), plan.source_semantics_certificate_sha256)
        if (conditions and list(conditions) != [canonical_json(plan.game_config)]):
            raise ExperimentError("certified continuation source game configuration differs from its plan")
        for source in audit["original_sources"]:
            if (source["source"]["context_id"] != context
                    or source["source"]["opponent_snapshot_ids"] != [opponent.snapshot_id] * 3):
                raise ExperimentError("certified continuation exact opponent/context differs from its plan")
        return parent, opponent
    # Historical infrastructure versions remain inspectable. The semantic game,
    # data, rules, actual behavior and consumed evidence must still match.
    manifests = {}
    reconstructed = FrozenAModel.empty(parent.config)
    for raw_batch in parent.source_batches:
        batch = json.loads(raw_batch)
        if batch["parent_artifact_sha256"] != reconstructed.artifact_sha256:
            raise ExperimentError("continuation parent artifact chain differs from its source evidence")
        games = []
        for source in batch["games"]:
            if (source["context_id"] != context
                    or source["opponent_snapshot_ids"] != [opponent.snapshot_id] * 3):
                raise ExperimentError("continuation source has a different exact opponent context")
            run_id = source["run_id"]
            if run_id not in manifests:
                manifest = store.manifest(run_id)
                # Deliberately avoid store.run_spec: a changed trainer/archive
                # fingerprint does not invalidate unchanged historical games.
                spec = RunSpec.from_dict(manifest["spec"])
                if (spec.purpose != Purpose.TRAIN or not spec.learning
                        or {policy.model_type for policy in spec.policies} != {"A"}
                        or spec.data_versions != probe.data_versions
                        or canonical_json(spec.game_config) != canonical_json(plan.game_config)):
                    raise ExperimentError("continuation source is not compatible verified TRAIN game data")
                manifests[run_id] = spec
            spec = manifests[run_id]
            focus = source["focus_player_id"]
            behavior = spec.policies[focus]
            require_trainable_snapshot(behavior)
            if (opponent_context_id(spec, focus) != context
                    or behavior.snapshot_id != source["behavior_snapshot_id"]
                    or behavior.artifact_sha256 != reconstructed.artifact_sha256
                    or not behavior.frozen
                    or json.loads(behavior.parameters_json).get("mode") != "thompson"):
                raise ExperimentError("continuation source behavior or exact opponent lineage differs")
            rows = store._rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (source["game_id"], run_id))
            if len(rows) != 1 or rows[0]["status"] != "completed" or rows[0]["replay_verified"] != 1:
                raise ExperimentError("continuation parent source must be a completed replay-verified TRAIN game")
            row = rows[0]
            records = store._load_records(row)
            if (records["game"]["purpose"] != "train" or not records["game"]["completed"]
                    or source["game_id"] != f"{spec.spec_id}:{row['game_index']}"):
                raise ExperimentError("continuation source game identity or training eligibility differs")
            trace = json.loads(store.read_artifact(row["trace_hash"]))
            seeds = store.seeds_for(run_id, row["game_index"])
            if seeds != spec.seed_plan.derive(row["game_index"]):
                raise ExperimentError("continuation source reserved seed plan differs")
            store._validate_trace_identity(spec, seeds, trace)
            replay_game_trace(trace)
            rebuilt = store._records(trace, spec, seeds)
            rebuilt["game"] = store._summary(rebuilt, spec, seeds, row["game_index"], "completed")
            if content_hash(rebuilt) != row["records_hash"]:
                raise ExperimentError("continuation source records differ from their replayed trace")
            games.append({**source, "transitions": [item for item in records["transitions"]
                                                    if item["player_id"] == focus]})
        reconstructed = reconstructed.update_games(games, batch_id=batch["batch_id"])
    if reconstructed.to_bytes() != parent_bytes:
        raise ExperimentError("continuation parent does not reproduce from its retained TRAIN evidence")
    return parent, opponent


def _catalog(store, spec, focus):
    output = []
    for row in store._game_rows(spec.run_id, spec.purpose):
        records = store._load_records(row)
        game = records["game"]
        selected_audits = [decision["model_a_audit"] for decision in records["decisions"]
                           if decision["player_id"] == focus and "model_a_audit" in decision]
        informed = sum(audit["estimates"][audit["selected_index"]]["observations"] > 0 for audit in selected_audits)
        preferences = sum(audit["mode"] == "mean" and len(audit["action_keys"]) > 1
                          and len(audit["tied_best_indices"]) < len(audit["action_keys"])
                          and any(estimate["observations"] > 0 for estimate in audit["estimates"])
                          and audit.get("mixture_component", "model") == "model"
                          for audit in selected_audits)
        output.append({**game, "focus_player_id": focus, "trace_artifact_sha256": row["trace_hash"],
                       "trace_relative_path": f"artifacts/{row['trace_hash'][:2]}/{row['trace_hash']}",
                       "focus_decisions": len(selected_audits), "focus_selected_with_evidence": informed,
                       "mean_decisions_with_learned_preferences": preferences,
                       "focus_repetition_fallbacks": sum(audit.get("mixture_component") == "repetition" for audit in selected_audits),
                       "focus_repetition_opportunities": sum(audit.get("repetition_triggered", False) for audit in selected_audits),
                       "focus_uniform_explorations": sum(audit.get("mixture_component") == "uniform" for audit in selected_audits),
                       "focal_total": game["final_scores"][focus] if game["completed"] else None,
                       "focal_co_winner": focus in game["winners"] if game["completed"] else None})
    for row in store._rows("SELECT game_index,artifact_hash FROM quarantined_games WHERE run_id=?", (spec.run_id,)):
        artifact = json.loads(store.read_artifact(row["artifact_hash"]))
        output.append({"game_id": f"{spec.spec_id}:{row['game_index']}", "game_index": row["game_index"],
                       "run_id": spec.run_id, "purpose": spec.purpose.value, "status": "quarantined",
                       "completed": False, "focus_player_id": focus, "focal_total": None,
                       "seat_snapshot_ids": list(spec.seat_snapshot_ids),
                       "quarantine_artifact_sha256": row["artifact_hash"], "failure": artifact["failure"]})
    return sorted(output, key=lambda item: item["game_index"])


def _evaluate(store, plan, model, initial, stage, workers, *, guard=None):
    frozen = _pilot_snapshot(store, plan, model, f"{plan.pilot_id}-{stage}", "mean")
    before = model.artifact_sha256
    catalog, runs = [], []
    for seat in range(4):
        if guard:
            guard(f"{stage}-seat-{seat}", plan.evaluation_games_per_seat)
        spec = _run_spec(plan, Purpose.EVALUATION, stage, seat, frozen, initial)
        indices = range(seat * plan.evaluation_games_per_seat, (seat + 1) * plan.evaluation_games_per_seat)
        summary = run_experiment(store, spec, indices, workers=workers)
        store.audit_run(spec.run_id)
        runs.append(summary)
        catalog.extend(_catalog(store, spec, seat))
    if before != model.artifact_sha256:
        raise ExperimentError("evaluation changed the frozen model")
    return catalog, runs


def _learning_health(parent, child, source_games):
    """Audit actual score updates on training inputs, never a strength metric.

    Prediction changes on these already-consumed requests demonstrate that
    samples affect estimates and preferences. They are in-sample engineering
    diagnostics; only independently frozen evaluation can assess score gains.
    """
    from .model_a_features import encode_request

    decisions = multi = changed_values = changed_preferences = prior_before = prior_after = 0
    returns, per_scope = [], {}
    for game in source_games:
        for row in game["transitions"]:
            if row["player_id"] != game["focus_player_id"]:
                continue
            encoded = encode_request(row["request"], race_index=row["race_index"] if row["scope"] == "race" else None)
            before = parent.select(game["context_id"], encoded, mode="mean", rng=random.Random(0)).audit
            after = child.select(game["context_id"], encoded, mode="mean", rng=random.Random(0)).audit
            old_values = [estimate["mean"] for estimate in before["estimates"]]
            new_values = [estimate["mean"] for estimate in after["estimates"]]
            decisions += 1
            per_scope[row["scope"]] = per_scope.get(row["scope"], 0) + 1
            returns.append(row["return_to_go"])
            changed_values += old_values != new_values
            if len(encoded.actions) > 1:
                multi += 1
                changed_preferences += before["tied_best_indices"] != after["tied_best_indices"]
                prior_before += all(estimate["observations"] == 0 for estimate in before["estimates"])
                prior_after += all(estimate["observations"] == 0 for estimate in after["estimates"])
    observations_before = sum(row.observations for row in parent.rows)
    observations_after = sum(row.observations for row in child.rows)
    if (not decisions or parent.artifact_sha256 == child.artifact_sha256
            or observations_after <= observations_before
            or child.training_games != parent.training_games + len(source_games)):
        raise ExperimentError("training samples did not produce a verifiable score update; checkpoint refused")
    warnings = []
    if not changed_values:
        warnings.append("observations increased but all inspected means stayed equal; inspect return variation and prior mass")
    if multi and not changed_preferences:
        warnings.append("no greedy preference changed in this batch; inspect grouping/sample coverage before increasing scale")
    return {"verified_score_updates": True, "independent_complete_games": len(source_games),
            "hierarchical_observations_before": observations_before,
            "hierarchical_observations_after": observations_after,
            "descriptive_decision_count": decisions, "decisions_by_scope": per_scope,
            "requests_with_changed_mean_estimates": changed_values,
            "multi_action_requests": multi, "requests_with_changed_greedy_preferences": changed_preferences,
            "multi_action_requests_entirely_prior_before": prior_before,
            "multi_action_requests_entirely_prior_after": prior_after,
            "remaining_score_return_min": min(returns), "remaining_score_return_max": max(returns),
            "parent_artifact": parent.artifact_sha256, "child_artifact": child.artifact_sha256,
            "warnings": warnings,
            "scope": "in-sample update audit only; not independent samples or evidence of score improvement"}


def _train_batch(store, plan, model, initial, batch_index, workers, *, guard=None):
    behavior = _pilot_snapshot(store, plan, model, f"{plan.pilot_id}-c{batch_index}")
    require_trainable_snapshot(behavior)
    source_games, sources, catalogs, runs = [], [], [], []
    for seat in range(4):
        if guard:
            guard(f"training-batch-{batch_index}-seat-{seat}", plan.games_per_seat)
        spec = _run_spec(plan, Purpose.TRAIN, f"b{batch_index}", seat, behavior, initial)
        start = (batch_index * 4 + seat) * plan.games_per_seat
        summary = run_experiment(store, spec, range(start, start + plan.games_per_seat), workers=workers)
        store.audit_run(spec.run_id)
        runs.append(summary)
        catalogs.extend(_catalog(store, spec, seat))
        transitions = store.training_transitions(spec.run_id, model_type="A", snapshot_id=behavior.snapshot_id)
        by_game = {}
        for transition in transitions:
            if transition["player_id"] == seat:
                by_game.setdefault(transition["game_id"], []).append(transition)
        for row in store._game_rows(spec.run_id, Purpose.TRAIN):
            if row["status"] != "completed" or not row["replay_verified"]:
                continue
            opponent_ids = [spec.policies[(seat + offset) % 4].snapshot_id for offset in (1, 2, 3)]
            source_games.append({"game_id": row["game_id"], "context_id": opponent_context_id(spec, seat),
                                 "focus_player_id": seat, "purpose": "train", "run_id": spec.run_id,
                                 "behavior_snapshot_id": behavior.snapshot_id,
                                 "opponent_snapshot_ids": opponent_ids, "transitions": by_game[row["game_id"]]})
            sources.append({"game_id": row["game_id"], "trace_hash": row["trace_hash"],
                            "run_id": spec.run_id, "focus_player_id": seat,
                            "behavior_snapshot_id": behavior.snapshot_id, "opponent_snapshot_ids": opponent_ids})
    if any(run["quarantined"] or run["error"] or run["uncommitted_games"] for run in runs):
        raise ExperimentError("a training batch has quarantined/error/uncommitted games; no update was applied")
    if plan.strict_training_health and any(run["completed"] != run["planned_games"] for run in runs):
        raise ExperimentError("strict training batch contains diagnostic/incomplete games; no update was applied")
    if not source_games:
        raise ExperimentError("training batch has no complete games; diagnostics were retained and no update was applied")
    if guard:
        guard(f"training-batch-{batch_index}-before-update")
    child = model.update_games(source_games, batch_id=f"{plan.pilot_id}-batch-{batch_index}")
    health = _learning_health(model, child, source_games)
    digest = store.put_artifact(child.to_bytes())
    evidence = {"protocol": PILOT_PROTOCOL, "pilot_id": plan.pilot_id, "batch_index": batch_index,
                "parent_artifact": model.artifact_sha256, "child_artifact": digest,
                "consumed_independent_games": len(source_games), "source_games": sources,
                "source_run_summaries": runs, "model_summary": child.summary(), "learning_health": health}
    if plan.source_revision_reference_pilot_id is not None:
        evidence.update(protocol=plan.to_dict()["protocol"],
                        source_revision_reference_pilot_id=plan.source_revision_reference_pilot_id,
                        source_revision_reference_plan_sha256=plan.source_revision_reference_plan_sha256,
                        original_failed_or_complete_attempts_consumed=0)
    if plan.source_semantics_certificate_sha256 is not None:
        evidence.update(protocol=plan.to_dict()["protocol"],
                        source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256,
                        semantic_diagnostic_games_used_as_training_samples=0)
    # Invocation-only counters must not change the immutable learning evidence
    # when this batch is recovered after a process stopped before committing it.
    for summary in evidence["source_run_summaries"]:
        summary.pop("previously_committed_games", None)
        summary.pop("newly_attempted_games", None)
    with store._transaction() as connection:
        previous = connection.execute("SELECT * FROM a_updates WHERE pilot_id=? AND batch_index=?", (plan.pilot_id, batch_index)).fetchone()
        if previous is not None:
            if (_checked_json(previous["evidence"], previous["evidence_hash"]) != evidence
                    or previous["parent_artifact"] != model.artifact_sha256 or previous["child_artifact"] != digest):
                raise ExperimentError("checkpoint update already committed with different evidence")
        else:
            connection.execute("INSERT INTO a_updates VALUES (?,?,?,?,?,?)", (
                plan.pilot_id, batch_index, model.artifact_sha256, digest,
                canonical_json(evidence), content_hash(evidence),
            ))
    return child, catalogs, runs


def _paired_report(initial, final, master_seed):
    left = {game["pairing_key"]: game for game in initial if game["completed"]}
    right = {game["pairing_key"]: game for game in final if game["completed"]}
    pairs = sorted(set(left).intersection(right))
    differences = [right[key]["focal_total"] - left[key]["focal_total"] for key in pairs]
    if not differences:
        return {"paired_games": 0, "conclusion": "no complete paired evaluation games"}
    rng_seed = int(content_hash({"statistic": "paired-game-bootstrap-v1", "master_seed": master_seed}), 16) & ((1 << 63) - 1)
    rng = random.Random(rng_seed)
    means = sorted(statistics.fmean(rng.choices(differences, k=len(differences))) for _ in range(2000))
    return {"paired_games": len(pairs), "independent_sample_unit": "paired_game_seed_and_focus_seat",
            "initial_mean_total": statistics.fmean(left[key]["focal_total"] for key in pairs),
            "final_mean_total": statistics.fmean(right[key]["focal_total"] for key in pairs),
            "mean_paired_total_difference": statistics.fmean(differences),
            "paired_bootstrap_95_percent_interval": [means[49], means[1949]],
            "bootstrap_replicates": 2000, "bootstrap_seed": rng_seed,
            "conclusion": "pilot pipeline check only; limited sample and one frozen opponent distribution, no general strength claim"}


def run_pilot(store: ExperimentStore, plan: PilotPlan, *, workers=1, progress=None,
              resource_limits: ResourceLimits | None = None) -> dict:
    """Execute one immutable pilot with separate timing and local work paths."""
    if resource_limits is not None and not isinstance(resource_limits, ResourceLimits):
        raise ValueError("resource_limits must be ResourceLimits or None")
    started = time.perf_counter()
    invocation = {"pilot_id": plan.pilot_id, "workers": workers,
                  "started_at_unix_seconds": time.time(), "status": "failed",
                  "resource_limits": resource_limits.to_dict() if resource_limits else None}
    try:
        with local_work_environment(store):
            report = _run_pilot(store, plan, workers=workers, progress=progress,
                                resource_limits=resource_limits)
        invocation["status"] = report["status"]
        invocation["retained_game_records"] = report["retained_game_records"]
        return report
    except BaseException as error:
        invocation["failure_type"] = type(error).__name__
        raise
    finally:
        invocation["wall_elapsed_seconds"] = time.perf_counter() - started
        _write_immutable(store.root / "model-a-pilot-invocations" / plan.pilot_id / (uuid.uuid4().hex + ".json"),
                         canonical_json(invocation).encode("utf-8"))


def _run_pilot(store: ExperimentStore, plan: PilotPlan, *, workers=1, progress=None,
               resource_limits=None) -> dict:
    expected_source = (GameEngine._code_fingerprint(), infrastructure_fingerprint())

    def guard(phase, records=0):
        return check_resources(store, resource_limits, records, f"{plan.pilot_id}:{phase}",
                               expected_source=expected_source)

    start_batch = plan.source_revision_start_batch_index or 0
    planned = 4 * (plan.batches - start_batch) * plan.games_per_seat + 8 * plan.evaluation_games_per_seat
    guard("before-pilot", planned)
    _ensure_tables(store)
    source_archive = _archive_sources(store)
    initial_model, initial = _starting_model_and_opponent(store, plan)
    guard("after-parent-verification", planned)
    payload = canonical_json(plan.to_dict())
    plan_hash = content_hash(plan.to_dict())
    with store._transaction() as connection:
        row = connection.execute("SELECT plan,plan_hash,initial_artifact FROM a_pilots WHERE pilot_id=?", (plan.pilot_id,)).fetchone()
        if row is not None:
            if _checked_json(row["plan"], row["plan_hash"]) != plan.to_dict() or row["initial_artifact"] != initial_model.artifact_sha256:
                raise ExperimentError("pilot plan/version is immutable; use a new pilot_id")
        else:
            connection.execute("INSERT INTO a_pilots VALUES (?,?,?,?)", (plan.pilot_id, payload, plan_hash, initial_model.artifact_sha256))
    if progress:
        progress(f"Initial frozen evaluation: {4 * plan.evaluation_games_per_seat} games; every game retained")
    before, initial_runs = _evaluate(store, plan, initial_model, initial, "eval-initial", workers, guard=guard)
    if plan.strict_training_health and any(run["completed"] != run["planned_games"] for run in initial_runs):
        raise ExperimentError("strict initial development evaluation is incomplete; no training was started")
    model, training_catalog, training_runs, updates = initial_model, [], [], []
    for batch_index in range(start_batch, plan.batches):
        if progress:
            progress(f"Training batch {batch_index + 1}/{plan.batches}: {4 * plan.games_per_seat} games, frozen behavior snapshot")
        # On recovery the run layer reuses committed games. Recompute the pure
        # update from its parent and audited sources, then require byte-identical
        # evidence/child identity instead of trusting a checkpoint pointer alone.
        model, catalog, runs = _train_batch(store, plan, model, initial, batch_index, workers, guard=guard)
        training_catalog.extend(catalog)
        training_runs.extend(runs)
        evidence = _checked_json(*store._rows("SELECT evidence,evidence_hash FROM a_updates WHERE pilot_id=? AND batch_index=?", (plan.pilot_id, batch_index))[0])
        updates.append(evidence)
        if plan.strict_training_health and not evidence["learning_health"]["verified_score_updates"]:
            raise ExperimentError("strict training learning audit failed; checkpoint retained without further training")
        if progress:
            progress(f"Checkpoint committed: {model.training_games} independent training games; artifact={model.artifact_sha256}")
            health = evidence["learning_health"]
            progress(f"Learning audit: {health['requests_with_changed_mean_estimates']}/{health['descriptive_decision_count']} training requests changed estimates; "
                     f"{health['requests_with_changed_greedy_preferences']}/{health['multi_action_requests']} changed greedy preferences")
    if progress:
        progress(f"Final frozen evaluation: {4 * plan.evaluation_games_per_seat} held-out games; no learning")
    after, final_runs = _evaluate(store, plan, model, initial, "eval-final", workers, guard=guard)
    if ((plan.repetition_guard is not None or plan.evaluation_guard_scheme is not None) and plan.strict_training_health
            and any(run["completed"] != run["planned_games"] for run in final_runs)):
        raise ExperimentError("strict final guard evaluation is incomplete; no score improvement report")
    evaluation_catalog = before + after
    all_catalog = training_catalog + evaluation_catalog
    all_runs = initial_runs + training_runs + final_runs
    if len(all_catalog) != sum(run["planned_games"] for run in all_runs):
        raise ExperimentError("a played/planned game is missing from the retained catalog")
    appearances = {}
    for game in all_catalog:
        for snapshot_id in set(game["seat_snapshot_ids"]):
            appearances.setdefault(snapshot_id, {"games_appeared": 0, "train_games": 0, "evaluation_games": 0})
            appearances[snapshot_id]["games_appeared"] += 1
            appearances[snapshot_id]["train_games" if game["purpose"] == "train" else "evaluation_games"] += 1
    versions = []
    for snapshot_id, counts in sorted(appearances.items()):
        snapshot = store.read_snapshot(snapshot_id)
        versions.append({**counts, "snapshot_id": snapshot_id, "artifact_sha256": snapshot.artifact_sha256,
                         "model_version": snapshot.model_version, "training_games_consumed": snapshot.training_games,
                         "decision_mode": json.loads(snapshot.parameters_json)["mode"]})
    for run in (*initial_runs, *training_runs, *final_runs):
        run.pop("previously_committed_games", None)
        run.pop("newly_attempted_games", None)
    report = {"protocol": PILOT_PROTOCOL, "pilot_id": plan.pilot_id, "plan_hash": plan_hash,
              "source_archive_artifact_sha256": source_archive,
              "plan": plan.to_dict(), "model": model.summary(), "updates": updates,
              "training_game_records": len(training_catalog), "evaluation_game_records": len(evaluation_catalog),
              "retained_game_records": len(all_catalog), "versions_played": versions,
              "status": "completed" if all(run["completed"] == run["planned_games"] for run in all_runs) else "completed_with_failures",
              "training_runs": training_runs, "initial_evaluation_runs": initial_runs, "final_evaluation_runs": final_runs,
              "paired_evaluation": _paired_report(before, after, plan.master_seed),
              "training_selected_with_evidence": sum(game.get("focus_selected_with_evidence", 0) for game in training_catalog),
              "final_evaluation_selected_with_evidence": sum(game.get("focus_selected_with_evidence", 0) for game in after),
              "final_evaluation_decisions_with_learned_preferences": sum(game.get("mean_decisions_with_learned_preferences", 0) for game in after),
              "training_repetition_fallbacks": sum(game.get("focus_repetition_fallbacks", 0) for game in training_catalog),
              "evaluation_repetition_fallbacks": sum(game.get("focus_repetition_fallbacks", 0) for game in evaluation_catalog),
              "learning_health": [update["learning_health"] for update in updates],
              "parent_artifact_sha256": plan.parent_artifact_sha256,
              "fixed_opponent_snapshot_id": initial.snapshot_id,
              "inherited_training_games": initial_model.training_games,
              "new_training_games": model.training_games - initial_model.training_games,
              "cumulative_training_games": model.training_games,
              "every_played_game_retained": True, "failed_games_used_for_updates": 0,
              "evaluation_games_used_for_updates": 0, "model_B_started": False}
    if plan.source_semantics_certificate_sha256 is not None:
        report.update(protocol=plan.to_dict()["protocol"],
                      source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256,
                      historical_source_policy="verified original TRAIN lineage inherited; DIAGNOSTIC revalidations are not new samples",
                      semantic_diagnostic_games_used_as_training_samples=0)
    directory = store.root / "model-a-pilots" / plan_hash
    guard("before-report")
    for filename, catalog in (("training-games.jsonl", training_catalog), ("evaluation-games.jsonl", evaluation_catalog)):
        data = "".join(canonical_json(game) + "\n" for game in catalog).encode("utf-8")
        _write_immutable(directory / filename, data)
        report[filename.replace(".jsonl", "_sha256")] = hashlib.sha256(data).hexdigest()
    _write_immutable(directory / "report.json", canonical_json(report).encode("utf-8"))
    return {**report, "report_path": str(directory / "report.json"),
            "training_catalog_path": str(directory / "training-games.jsonl"),
            "evaluation_catalog_path": str(directory / "evaluation-games.jsonl")}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Logged model-A pilot: first-visit MC score returns and Thompson Sampling")
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--master-seed", type=int, default=20261029)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--games-per-seat", type=int, default=8)
    parser.add_argument("--evaluation-games-per-seat", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parent-artifact-sha256", help="Registered learned checkpoint to continue without resetting its knowledge")
    parser.add_argument("--opponent-snapshot-id", help="Exact existing frozen Thompson opponent snapshot, used in all three other seats")
    parser.add_argument("--source-semantics-certificate-sha256", help="Exact whole-parent current-source semantic compatibility certificate")
    parser.add_argument("--exploration-floor", type=float, default=0.05)
    parser.add_argument("--repetition-limit", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--max-events-per-race", type=int, default=10_000)
    parser.add_argument("--max-turns-per-race", type=int, default=2_000)
    parser.add_argument("--minimum-project-free-gib", type=int, default=30)
    parser.add_argument("--minimum-system-free-gib", type=int, default=5)
    parser.add_argument("--record-budget-mb", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        report = run_pilot(ExperimentStore(args.root), PilotPlan(args.pilot_id, args.master_seed, args.batches,
                            args.games_per_seat, args.evaluation_games_per_seat,
                            AConfig(exploration_floor=args.exploration_floor, repetition_limit=args.repetition_limit,
                                    min_samples=args.min_samples),
                            replace(pilot_config(), max_events_per_race=args.max_events_per_race,
                                    max_turns_per_race=args.max_turns_per_race),
                            parent_artifact_sha256=args.parent_artifact_sha256,
                            opponent_snapshot_id=args.opponent_snapshot_id,
                            source_semantics_certificate_sha256=args.source_semantics_certificate_sha256), workers=args.workers,
                           progress=lambda message: print(message, file=sys.stderr, flush=True),
                           resource_limits=ResourceLimits(
                               minimum_project_free_bytes=args.minimum_project_free_gib * 1024**3,
                               minimum_system_free_bytes=args.minimum_system_free_gib * 1024**3,
                               bytes_per_planned_record=args.record_budget_mb * 1_000_000))
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Model A pilot error: {exc}\n")
    print(json.dumps({key: report[key] for key in ("pilot_id", "model", "training_game_records", "evaluation_game_records",
                     "paired_evaluation", "final_evaluation_selected_with_evidence", "report_path",
                     "training_catalog_path", "evaluation_catalog_path")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
