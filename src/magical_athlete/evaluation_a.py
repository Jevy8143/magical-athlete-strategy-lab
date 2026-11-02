"""Immutable, independently seeded comparisons of two frozen model-A artifacts.

The focal policies use the full frozen mean/history/uniform mixture. Their
three opponents retain the specified *exact* Thompson snapshot identity.
Evaluation never trains a model, remaps opponent contexts, or replaces failed
games with invented final scores. Timing belongs to separate invocation logs.
"""

import argparse
import ast
import hashlib
import json
import random
import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .experiment_spec import Purpose, RunSpec, SeedPlan, canonical_json, content_hash
from .experiment_store import ExperimentError, ExperimentStore, _checked_json, _write_immutable, infrastructure_fingerprint, runtime_identity
from .experiments import run_experiment
from .game import GameConfig, GameEngine
from .model_a import FrozenAModel
from .policy_runtime import opponent_context_id
from .training_a import _archive_sources, _catalog, model_snapshot


EVALUATION_PROTOCOL = "frozen-a-seat-balanced-comparison-v1"
CERTIFIED_EVALUATION_PROTOCOL = "frozen-a-seat-balanced-certified-sources-v1"
TRANSFER_KIND = "frozen_policy_transfer_to_current_engine"
PROTECTED_POLICY_FILES = ("model_a.py", "model_a_features.py", "policy_runtime.py",
                          "experiment_records.py", "experiment_spec.py")
# An exact, reviewed data-contract extension. This is available only to the
# full TRAIN semantics certifier; ordinary frozen transfer remains byte-strict.
REVIEWED_SCHEMA_BRIDGE = "guarded-deployment-public-baseline-spec-extension-v1"
_REVIEWED_SPEC_PAIR = (
    "a5c2464fb7fdb12d59a9746d76af9751cde58ca8e423facdb4e8a7e7782ae7a6",
    "a3587805b4b84732d96fc015749984086805b7e8aedcdc511d7827c118becbc3",
)


@dataclass(frozen=True, slots=True)
class FrozenEvaluationPlan:
    evaluation_id: str
    left_artifact: str
    right_artifact: str
    opponent_snapshot_id: str
    games_per_seat: int = 32
    master_seed: int = 20261103
    panel_role: str = "confirmation"
    game_config: GameConfig | None = None
    origin_source_archive_sha256: str | None = None
    source_semantics_certificate_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.evaluation_id, str) or len(self.evaluation_id) > 48:
            raise ValueError("evaluation_id must be a portable identifier of at most 48 characters")
        if self.panel_role not in {"development", "confirmation"}:
            raise ValueError("panel_role must be development or confirmation")
        SeedPlan(self.master_seed, Purpose.EVALUATION, self.study_id)
        if type(self.games_per_seat) is not int or self.games_per_seat < 1:
            raise ValueError("games_per_seat must be a positive integer")
        for value in (self.left_artifact, self.right_artifact, self.opponent_snapshot_id):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("model artifacts and opponent snapshot must be SHA256 identities")
        if self.game_config is not None and (not isinstance(self.game_config, GameConfig) or not self.game_config.official):
            raise ValueError("evaluation requires verified official game data")
        if (self.origin_source_archive_sha256 is not None
                and (not isinstance(self.origin_source_archive_sha256, str)
                     or not re.fullmatch(r"[0-9a-f]{64}", self.origin_source_archive_sha256))):
            raise ValueError("origin source archive must be a SHA256 identity")
        if (self.source_semantics_certificate_sha256 is not None
                and (not isinstance(self.source_semantics_certificate_sha256, str)
                     or not re.fullmatch(r"[0-9a-f]{64}", self.source_semantics_certificate_sha256))):
            raise ValueError("source semantics certificate must be a SHA256 identity")
        if self.origin_source_archive_sha256 is not None and self.source_semantics_certificate_sha256 is not None:
            raise ValueError("read-only origin transfer and certified TRAIN semantics are mutually exclusive")

    @property
    def study_id(self):
        return f"{self.evaluation_id}-{self.panel_role}"

    def to_dict(self, game_config, *, origin_engine_fingerprint=None):
        result = {
            "protocol": EVALUATION_PROTOCOL, "evaluation_id": self.evaluation_id,
            "panel_role": self.panel_role, "study_id": self.study_id,
            "left_artifact": self.left_artifact, "right_artifact": self.right_artifact,
            "opponent_snapshot_id": self.opponent_snapshot_id, "games_per_seat": self.games_per_seat,
            "master_seed": self.master_seed, "focal_mode": "mean",
            "focal_mode_definition": "frozen posterior-mean/public-history/uniform mixture; not pure greedy",
            "retention_policy": "all_games_full_trace_and_summary",
            "game_config": json.loads(canonical_json(game_config)),
            "engine_fingerprint": GameEngine._code_fingerprint(),
            "infrastructure_fingerprint": infrastructure_fingerprint(), "runtime": runtime_identity(),
            "statistic": "equal-seat stratified paired-game bootstrap, 2000 replicates, 95% percentile interval",
            "failure_policy": "retain every attempt; never impute scores or replace failed pairs; no outcome-based stopping",
        }
        if self.origin_source_archive_sha256 is not None:
            result.update(origin_source_archive_sha256=self.origin_source_archive_sha256,
                          evaluation_kind=TRANSFER_KIND, origin_engine_fingerprint=origin_engine_fingerprint,
                          current_engine_fingerprint=GameEngine._code_fingerprint(),
                          old_training_data_used_for_current_engine_updates=False)
        if self.source_semantics_certificate_sha256 is not None:
            result.update(protocol=CERTIFIED_EVALUATION_PROTOCOL,
                          evaluation_kind="frozen_comparison_of_source_semantics_certified_models",
                          source_semantics_certificate_sha256=self.source_semantics_certificate_sha256,
                          historical_train_sources_eligible_via_exact_semantics_certificate=True,
                          original_training_game_ids_and_fingerprints_retained=True)
        return result


def _verified_origin_archive(store, digest, *, schema_bridge=None):
    """Authenticate archived bytes without executing any historical source."""
    if schema_bridge is not None and (not isinstance(schema_bridge, str) or schema_bridge != REVIEWED_SCHEMA_BRIDGE):
        raise ExperimentError("unsupported reviewed policy schema bridge")
    bundle = json.loads(store.read_artifact(digest))
    if bundle.get("kind") != "magical-athlete-source-v2" or not isinstance(bundle.get("files"), dict):
        raise ExperimentError("origin source archive has an unsupported bundle schema")
    raw_files = {}
    for name, record in bundle["files"].items():
        if (not isinstance(name, str) or not isinstance(record, dict)
                or set(record) != {"sha256", "utf8"} or not isinstance(record["utf8"], str)):
            raise ExperimentError("origin source archive has invalid file records")
        raw = record["utf8"].encode("utf-8")
        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ExperimentError("origin source archive file checksum differs")
        raw_files[name] = raw
    prefix = "src/magical_athlete/"
    try:
        tree = ast.parse(raw_files[prefix + "game.py"].decode("utf-8"))
        game_class, = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GameEngine"]
        fingerprint, = [node for node in game_class.body if isinstance(node, ast.FunctionDef) and node.name == "_code_fingerprint"]
        loop, = [node for node in fingerprint.body if isinstance(node, ast.For)]
        names = ast.literal_eval(loop.iter)
    except (KeyError, SyntaxError, ValueError, TypeError) as error:
        raise ExperimentError("origin GameEngine fingerprint file list is not a supported static constant") from error
    if (not isinstance(names, (tuple, list)) or not names
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*\.py", name) for name in names)):
        raise ExperimentError("origin GameEngine fingerprint file list is invalid")
    fingerprint_hash = hashlib.sha256()
    for name in names:
        if prefix + name not in raw_files:
            raise ExperimentError("origin GameEngine fingerprint requires a missing archived file")
        fingerprint_hash.update(name.encode("utf-8"))
        fingerprint_hash.update(raw_files[prefix + name])
    origin_fp = fingerprint_hash.hexdigest()
    if bundle.get("engine_fingerprint") != origin_fp:
        raise ExperimentError("origin GameEngine fingerprint differs from its archived source bytes")
    protected, bridge = {}, None
    for name in PROTECTED_POLICY_FILES:
        archived = raw_files.get(prefix + name)
        current = Path(__file__).with_name(name).read_bytes()
        if archived != current:
            pair = (hashlib.sha256(archived).hexdigest() if archived is not None else None,
                    hashlib.sha256(current).hexdigest())
            if not (name == "experiment_spec.py" and schema_bridge == REVIEWED_SCHEMA_BRIDGE
                    and pair == _REVIEWED_SPEC_PAIR):
                raise ExperimentError(f"origin protected policy/observation source differs: {name}")
            bridge = {"scheme": REVIEWED_SCHEMA_BRIDGE, "file": prefix + name,
                      "archived_sha256": pair[0], "current_sha256": pair[1],
                      "required_full_source_semantics": True,
                      "scope": "exact additive guarded-deployment and public-memory baseline schema; other protected policy/observation bytes remain identical"}
        protected[name] = hashlib.sha256(current).hexdigest()
    result = {"source_archive_sha256": digest, "engine_fingerprint": origin_fp,
              "protected_policy_files_sha256": protected, "fingerprint_file_names": list(names)}
    if bridge is not None:
        result["protected_policy_schema_bridge"] = bridge
    return result


def _source_conditions(store, models, *, origin=None, source_semantics_certificate_sha256=None):
    """Check actual TRAIN lineage without demanding historical infra equality.

    By default engine/rules/data match current execution. An authenticated
    optional origin permits only an explicit read-only frozen-policy transfer;
    a current-parser projection checks non-FP fields without changing history.
    """
    if source_semantics_certificate_sha256 is not None:
        if origin is not None:
            raise ExperimentError("read-only origin transfer and certified TRAIN semantics are mutually exclusive")
        from .source_compatibility import certified_source_conditions
        return certified_source_conditions(store, tuple(models), source_semantics_certificate_sha256)
    if origin is not None and "protected_policy_schema_bridge" in origin:
        raise ExperimentError("reviewed schema bridges require full TRAIN source semantics certification; read-only transfer is insufficient")
    checked_models, checking, checked_games, configs, model_cache = set(), set(), {}, {}, {}
    original_runs, original_games, parsed_runs = {}, {}, {}

    def source_view(run_id):
        if run_id in parsed_runs:
            return parsed_runs[run_id]
        manifest = store.manifest(run_id)
        original = manifest["spec"]
        expected_fp = origin["engine_fingerprint"] if origin is not None else GameEngine._code_fingerprint()
        if original["engine_fingerprint"] != expected_fp:
            raise ExperimentError("model TRAIN source engine differs from the explicit origin/current engine")
        # This object is only a schema/rules/config/data/seed metadata checker.
        # Its changed hash is never an original GameID, run, or replay claim.
        projected = {**original, "engine_fingerprint": GameEngine._code_fingerprint()}
        view = RunSpec.from_dict(projected)
        identity = SimpleNamespace(
            policies=view.policies, game_config=view.game_config, rule_version=original["rule_version"],
            engine_fingerprint=original["engine_fingerprint"], spec_id=manifest["spec_hash"],
            run_id=run_id, purpose=view.purpose, matchup=view.matchup,
            seat_snapshot_ids=view.seat_snapshot_ids, seed_plan=view.seed_plan,
        )
        original_runs[run_id] = {"run_id": run_id, "original_spec_hash": manifest["spec_hash"],
                                 "original_engine_fingerprint": original["engine_fingerprint"],
                                 "original_raw_spec": original,
                                 "planned_game_count": len(manifest["game_indices"]),
                                 "committed_status_counts": {row["status"]: row["n"] for row in store._rows(
                                     "SELECT status,COUNT(*) AS n FROM games WHERE run_id=? GROUP BY status", (run_id,))},
                                 "quarantined_game_count": store._rows(
                                     "SELECT COUNT(*) FROM quarantined_games WHERE run_id=?", (run_id,))[0][0]}
        original_runs[run_id]["uncommitted_game_count"] = (
            original_runs[run_id]["planned_game_count"]
            - sum(original_runs[run_id]["committed_status_counts"].values())
            - original_runs[run_id]["quarantined_game_count"])
        parsed_runs[run_id] = view, identity
        return view, identity

    def check(model):
        digest = model.artifact_sha256
        if digest in checking:
            raise ExperimentError("cyclic model source graph")
        if digest in checked_models:
            return
        checking.add(digest)
        model_cache[digest] = model
        prefix_games = []
        for batch_index, raw in enumerate(model.source_batches):
            batch = json.loads(raw)
            parent_digest = batch["parent_artifact_sha256"]
            if parent_digest not in checked_models:
                parent = FrozenAModel.from_bytes(store.read_artifact(parent_digest))
                if parent.artifact_sha256 != parent_digest:
                    raise ExperimentError("model parent artifact is not its canonical frozen identity")
                if parent.config != model.config:
                    raise ExperimentError("model source graph changes the frozen A configuration")
                check(parent)
            parent = model_cache[parent_digest]
            if (parent.config != model.config or parent.source_batches != model.source_batches[:batch_index]
                    or parent.training_game_ids != tuple(sorted(prefix_games))):
                raise ExperimentError("model source graph does not preserve its cumulative parent lineage")
            for source in batch["games"]:
                prefix_games.append(source["game_id"])
                if source["purpose"] != "train" or source["run_id"] is None:
                    raise ExperimentError("model requires explicitly committed TRAIN source games")
                spec, original_identity = source_view(source["run_id"])
                focus = source["focus_player_id"]
                opponents = [spec.policies[(focus + offset) % 4].snapshot_id for offset in (1, 2, 3)]
                from .policy_runtime_router import require_trainable_snapshot
                require_trainable_snapshot(spec.policies[focus])
                if (spec.purpose != Purpose.TRAIN or not spec.learning
                        or {snapshot.model_type for snapshot in spec.policies} != {"A"}
                        or spec.policies[focus].snapshot_id != source["behavior_snapshot_id"]
                        or spec.policies[focus].artifact_sha256 != parent_digest
                        or not spec.policies[focus].frozen
                        or json.loads(spec.policies[focus].parameters_json).get("mode") != "thompson"
                        or opponents != source["opponent_snapshot_ids"]
                        or opponent_context_id(spec, focus) != source["context_id"]):
                    raise ExperimentError("model TRAIN source condition differs from its committed run")
                previous = checked_games.get(source["game_id"])
                if previous is not None:
                    if previous != source:
                        raise ExperimentError("one source game has conflicting model lineage")
                    continue
                rows = store._rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (source["game_id"], spec.run_id))
                if len(rows) != 1 or rows[0]["status"] != "completed" or not rows[0]["replay_verified"]:
                    raise ExperimentError("model source is not one verified complete TRAIN game")
                records = store._load_records(rows[0])
                if not records["transitions"] or not records["game"]["completed"]:
                    raise ExperimentError("model TRAIN source has no complete player trajectories")
                row = rows[0]
                if source["game_id"] != f"{original_identity.spec_id}:{row['game_index']}":
                    raise ExperimentError("model TRAIN source original GameID differs from its raw spec")
                seeds = store.seeds_for(spec.run_id, row["game_index"])
                if seeds != spec.seed_plan.derive(row["game_index"]):
                    raise ExperimentError("model TRAIN source reserved seed plan differs")
                trace = json.loads(store.read_artifact(row["trace_hash"]))
                if not trace["completed"] or len(trace["race_traces"]) != 4:
                    raise ExperimentError("model TRAIN source lacks a complete original four-race trace")
                # Original identity/FP are used for trace checks and summary.
                # No original trace is relabelled or run on current physics.
                store._validate_trace_identity(original_identity, seeds, trace)
                rebuilt = store._records(trace, spec, seeds)
                rebuilt["game"] = store._summary(rebuilt, original_identity, seeds,
                                                  row["game_index"], "completed")
                if content_hash(rebuilt) != row["records_hash"]:
                    raise ExperimentError("model TRAIN source trace/behavior/records checksums differ")
                original_games[source["game_id"]] = {
                    "original_game_id": source["game_id"], "run_id": spec.run_id,
                    "game_index": row["game_index"], "focus_player_id": focus,
                    "original_spec_hash": original_identity.spec_id,
                    "original_engine_fingerprint": original_identity.engine_fingerprint,
                    "original_trace_sha256": row["trace_hash"],
                    "original_records_sha256": row["records_hash"],
                    "original_summary_sha256": row["summary_hash"],
                    "purpose": "train", "completed": True, "replay_verified_at_origin": True,
                }
                configs[canonical_json(spec.game_config)] = spec.game_config
                checked_games[source["game_id"]] = source
        checking.remove(digest)
        checked_models.add(digest)

    for model in models:
        check(model)
    audit = {"model_artifacts_checked": len(checked_models), "complete_train_games_checked": len(checked_games),
             "historical_infrastructure_policy": "retain historical metadata; require same engine/rules/data, not same infrastructure"}
    if origin is not None:
        failures = sum(sum(count for status, count in item["committed_status_counts"].items() if status != "completed")
                       + item["quarantined_game_count"] for item in original_runs.values())
        audit.update(evaluation_kind=TRANSFER_KIND, origin_source_archive_sha256=origin["source_archive_sha256"],
                     origin_engine_fingerprint=origin["engine_fingerprint"],
                     current_engine_fingerprint=GameEngine._code_fingerprint(),
                     protected_policy_files_sha256=origin["protected_policy_files_sha256"],
                     source_parser_projection="only engine_fingerprint replaced for non-FP validation; not an original run or replay",
                     original_runs=[original_runs[key] for key in sorted(original_runs)],
                     original_games=[original_games[key] for key in sorted(original_games)],
                     historical_training_planned_games=sum(item["planned_game_count"] for item in original_runs.values()),
                     historical_training_failures_not_used=failures,
                     historical_training_uncommitted_games=sum(item["uncommitted_game_count"] for item in original_runs.values()),
                     old_training_games_added_to_current_training=0,
                     original_source_games_replayed_under_current_engine=0,
                     historical_infrastructure_policy="explicit frozen-policy transfer only; original FP/IDs retained; same protected policy/rules/data required",
                     transfer_limitation="measures existing frozen strategies on current physics; does not establish lossless historical training or relabel old games as complete current-engine samples")
    return configs, audit


def _resolve_config(store, plan, conditions, *, origin=None):
    if plan.game_config is not None:
        return plan.game_config
    if len(conditions) == 1:
        return next(iter(conditions.values()))
    if not conditions:
        # Empty A0-vs-A0 calibration can use the exact old opponent's TRAIN
        # configuration. Never guess a budget that creates a new context.
        for row in store._rows("SELECT run_id FROM runs WHERE purpose='train'"):
            raw = store.manifest(row["run_id"])["spec"]
            if (raw["engine_fingerprint"] == (origin["engine_fingerprint"] if origin else GameEngine._code_fingerprint())
                    and plan.opponent_snapshot_id in [content_hash(item) for item in raw["policies"]]):
                spec = RunSpec.from_dict({**raw, "engine_fingerprint": GameEngine._code_fingerprint()})
                conditions[canonical_json(spec.game_config)] = spec.game_config
    if len(conditions) != 1:
        raise ExperimentError("training configuration is absent/ambiguous; provide one explicit official game_config")
    return next(iter(conditions.values()))


def _make_specs(plan, config, left, right, opponent):
    seed_plan = SeedPlan(plan.master_seed, Purpose.EVALUATION, plan.study_id)
    result = []
    for arm, current in (("left", left), ("right", right)):
        for seat in range(4):
            policies = [opponent] * 4
            policies[seat] = current
            spec = RunSpec(f"{plan.evaluation_id}-{plan.panel_role}-{arm}-s{seat}", plan.study_id,
                           Purpose.EVALUATION, tuple(policies), seed_plan, config, learning=False,
                           engine_fingerprint=GameEngine._code_fingerprint())
            indices = tuple(range(seat * plan.games_per_seat, (seat + 1) * plan.games_per_seat))
            result.append((arm, seat, spec, indices))
    return result


def _paired_evaluation(left, right, plan):
    maps = [{(row["focus_player_id"], row["game_index"]): row for row in arm} for arm in (left, right)]
    strata, by_seat, paired_left, paired_right, excluded = {}, {}, [], [], []
    for seat in range(4):
        totals_left, totals_right, differences = [], [], []
        for index in range(seat * plan.games_per_seat, (seat + 1) * plan.games_per_seat):
            l, r = (mapping.get((seat, index)) for mapping in maps)
            if l and r and l.get("completed") is True and r.get("completed") is True:
                if l["pairing_key"] != r["pairing_key"] or l["seeds"] != r["seeds"]:
                    raise ExperimentError("paired games have different reserved random streams")
                totals_left.append(l["focal_total"])
                totals_right.append(r["focal_total"])
                differences.append(r["focal_total"] - l["focal_total"])
                paired_left.append(l)
                paired_right.append(r)
            else:
                excluded.append({"focus_player_id": seat, "game_index": index,
                                 "left_status": l["status"] if l else "missing",
                                 "right_status": r["status"] if r else "missing"})
        strata[seat] = differences
        by_seat[str(seat)] = {"planned_pairs": plan.games_per_seat, "paired_games": len(differences),
                              "excluded_pairs": plan.games_per_seat - len(differences),
                              "initial_mean_total": statistics.fmean(totals_left) if differences else None,
                              "final_mean_total": statistics.fmean(totals_right) if differences else None,
                              "mean_paired_total_difference": statistics.fmean(differences) if differences else None}
    valid = all(strata.values())
    seed = int(content_hash({"statistic": "seat-stratified-paired-game-bootstrap-v1", "master_seed": plan.master_seed,
                             "study_id": plan.study_id}), 16) & ((1 << 63) - 1)
    interval = None
    if valid:
        rng = random.Random(seed)
        means = sorted(statistics.fmean(statistics.fmean(rng.choices(values, k=len(values)))
                                        for values in strata.values()) for _ in range(2000))
        interval = [means[49], means[1949]]
    paired = len(paired_left)
    return {
        "planned_pairs": 4 * plan.games_per_seat, "paired_games": paired, "excluded_pairs": len(excluded),
        "excluded_pair_records": excluded, "by_seat": by_seat,
        "initial_mean_total": statistics.fmean(item["initial_mean_total"] for item in by_seat.values()) if valid else None,
        "final_mean_total": statistics.fmean(item["final_mean_total"] for item in by_seat.values()) if valid else None,
        "mean_paired_total_difference": statistics.fmean(statistics.fmean(values) for values in strata.values()) if valid else None,
        "paired_bootstrap_95_percent_interval": interval, "bootstrap_replicates": 2000, "bootstrap_seed": seed,
        "pool_mean_paired_difference_descriptive": statistics.fmean(r["focal_total"] - l["focal_total"]
                                                                  for l, r in zip(paired_left, paired_right)) if paired else None,
        "independent_sample_unit": "paired game seed and focal seat; distinct seeds across seats; not four players per game",
        "estimand": "equally weighted four-seat difference in expected formal four-race focal total",
        "conditional_on_both_completed": bool(excluded),
        "conclusion": ("complete-pair conditional comparison only; failures/missing pairs prevent unconditional score claims"
                       if excluded else "independent frozen comparison under this opponent/seed distribution; no global-strength claim"),
        "all_four_seats_estimable": valid,
    }


def _coverage(store, specs):
    def counters():
        return {"requests": 0, "multi_action_requests": 0, "multi_action_selected_with_observations": 0,
                "multi_action_entirely_prior_only": 0, "candidate_actions": 0, "candidate_actions_without_observations": 0,
                "selected_observations_histogram": {}, "common_level_counts": {},
                "uniform_explorations": 0, "repetition_fallbacks": 0}

    result = {arm: {subset: {scope: counters() for scope in ("draft", "entry", "race")}
                    for subset in ("completed_only", "all_recorded")} for arm in ("left", "right")}
    for arm, seat, spec, _ in specs:
        for game in store._game_rows(spec.run_id, Purpose.EVALUATION):
            for row in store._rows("SELECT payload,checksum FROM decisions WHERE game_id=? AND player_id=? ORDER BY decision_seq",
                                   (game["game_id"], seat)):
                decision = _checked_json(row["payload"], row["checksum"])
                audit = decision["model_a_audit"]
                observations = [estimate["observations"] for estimate in audit["estimates"]]
                multi = len(observations) > 1
                for subset in (("all_recorded", "completed_only") if game["status"] == "completed" else ("all_recorded",)):
                    values = result[arm][subset][decision["scope"]]
                    values["requests"] += 1
                    values["multi_action_requests"] += multi
                    values["multi_action_selected_with_observations"] += multi and observations[audit["selected_index"]] > 0
                    values["multi_action_entirely_prior_only"] += multi and not any(observations)
                    values["candidate_actions"] += len(observations)
                    values["candidate_actions_without_observations"] += sum(value == 0 for value in observations)
                    selected = str(observations[audit["selected_index"]])
                    values["selected_observations_histogram"][selected] = values["selected_observations_histogram"].get(selected, 0) + 1
                    level = str(audit["common_level"])
                    values["common_level_counts"][level] = values["common_level_counts"].get(level, 0) + 1
                    values["uniform_explorations"] += audit["mixture_component"] == "uniform"
                    values["repetition_fallbacks"] += audit["mixture_component"] == "repetition"
    return result


def _arm_summary(catalog, planned):
    counts = {status: sum(row["status"] == status for row in catalog)
              for status in ("completed", "diagnostic", "error", "quarantined")}
    counts.update({"planned": planned, "missing": planned - len(catalog)})
    completed = [row for row in catalog if row.get("completed") is True]
    return {**counts, "failure_rate": sum(counts[key] for key in ("diagnostic", "error", "quarantined")) / planned,
            "missing_rate": counts["missing"] / planned,
            "completed_focal_mean_descriptive": statistics.fmean(row["focal_total"] for row in completed) if completed else None,
            "focal_co_winner_rate_completed_only": statistics.fmean(row["focal_co_winner"] for row in completed) if completed else None,
            "co_winner_counting": "each tied-highest focal player counts as one; four players are not independent games"}


def run_frozen_evaluation(store: ExperimentStore, plan: FrozenEvaluationPlan, *, workers=1, progress=None):
    started, wall_start = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    invocation = {"evaluation_id": plan.evaluation_id, "panel_role": plan.panel_role, "started_at_utc": started,
                  "workers": workers, "run_timings": [], "newly_attempted_games": 0, "status": "failed"}
    directory = store.root / "model-a-evaluations" / "failed-setup"
    try:
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ExperimentError("workers must be an integer from 1 through 64")
        artifact_ids = {plan.left_artifact, plan.right_artifact}
        before = {digest: store.read_artifact(digest) for digest in artifact_ids}
        models = {digest: FrozenAModel.from_bytes(raw) for digest, raw in before.items()}
        if any(model.artifact_sha256 != digest for digest, model in models.items()):
            raise ExperimentError("model artifact is not its canonical frozen identity")
        left_model, right_model = models[plan.left_artifact], models[plan.right_artifact]
        if left_model.config != right_model.config:
            raise ExperimentError("frozen comparison requires the same A decision/prior configuration")
        opponent = store.read_snapshot(plan.opponent_snapshot_id)
        if (opponent.model_type != "A" or not opponent.frozen
                or json.loads(opponent.parameters_json).get("mode") != "thompson" or opponent.training_games != 0):
            raise ExperimentError("opponent must be the explicit frozen initial A0 Thompson snapshot")
        opponent_raw = store.read_artifact(opponent.artifact_sha256)
        opponent_model = FrozenAModel.from_bytes(opponent_raw)
        if (opponent_model.artifact_sha256 != opponent.artifact_sha256
                or opponent_model.config != left_model.config or opponent_model.training_games != 0):
            raise ExperimentError("A0 opponent content/configuration differs from the frozen comparison")
        before[opponent.artifact_sha256] = opponent_raw
        models[opponent.artifact_sha256] = opponent_model
        origin = (_verified_origin_archive(store, plan.origin_source_archive_sha256)
                  if plan.origin_source_archive_sha256 is not None else None)
        conditions, source_audit = _source_conditions(store, (left_model, right_model), origin=origin,
            source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256)
        config = _resolve_config(store, plan, conditions, origin=origin)
        payload = plan.to_dict(config, origin_engine_fingerprint=origin["engine_fingerprint"] if origin else None)
        certificate_raw = (store.read_artifact(plan.source_semantics_certificate_sha256)
                           if plan.source_semantics_certificate_sha256 is not None else None)

        def check_certificate_binding():
            if certificate_raw is not None and (
                    GameEngine._code_fingerprint() != payload["engine_fingerprint"]
                    or infrastructure_fingerprint() != payload["infrastructure_fingerprint"]
                    or store.read_artifact(plan.source_semantics_certificate_sha256) != certificate_raw):
                raise ExperimentError("certified evaluation source/certificate changed; original records retained")

        plan_hash = content_hash(payload)
        directory = store.root / "model-a-evaluations" / plan_hash
        invocation["plan_hash"] = plan_hash
        source_archive = _archive_sources(store)
        with store._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS a_evaluations (
                evaluation_id TEXT PRIMARY KEY, plan TEXT NOT NULL, plan_hash TEXT NOT NULL,
                source_archive TEXT NOT NULL REFERENCES artifacts(sha256), report_hash TEXT REFERENCES artifacts(sha256))""")
            old = connection.execute("SELECT * FROM a_evaluations WHERE evaluation_id=?", (plan.evaluation_id,)).fetchone()
            if old is not None:
                if (_checked_json(old["plan"], old["plan_hash"]) != payload or old["source_archive"] != source_archive):
                    raise ExperimentError("evaluation plan/source is immutable; use a new evaluation_id")
            else:
                connection.execute("INSERT INTO a_evaluations VALUES (?,?,?,?,NULL)",
                                   (plan.evaluation_id, canonical_json(payload), plan_hash, source_archive))
        _write_immutable(directory / "plan.json", canonical_json(payload).encode("utf-8"))
        left_snapshot = model_snapshot(store, left_model, f"{plan.evaluation_id}-left", "mean")
        right_snapshot = model_snapshot(store, right_model, f"{plan.evaluation_id}-right", "mean")
        specs = _make_specs(plan, config, left_snapshot, right_snapshot, opponent)
        # Reserve every seat/arm and check all collisions before playing any.
        contexts = {"left": [], "right": []}
        for arm, seat, spec, indices in specs:
            store.create_run(spec, indices)
            context = opponent_context_id(spec, seat)
            rows = [row for row in models[spec.policies[seat].artifact_sha256].rows if row.context_id == context]
            contexts[arm].append({"focus_player_id": seat, "context_id": context,
                                  "trained_state_action_rows": len(rows), "cold_start": not rows,
                                  "cold_start_reason": ("untrained_reference" if not models[spec.policies[seat].artifact_sha256].training_games
                                                        else "opponent_context_not_in_checkpoint") if not rows else None})
        invocation["setup_wall_seconds"] = time.perf_counter() - wall_start
        catalogs, summaries = {"left": [], "right": []}, {"left": [], "right": []}
        for arm, seat, spec, indices in specs:
            check_certificate_binding()
            if progress:
                progress(f"Frozen {plan.panel_role}: {arm}, seat {seat}, {len(indices)} planned games; all attempts retained")
            stage = time.perf_counter()
            summary = run_experiment(store, spec, indices, workers=workers)
            run_seconds = time.perf_counter() - stage
            invocation["newly_attempted_games"] += summary["newly_attempted_games"]
            stage = time.perf_counter()
            store.audit_run(spec.run_id, replay=True)
            audit_seconds = time.perf_counter() - stage
            stage = time.perf_counter()
            catalogs[arm].extend(_catalog(store, spec, seat))
            invocation["run_timings"].append({"run_id": spec.run_id,
                "run_experiment_wall_seconds_including_simulation_validation_and_persistence": run_seconds,
                "full_post_audit_wall_seconds": audit_seconds, "catalog_wall_seconds": time.perf_counter() - stage})
            summaries[arm].append({key: value for key, value in summary.items()
                                   if key not in {"previously_committed_games", "newly_attempted_games"}})
        frozen = all(models[digest].to_bytes() == raw and store.read_artifact(digest) == raw for digest, raw in before.items())
        check_certificate_binding()
        if not frozen:
            raise ExperimentError("evaluation changed frozen model bytes")
        planned = 4 * plan.games_per_seat
        report = {"protocol": payload["protocol"], "evaluation_id": plan.evaluation_id, "panel_role": plan.panel_role,
                  "plan": payload, "plan_hash": plan_hash, "source_archive_artifact_sha256": source_archive,
                  "left_model": left_model.summary(), "right_model": right_model.summary(),
                  "opponent_snapshot": opponent.to_dict(), "source_audit": source_audit, "context_coverage": contexts,
                  "left_runs": summaries["left"], "right_runs": summaries["right"],
                  "arm_summary": {arm: _arm_summary(catalogs[arm], planned) for arm in catalogs},
                  "paired_evaluation": _paired_evaluation(catalogs["left"], catalogs["right"], plan),
                  "coverage": _coverage(store, specs),
                  "coverage_definition": "actual focal audits; candidate zero-observation counts use the selected common hierarchy level",
                  "retained_game_records": sum(len(catalog) for catalog in catalogs.values()),
                  "planned_game_records": 2 * planned, "every_played_game_retained": True,
                  "frozen_model_bytes_unchanged": frozen, "evaluation_games_used_for_updates": 0,
                  "model_updates_performed": 0,
                  "performance_scope": "timing is recorded only in separate immutable invocation files; commit timestamps are not simulation durations"}
        if origin is not None:
            report.update(evaluation_kind=TRANSFER_KIND,
                          origin_source_archive_sha256=origin["source_archive_sha256"],
                          origin_engine_fingerprint=origin["engine_fingerprint"],
                          current_engine_fingerprint=GameEngine._code_fingerprint(),
                          old_training_games_added_to_current_training=0,
                          historical_training_failures_not_used=source_audit["historical_training_failures_not_used"],
                          transfer_limitation=source_audit["transfer_limitation"],
                          model_artifacts_transferred_without_mutation=True)
        if plan.source_semantics_certificate_sha256 is not None:
            report.update(evaluation_kind=payload["evaluation_kind"],
                          source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256,
                          historical_train_sources_eligible_via_exact_semantics_certificate=True,
                          original_training_game_ids_and_fingerprints_retained=True,
                          source_semantics_scope="exact certified historical TRAIN prefix and strictly verified current TRAIN descendants; no blanket historical-FP exemption")
        report["status"] = ("incomplete" if any(value["missing"] for value in report["arm_summary"].values()) else
                            "completed" if all(value["completed"] == planned for value in report["arm_summary"].values()) else
                            "completed_with_failures")
        for arm, catalog in catalogs.items():
            data = "".join(canonical_json(game) + "\n" for game in catalog).encode("utf-8")
            _write_immutable(directory / f"{arm}-games.jsonl", data)
            report[f"{arm}_games_sha256"] = hashlib.sha256(data).hexdigest()
        packed = canonical_json(report).encode("utf-8")
        digest = store.put_artifact(packed)
        with store._transaction() as connection:
            existing = connection.execute("SELECT report_hash FROM a_evaluations WHERE evaluation_id=?", (plan.evaluation_id,)).fetchone()[0]
            if existing is not None and existing != digest:
                raise ExperimentError("recovered evaluation evidence differs from its immutable report")
            connection.execute("UPDATE a_evaluations SET report_hash=? WHERE evaluation_id=?", (digest, plan.evaluation_id))
        _write_immutable(directory / "report.json", packed)
        invocation["status"] = report["status"]
        return {**report, "report_path": str(directory / "report.json"),
                "left_catalog_path": str(directory / "left-games.jsonl"), "right_catalog_path": str(directory / "right-games.jsonl"),
                "invocation_path": str(directory / "invocations" / (invocation_id := uuid.uuid4().hex + ".json"))}
    except BaseException as error:
        invocation["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        invocation["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        invocation["wall_seconds"] = time.perf_counter() - wall_start
        filename = locals().get("invocation_id", uuid.uuid4().hex + ".json")
        _write_immutable(directory / "invocations" / filename, canonical_json(invocation).encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--left-artifact", required=True)
    parser.add_argument("--right-artifact", required=True)
    parser.add_argument("--opponent-snapshot-id", required=True)
    parser.add_argument("--games-per-seat", type=int, default=32)
    parser.add_argument("--master-seed", type=int, default=20261103)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--panel-role", choices=("development", "confirmation"), default="confirmation")
    parser.add_argument("--origin-source-archive-sha256", help="Explicit authenticated historical policy source for read-only transfer evaluation")
    parser.add_argument("--source-semantics-certificate-sha256", help="Exact per-Game current-semantics certificate for eligible historical TRAIN lineage")
    args = parser.parse_args(argv)
    store = ExperimentStore(args.root)
    plan = FrozenEvaluationPlan(args.evaluation_id, args.left_artifact, args.right_artifact, args.opponent_snapshot_id,
                                args.games_per_seat, args.master_seed, args.panel_role,
                                origin_source_archive_sha256=args.origin_source_archive_sha256,
                                source_semantics_certificate_sha256=args.source_semantics_certificate_sha256)
    report = run_frozen_evaluation(store, plan, workers=args.workers, progress=lambda message: print(message, flush=True))
    print(json.dumps({key: report[key] for key in ("evaluation_id", "status", "retained_game_records", "paired_evaluation", "report_path")}, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
