"""Auditable frozen A comparisons under explicitly changed opponents.

The lookup context is an immutable part of each deployed policy. Actual
opponents, random streams, and public observations remain the real ones.
These EVAL games never become training sources or silently pooled contexts.
"""

import argparse
import hashlib
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from .evaluation_a import _arm_summary, _coverage, _paired_evaluation, _source_conditions
from .experiment_spec import Purpose, RunSpec, SeedPlan, canonical_json, content_hash
from .experiment_store import ExperimentError, ExperimentStore, _checked_json, _write_immutable, infrastructure_fingerprint, runtime_identity
from .experiments import baseline_snapshot, run_experiment
from .game import GameConfig, GameEngine
from .model_a import FrozenAModel
from .policy_runtime import opponent_context_id
from .training_a import _archive_sources, _catalog


BENCHMARK_PROTOCOL = "frozen-a-fixed-context-opponent-benchmark-v1"
CERTIFIED_BENCHMARK_PROTOCOL = "frozen-a-fixed-context-certified-sources-benchmark-v1"
GUARDED_BENCHMARK_PROTOCOL = "frozen-a-guarded-fixed-context-opponent-benchmark-v1"
CERTIFIED_GUARDED_BENCHMARK_PROTOCOL = "frozen-a-guarded-fixed-context-certified-sources-benchmark-v1"


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 identity")


def _identifier(value, name, maximum):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0," + str(maximum - 1) + "}", value):
        raise ValueError(f"{name} must be a portable identifier of at most {maximum} characters")


@dataclass(frozen=True, slots=True)
class BenchmarkPanel:
    panel_id: str
    opponent_kind: str
    opponent_artifact: str | None = None
    opponent_lookup_context_id: str | None = None
    opponent_mode: str = "thompson"
    opponent_source_snapshot_id: str | None = None

    def __post_init__(self):
        _identifier(self.panel_id, "panel_id", 18 if self.opponent_kind == "first_legal_memory" else 16)
        if self.opponent_kind not in {"random", "first_legal", "first_legal_memory", "frozen_a"}:
            raise ValueError("opponent_kind must be random, first_legal, first_legal_memory or frozen_a")
        if self.opponent_mode not in {"mean", "thompson"}:
            raise ValueError("opponent_mode must be mean or thompson")
        if self.opponent_kind == "frozen_a":
            _sha(self.opponent_artifact, "opponent_artifact")
            _sha(self.opponent_lookup_context_id, "opponent_lookup_context_id")
            if self.opponent_source_snapshot_id is not None:
                _sha(self.opponent_source_snapshot_id, "opponent_source_snapshot_id")
        elif self.opponent_artifact is not None or self.opponent_lookup_context_id is not None:
            raise ValueError("a baseline panel cannot declare learned model artifacts or contexts")
        elif self.opponent_source_snapshot_id is not None:
            raise ValueError("a baseline panel cannot declare a learned source snapshot")
        elif self.opponent_mode != "thompson":
            raise ValueError("baseline panels do not support a model decision mode")

    def to_dict(self):
        result = asdict(self)
        if self.opponent_source_snapshot_id is None:
            result.pop("opponent_source_snapshot_id")
        return result


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkPlan:
    benchmark_id: str
    left_artifact: str
    right_artifact: str
    left_lookup_context_id: str
    right_lookup_context_id: str
    panels: tuple[BenchmarkPanel, ...] = (
        BenchmarkPanel("random", "random"), BenchmarkPanel("first-legal", "first_legal"),
    )
    games_per_seat: int = 16
    master_seed: int = 20261220
    panel_role: str = "development"
    primary_panel_id: str | None = None
    game_config: GameConfig | None = None
    source_semantics_certificate_sha256: str | None = None
    left_source_snapshot_id: str | None = None
    right_source_snapshot_id: str | None = None

    def __post_init__(self):
        _identifier(self.benchmark_id, "benchmark_id", 40)
        for name in ("left_artifact", "right_artifact", "left_lookup_context_id", "right_lookup_context_id"):
            _sha(getattr(self, name), name)
        if self.source_semantics_certificate_sha256 is not None:
            _sha(self.source_semantics_certificate_sha256, "source_semantics_certificate_sha256")
        if (self.left_source_snapshot_id is None) != (self.right_source_snapshot_id is None):
            raise ValueError("guarded benchmark requires both left and right source snapshot IDs")
        if self.left_source_snapshot_id is not None:
            _sha(self.left_source_snapshot_id, "left_source_snapshot_id")
            _sha(self.right_source_snapshot_id, "right_source_snapshot_id")
        object.__setattr__(self, "panels", tuple(self.panels))
        if not self.panels or not all(isinstance(panel, BenchmarkPanel) for panel in self.panels):
            raise ValueError("at least one explicit BenchmarkPanel is required")
        if len({panel.panel_id for panel in self.panels}) != len(self.panels):
            raise ValueError("benchmark panel IDs must be unique")
        for panel in self.panels:
            if panel.opponent_kind == "frozen_a":
                if self.uses_guarded_deployment != (panel.opponent_source_snapshot_id is not None):
                    raise ValueError("guarded frozen-A panels require explicit focal and opponent source snapshots")
        if type(self.games_per_seat) is not int or self.games_per_seat < 1:
            raise ValueError("games_per_seat must be a positive integer")
        if self.panel_role not in {"development", "confirmation"}:
            raise ValueError("panel_role must be development or confirmation")
        if self.primary_panel_id is not None and self.primary_panel_id not in {panel.panel_id for panel in self.panels}:
            raise ValueError("primary_panel_id must name one declared panel")
        if self.panel_role == "confirmation" and self.primary_panel_id is None:
            raise ValueError("confirmation must predeclare a primary_panel_id")
        if self.game_config is not None and (not isinstance(self.game_config, GameConfig) or not self.game_config.official):
            raise ValueError("benchmark requires verified official game data")
        for panel in self.panels:
            SeedPlan(self.master_seed, Purpose.EVALUATION, self.study_id(panel))

    def study_id(self, panel):
        return f"{self.benchmark_id}-{self.panel_role}-{panel.panel_id}"

    @property
    def uses_guarded_deployment(self):
        return self.left_source_snapshot_id is not None

    def to_dict(self, config):
        result = {
            "protocol": BENCHMARK_PROTOCOL, "benchmark_id": self.benchmark_id,
            "left_artifact": self.left_artifact, "right_artifact": self.right_artifact,
            "left_lookup_context_id": self.left_lookup_context_id,
            "right_lookup_context_id": self.right_lookup_context_id,
            "panels": [panel.to_dict() for panel in self.panels],
            "games_per_seat": self.games_per_seat, "master_seed": self.master_seed,
            "panel_role": self.panel_role, "primary_panel_id": self.primary_panel_id,
            "game_config": json.loads(canonical_json(config)),
            "engine_fingerprint": GameEngine._code_fingerprint(),
            "infrastructure_fingerprint": infrastructure_fingerprint(),
            "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": runtime_identity(), "purpose": "evaluation", "learning": False,
            "focal_mode": "mean", "focal_mode_definition": "frozen posterior-mean/public-history/uniform mixture; not pure greedy",
            "deployment_definition": "fixed trained lookup context; actual opponents and public observations are not relabelled",
            "context_coverage_definition": "bound context already has trained rows; unseen individual state/action estimates still use declared priors",
            "independent_sample_unit": "paired full-game seed and focal seat, with distinct seed blocks per seat and panel",
            "statistic": "equal-seat stratified paired-game bootstrap, 2000 replicates, 95% percentile interval, per panel",
            "multiplicity_policy": "only the predeclared primary panel is confirmatory; all other panels exploratory; no pooled strength claim",
            "failure_policy": "retain every attempt; no zero-imputation, replacement retries or outcome-driven stopping",
            "retention_policy": "all_games_full_trace_and_summary",
        }
        if self.uses_guarded_deployment:
            result.update(protocol=GUARDED_BENCHMARK_PROTOCOL,
                          left_source_snapshot_id=self.left_source_snapshot_id,
                          right_source_snapshot_id=self.right_source_snapshot_id,
                          guarded_deployment_definition="preserve each explicitly registered source guard; deployed mode is separately frozen; real actual contexts and trained lookup sources remain distinct")
        if self.source_semantics_certificate_sha256 is not None:
            result.update(protocol=(CERTIFIED_GUARDED_BENCHMARK_PROTOCOL if self.uses_guarded_deployment
                                    else CERTIFIED_BENCHMARK_PROTOCOL),
                          source_semantics_certificate_sha256=self.source_semantics_certificate_sha256,
                          historical_train_sources_eligible_via_exact_semantics_certificate=True,
                          original_training_game_ids_and_fingerprints_retained=True)
        return result


def _verified_models(store, digests):
    raw = {digest: store.read_artifact(digest) for digest in digests}
    models = {digest: FrozenAModel.from_bytes(value) for digest, value in raw.items()}
    if any(model.artifact_sha256 != digest for digest, model in models.items()):
        raise ExperimentError("benchmark model artifact is not its canonical identity")
    return raw, models


def _guarded_source_snapshots(store, plan, models, raw_models):
    """Authenticate explicit registered native policies before any benchmark play.

    The deployment factory remains a second independent registration boundary.
    These records bind the original mode and guard even when the benchmark
    explicitly freezes a different evaluation mode for the same weights.
    """
    if not plan.uses_guarded_deployment:
        return {}
    from .policy_runtime_router import load_model, require_implemented

    declarations = [("left", plan.left_artifact, plan.left_source_snapshot_id),
                    ("right", plan.right_artifact, plan.right_source_snapshot_id)]
    declarations.extend(("opponent:" + panel.panel_id, panel.opponent_artifact, panel.opponent_source_snapshot_id)
                        for panel in plan.panels if panel.opponent_kind == "frozen_a")
    snapshots = {}
    for role, model_id, snapshot_id in declarations:
        snapshot = store.read_snapshot(snapshot_id)
        parameters = json.loads(snapshot.parameters_json)
        if (snapshot.snapshot_id != snapshot_id or snapshot.model_type != "A"
                or snapshot.policy_kind != "model_artifact" or not snapshot.frozen
                or snapshot.artifact_sha256 != model_id
                or snapshot.training_games != models[model_id].training_games
                or set(parameters) != {"algorithm", "feature_version", "mode", "repetition_guard"}):
            raise ExperimentError("guarded benchmark source must be the exact registered native guarded model snapshot")
        require_implemented(snapshot)
        decoded = load_model(snapshot, {model_id: raw_models[model_id]})
        if decoded.to_bytes() != raw_models[model_id] or decoded.config != models[model_id].config:
            raise ExperimentError("guarded benchmark source weights/configuration differ from its frozen model")
        snapshots[role] = snapshot
    return snapshots


def _reconstruct_train_parameters(store, models):
    """Pure verification against existing TRAIN, never a new persisted update."""
    verified = set()

    def verify(model):
        if model.artifact_sha256 in verified:
            return
        if not model.source_batches:
            expected = FrozenAModel.empty(model.config)
        else:
            batch = json.loads(model.source_batches[-1])
            parent = FrozenAModel.from_bytes(store.read_artifact(batch["parent_artifact_sha256"]))
            verify(parent)
            sources = []
            for source in batch["games"]:
                row, = store._rows("SELECT games.*,runs.purpose AS source_purpose FROM games JOIN runs USING(run_id) WHERE game_id=? AND run_id=?",
                                   (source["game_id"], source["run_id"]))
                records = store._load_records(row)
                if source["purpose"] != "train" or row["source_purpose"] != "train":
                    raise ExperimentError("parameter verification accepts only committed TRAIN sources")
                sources.append({**source, "transitions": [{**item, "purpose": "train"} for item in records["transitions"]
                                                          if item["player_id"] == source["focus_player_id"]]})
            expected = parent.update_games(sources, batch_id=batch["batch_id"])
        if expected.to_bytes() != model.to_bytes():
            raise ExperimentError("benchmark model parameters do not reconstruct from their verified TRAIN lineage")
        verified.add(model.artifact_sha256)

    for model in models:
        verify(model)
    return {"model_artifacts_reconstructed": len(verified), "verification_sources": "existing verified TRAIN only",
            "parameter_reconstruction_persisted_updates": 0}


def _specs(plan, config, left, right, opponents):
    result = []
    for panel in plan.panels:
        seeds = SeedPlan(plan.master_seed, Purpose.EVALUATION, plan.study_id(panel))
        for arm, focal in (("left", left), ("right", right)):
            for seat in range(4):
                policies = [opponents[panel.panel_id]] * 4
                policies[seat] = focal
                spec = RunSpec(f"{plan.study_id(panel)}-{arm}-s{seat}", plan.study_id(panel), Purpose.EVALUATION,
                               tuple(policies), seeds, config, learning=False)
                indices = tuple(range(seat * plan.games_per_seat, (seat + 1) * plan.games_per_seat))
                result.append((panel, arm, seat, spec, indices))
    return result


def run_frozen_benchmark(store: ExperimentStore, plan: FrozenBenchmarkPlan, *, workers=1, progress=None):
    from .model_a_deployment import fixed_context_snapshot, guarded_fixed_context_snapshot

    started = time.perf_counter()
    invocation = {"benchmark_id": plan.benchmark_id, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "workers": workers, "status": "failed", "newly_attempted_games": 0, "run_timings": []}
    directory = store.root / "model-a-benchmarks" / "failed-setup"
    invocation_id = uuid.uuid4().hex + ".json"
    try:
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ExperimentError("workers must be an integer from 1 through 64")
        digests = {plan.left_artifact, plan.right_artifact,
                   *(panel.opponent_artifact for panel in plan.panels if panel.opponent_artifact is not None)}
        before, models = _verified_models(store, digests)
        left, right = models[plan.left_artifact], models[plan.right_artifact]
        if left.config != right.config:
            raise ExperimentError("benchmark focal models must share their A configuration")
        bindings = [(left, plan.left_lookup_context_id), (right, plan.right_lookup_context_id)]
        bindings.extend((models[panel.opponent_artifact], panel.opponent_lookup_context_id)
                        for panel in plan.panels if panel.opponent_kind == "frozen_a")
        if any(model.training_games <= 0 or not any(row.context_id == context for row in model.rows)
               for model, context in bindings):
            raise ExperimentError("every deployed benchmark model requires a genuinely trained lookup context")
        source_snapshots = _guarded_source_snapshots(store, plan, models, before)
        # No historical origin bypass: all sources must match current physics.
        conditions, source_audit = _source_conditions(store, tuple(models.values()),
            source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256)
        certificate_raw = (store.read_artifact(plan.source_semantics_certificate_sha256)
                           if plan.source_semantics_certificate_sha256 is not None else None)
        parameter_audit = _reconstruct_train_parameters(store, tuple(models.values()))
        config = plan.game_config
        if config is None:
            if len(conditions) != 1:
                raise ExperimentError("benchmark training configuration is ambiguous; provide explicit official game_config")
            config = next(iter(conditions.values()))
        base_payload = plan.to_dict(config)

        def deploy(model, version, lookup, role, mode):
            if source_snapshots:
                return guarded_fixed_context_snapshot(store, model, version, lookup,
                                                      source_snapshots[role], mode=mode)
            return fixed_context_snapshot(store, model, version, lookup, mode=mode)

        focals = [deploy(model, f"{plan.benchmark_id}-{arm}", context, arm, "mean")
                  for arm, model, context in (("left", left, plan.left_lookup_context_id),
                                              ("right", right, plan.right_lookup_context_id))]
        opponents = {}
        for panel in plan.panels:
            if panel.opponent_kind == "frozen_a":
                opponent = deploy(models[panel.opponent_artifact],
                    f"{plan.benchmark_id}-{panel.panel_id}-opponent", panel.opponent_lookup_context_id,
                    "opponent:" + panel.panel_id, panel.opponent_mode)
            else:
                opponent = baseline_snapshot(panel.opponent_kind)
                store.register_snapshot(opponent)
            opponents[panel.panel_id] = opponent
        specs = _specs(plan, config, *focals, opponents)
        contexts = {panel.panel_id: {arm: [] for arm in ("left", "right")} for panel in plan.panels}
        for panel, arm, seat, spec, indices in specs:
            model = left if arm == "left" else right
            lookup = plan.left_lookup_context_id if arm == "left" else plan.right_lookup_context_id
            actual = opponent_context_id(spec, seat)
            coverage = {"focus_player_id": seat, "actual_context_id": actual,
                "lookup_context_id": lookup, "cross_condition_deployment": actual != lookup,
                "trained_lookup_rows": sum(row.context_id == lookup for row in model.rows),
                "cold_start": False, "actual_opponent_snapshot_ids": [spec.policies[(seat + i) % 4].snapshot_id for i in (1, 2, 3)]}
            if source_snapshots:
                coverage.update(source_snapshot_id=source_snapshots[arm].snapshot_id,
                                deployment_snapshot_id=spec.policies[seat].snapshot_id,
                                repetition_guard=json.loads(source_snapshots[arm].parameters_json)["repetition_guard"])
            contexts[panel.panel_id][arm].append(coverage)
        profile = {
            "focal_snapshots": {arm: snapshot.to_dict() for arm, snapshot in zip(("left", "right"), focals)},
            "focal_snapshot_ids": {arm: snapshot.snapshot_id for arm, snapshot in zip(("left", "right"), focals)},
            "panels": {panel.panel_id: {
                "opponent_snapshot": opponents[panel.panel_id].to_dict(),
                "opponent_snapshot_id": opponents[panel.panel_id].snapshot_id,
                "focal_contexts": contexts[panel.panel_id],
                "runs": [{"arm": arm, "focus_player_id": seat, "run_spec": spec.to_dict(),
                          "game_indices": list(indices)}
                         for item, arm, seat, spec, indices in specs if item == panel],
            } for panel in plan.panels},
        }
        if source_snapshots:
            deployed_by_role = {**dict(zip(("left", "right"), focals)),
                                **{"opponent:" + panel.panel_id: opponents[panel.panel_id]
                                   for panel in plan.panels if panel.opponent_kind == "frozen_a"}}
            profile["guarded_policy_source_bindings"] = {
                role: {"source_snapshot_id": snapshot.snapshot_id, "source_snapshot": snapshot.to_dict(),
                       "source_model_artifact_sha256": snapshot.artifact_sha256,
                       "source_mode": json.loads(snapshot.parameters_json)["mode"],
                       "repetition_guard": json.loads(snapshot.parameters_json)["repetition_guard"],
                       "deployed_snapshot_id": deployed_by_role[role].snapshot_id,
                       "deployed_mode": json.loads(deployed_by_role[role].parameters_json)["mode"],
                       "lookup_context_id": json.loads(deployed_by_role[role].parameters_json)["deployment"]["lookup_context_id"]}
                for role, snapshot in source_snapshots.items()}
        profile_raw = canonical_json(profile).encode("utf-8")
        profile_sha = store.put_artifact(profile_raw)
        payload = {**base_payload, "deployment_profile_artifact_sha256": profile_sha}
        plan_hash = content_hash(payload)
        directory = store.root / "model-a-benchmarks" / plan_hash
        invocation["plan_hash"] = plan_hash
        source_archive = _archive_sources(store)
        with store._transaction() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS a_benchmarks (
                benchmark_id TEXT PRIMARY KEY, plan TEXT NOT NULL, plan_hash TEXT NOT NULL,
                source_archive TEXT NOT NULL REFERENCES artifacts(sha256), report_hash TEXT REFERENCES artifacts(sha256))""")
            old = connection.execute("SELECT * FROM a_benchmarks WHERE benchmark_id=?", (plan.benchmark_id,)).fetchone()
            if old is None:
                connection.execute("INSERT INTO a_benchmarks VALUES (?,?,?,?,NULL)",
                                   (plan.benchmark_id, canonical_json(payload), plan_hash, source_archive))
            elif _checked_json(old["plan"], old["plan_hash"]) != payload or old["source_archive"] != source_archive:
                raise ExperimentError("benchmark plan/source is immutable; use a new benchmark_id")
        _write_immutable(directory / "plan.json", canonical_json(payload).encode("utf-8"))
        _write_immutable(directory / "deployment-profile.json", profile_raw)
        # Reserve all panels/arms before simulating any game, including collisions.
        for panel, arm, seat, spec, indices in specs:
            store.create_run(spec, indices)
        invocation["setup_wall_seconds"] = time.perf_counter() - started
        catalogs = {panel.panel_id: {arm: [] for arm in ("left", "right")} for panel in plan.panels}
        summaries = {panel.panel_id: {arm: [] for arm in ("left", "right")} for panel in plan.panels}

        def check_guarded_sources():
            for snapshot in source_snapshots.values():
                if (store.read_snapshot(snapshot.snapshot_id) != snapshot
                        or store.read_artifact(snapshot.snapshot_id) != canonical_json(snapshot.to_dict()).encode("utf-8")):
                    raise ExperimentError("guarded benchmark source snapshot changed during execution; existing games retained")

        for panel, arm, seat, spec, indices in specs:
            if (plan.to_dict(config) != base_payload or certificate_raw is not None
                    and store.read_artifact(plan.source_semantics_certificate_sha256) != certificate_raw):
                raise ExperimentError("benchmark source changed during execution; existing games retained")
            check_guarded_sources()
            if progress:
                progress(f"Frozen benchmark {panel.panel_id}: {arm}, seat {seat}, {len(indices)} planned games")
            run_started = time.perf_counter()
            summary = run_experiment(store, spec, indices, workers=workers)
            invocation["newly_attempted_games"] += summary["newly_attempted_games"]
            store.audit_run(spec.run_id, replay=True)
            catalogs[panel.panel_id][arm].extend(_catalog(store, spec, seat))
            summaries[panel.panel_id][arm].append({key: value for key, value in summary.items()
                                                   if key not in {"previously_committed_games", "newly_attempted_games"}})
            invocation["run_timings"].append({"run_id": spec.run_id,
                "simulation_validation_audit_catalog_wall_seconds": time.perf_counter() - run_started})
        check_guarded_sources()
        if (plan.to_dict(config) != base_payload or certificate_raw is not None
                and store.read_artifact(plan.source_semantics_certificate_sha256) != certificate_raw
                or any(store.read_artifact(digest) != raw or models[digest].to_bytes() != raw
                                                 for digest, raw in before.items())):
            raise ExperimentError("benchmark changed source or frozen model bytes")
        panel_reports = {}
        planned_arm = 4 * plan.games_per_seat
        for panel in plan.panels:
            panel_specs = [(arm, seat, spec, indices) for item, arm, seat, spec, indices in specs if item == panel]
            statistic_plan = SimpleNamespace(games_per_seat=plan.games_per_seat, master_seed=plan.master_seed,
                                             study_id=plan.study_id(panel))
            panel_report = {"panel": panel.to_dict(), "opponent_snapshot": opponents[panel.panel_id].to_dict(),
                "opponent_snapshot_id": opponents[panel.panel_id].snapshot_id,
                "arm_summary": {arm: _arm_summary(catalogs[panel.panel_id][arm], planned_arm) for arm in ("left", "right")},
                "left_runs": summaries[panel.panel_id]["left"], "right_runs": summaries[panel.panel_id]["right"],
                "paired_evaluation": _paired_evaluation(catalogs[panel.panel_id]["left"], catalogs[panel.panel_id]["right"], statistic_plan),
                "coverage": _coverage(store, panel_specs), "context_coverage": contexts[panel.panel_id],
                "inference_role": "primary" if plan.panel_role == "confirmation" and panel.panel_id == plan.primary_panel_id else "exploratory"}
            for arm in ("left", "right"):
                raw = "".join(canonical_json(row) + "\n" for row in catalogs[panel.panel_id][arm]).encode("utf-8")
                filename = f"{panel.panel_id}-{arm}-games.jsonl"
                _write_immutable(directory / filename, raw)
                panel_report[f"{arm}_catalog_file"] = filename
                panel_report[f"{arm}_catalog_sha256"] = hashlib.sha256(raw).hexdigest()
            panel_reports[panel.panel_id] = panel_report
        values = [arm for panel in panel_reports.values() for arm in panel["arm_summary"].values()]
        status = ("incomplete" if any(arm["missing"] for arm in values) else
                  "completed" if all(arm["completed"] == planned_arm for arm in values) else "completed_with_failures")
        report = {"protocol": payload["protocol"], "benchmark_id": plan.benchmark_id, "plan": payload,
            "plan_hash": plan_hash, "source_archive_artifact_sha256": source_archive,
            "left_model": left.summary(), "right_model": right.summary(), "source_audit": source_audit,
            "parameter_lineage_audit": parameter_audit, "panels": panel_reports,
            "focal_deployment_snapshots": {arm: snapshot.to_dict() for arm, snapshot in zip(("left", "right"), focals)},
            "status": status, "planned_game_records": 2 * planned_arm * len(plan.panels),
            "retained_game_records": sum(len(rows) for panel in catalogs.values() for rows in panel.values()),
            "every_played_game_retained": True, "model_updates_performed": 0, "evaluation_games_used_for_updates": 0,
            "frozen_model_bytes_unchanged": True,
            "limitation": "explicit cross-opponent deployment using fixed trained lookup estimates; not same-condition training, automatic adaptation, pooled strength or global optimality",
            "timing_scope": "separate invocation files only; timing never changes deterministic plan/game/model/report bytes"}
        if plan.source_semantics_certificate_sha256 is not None:
            report.update(source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256,
                          historical_train_sources_eligible_via_exact_semantics_certificate=True,
                          original_training_game_ids_and_fingerprints_retained=True,
                          source_semantics_scope="all focal and frozen-opponent source graphs require the exact certified historical prefix or strictly verified current TRAIN descendants")
        if source_snapshots:
            report["guarded_policy_source_bindings"] = profile["guarded_policy_source_bindings"]
        raw_report = canonical_json(report).encode("utf-8")
        digest = store.put_artifact(raw_report)
        with store._transaction() as connection:
            old_hash = connection.execute("SELECT report_hash FROM a_benchmarks WHERE benchmark_id=?", (plan.benchmark_id,)).fetchone()[0]
            if old_hash is not None and old_hash != digest:
                raise ExperimentError("recovered benchmark report differs from its immutable evidence")
            connection.execute("UPDATE a_benchmarks SET report_hash=? WHERE benchmark_id=?", (digest, plan.benchmark_id))
        _write_immutable(directory / "report.json", raw_report)
        invocation["status"] = status
        return {**report, "report_path": str(directory / "report.json"),
                "invocation_path": str(directory / "invocations" / invocation_id)}
    except BaseException as error:
        invocation["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        invocation["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        invocation["wall_seconds"] = time.perf_counter() - started
        _write_immutable(directory / "invocations" / invocation_id, canonical_json(invocation).encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--left-artifact", required=True)
    parser.add_argument("--right-artifact", required=True)
    parser.add_argument("--left-lookup-context-id", required=True)
    parser.add_argument("--right-lookup-context-id", required=True)
    parser.add_argument("--frozen-opponent-artifact")
    parser.add_argument("--frozen-opponent-lookup-context-id")
    parser.add_argument("--left-source-snapshot-id")
    parser.add_argument("--right-source-snapshot-id")
    parser.add_argument("--frozen-opponent-source-snapshot-id")
    parser.add_argument("--first-legal-memory", action="store_true",
                        help="use the explicitly versioned public-memory baseline instead of raw first legal")
    parser.add_argument("--games-per-seat", type=int, default=16)
    parser.add_argument("--master-seed", type=int, default=20261220)
    parser.add_argument("--panel-role", choices=("development", "confirmation"), default="development")
    parser.add_argument("--primary-panel-id")
    parser.add_argument("--source-semantics-certificate-sha256")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    panels = [BenchmarkPanel("random", "random"),
              BenchmarkPanel("first-legal-memory", "first_legal_memory") if args.first_legal_memory
              else BenchmarkPanel("first-legal", "first_legal")]
    if (args.frozen_opponent_artifact is not None or args.frozen_opponent_lookup_context_id is not None
            or args.frozen_opponent_source_snapshot_id is not None):
        panels.append(BenchmarkPanel("historical-a", "frozen_a", args.frozen_opponent_artifact,
                                     args.frozen_opponent_lookup_context_id,
                                     opponent_source_snapshot_id=args.frozen_opponent_source_snapshot_id))
    plan = FrozenBenchmarkPlan(args.benchmark_id, args.left_artifact, args.right_artifact,
        args.left_lookup_context_id, args.right_lookup_context_id, tuple(panels), args.games_per_seat,
        args.master_seed, args.panel_role, args.primary_panel_id,
        source_semantics_certificate_sha256=args.source_semantics_certificate_sha256,
        left_source_snapshot_id=args.left_source_snapshot_id,
        right_source_snapshot_id=args.right_source_snapshot_id)
    report = run_frozen_benchmark(ExperimentStore(args.root), plan, workers=args.workers,
                                 progress=lambda message: print(message, flush=True))
    print(json.dumps({key: report[key] for key in ("benchmark_id", "status", "retained_game_records", "report_path")}, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
