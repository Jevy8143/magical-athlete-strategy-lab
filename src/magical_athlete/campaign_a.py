"""Bounded, resumable model-A continuation with a separate final confirmation.

Development panels may guide compute allocation and checkpoint selection.
The final confirmation has a fresh namespace and cannot trigger more training
inside this campaign. Every attempted game keeps the normal full trace.
"""

import argparse
import hashlib
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .evaluation_a import FrozenEvaluationPlan, run_frozen_evaluation
from .experiment_spec import Purpose, SeedPlan, canonical_json, content_hash
from .experiment_store import ExperimentError, ExperimentStore, _checked_json, _write_immutable, infrastructure_fingerprint
from .game import GameEngine
from .model_a import AConfig, FrozenAModel
from .model_a_features import encode_request
from .policy_runtime import opponent_context_id
from .training_a import (PilotPlan, ResourceLimits, _archive_sources, check_resources,
                         local_work_environment, model_snapshot, pilot_config, run_pilot)


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    campaign_id: str
    parent_artifact_sha256: str | None = None
    opponent_snapshot_id: str | None = None
    master_seed: int = 20261110
    first_stage_games: int = 128
    later_stage_games: int = 256
    maximum_new_training_games: int = 1024
    maximum_stages: int = 4
    development_games_per_seat: int = 16
    confirmation_games_per_seat: int = 32
    wall_budget_seconds: int = 3600
    resource_limits: ResourceLimits | None = None
    continue_healthy_training_tail: bool | None = None
    source_semantics_certificate_sha256: str | None = None

    def __post_init__(self):
        SeedPlan(self.master_seed, Purpose.TRAIN, self.campaign_id)
        if len(self.campaign_id) > 32:
            raise ValueError("campaign_id must have at most 32 characters")
        if (self.parent_artifact_sha256 is None) != (self.opponent_snapshot_id is None):
            raise ValueError("campaign parent and opponent must both be supplied or both omitted for a fresh start")
        if self.source_semantics_certificate_sha256 is not None and self.parent_artifact_sha256 is None:
            raise ValueError("campaign semantic certification requires an explicit parent")
        for value in (self.parent_artifact_sha256, self.opponent_snapshot_id, self.source_semantics_certificate_sha256):
            if value is not None and (not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                raise ValueError("campaign parent and opponent must be SHA256 identities")
        if self.resource_limits is not None and not isinstance(self.resource_limits, ResourceLimits):
            raise ValueError("campaign resource_limits must be ResourceLimits or None")
        if self.continue_healthy_training_tail is None:
            object.__setattr__(self, "continue_healthy_training_tail", self.parent_artifact_sha256 is None)
        if type(self.continue_healthy_training_tail) is not bool:
            raise ValueError("continue_healthy_training_tail must be boolean")
        for value in (self.first_stage_games, self.later_stage_games, self.maximum_new_training_games):
            if type(value) is not int or value < 16 or value % 16:
                raise ValueError("stage/training counts must be positive multiples of 16")
        for value in (self.maximum_stages, self.development_games_per_seat,
                      self.confirmation_games_per_seat, self.wall_budget_seconds):
            if type(value) is not int or value < 1:
                raise ValueError("campaign limits and evaluation counts must be positive integers")
        SeedPlan(self.master_seed + self.maximum_stages - 1, Purpose.TRAIN, self.campaign_id)
        SeedPlan(self.master_seed + 100_000, Purpose.EVALUATION, self.campaign_id)
        SeedPlan(self.master_seed + 50_000 + self.maximum_stages - 1, Purpose.EVALUATION, self.campaign_id)

    def to_dict(self):
        result = {"protocol": ("bounded-a-fresh-development-confirmation-v2" if self.parent_artifact_sha256 is None
                               else "bounded-a-continuation-development-confirmation-v2" if self.continue_healthy_training_tail
                               else "bounded-a-continuation-development-confirmation-v1"),
                **{field: getattr(self, field) for field in self.__dataclass_fields__
                   if field not in {"resource_limits", "continue_healthy_training_tail", "source_semantics_certificate_sha256"}},
                "engine_fingerprint": GameEngine._code_fingerprint(),
                "infrastructure_fingerprint": infrastructure_fingerprint(),
                "retention_policy": "all_games_full_trace_and_summary",
                "stopping_rule": {"plateau_stages": 2, "maximum_small_gain": 0.1,
                                  "maximum_small_race_coverage_gain": 0.02,
                                  "coverage_probe": "fixed first-stage public development race requests",
                                  "regression": "development paired 95% upper bound below zero",
                                  "budget": "measured completed-stage wall time; soft limit at stage boundaries"},
                "selection_rule": "accept child when its paired development mean exceeds parent; otherwise retain parent",
                "confirmation_rule": "one fixed-size fresh panel after selection; never used for further training in this campaign"}
        if self.resource_limits is not None:
            result["resource_limits"] = self.resource_limits.to_dict()
        if self.continue_healthy_training_tail or self.parent_artifact_sha256 is None:
            result.update(continue_healthy_training_tail=self.continue_healthy_training_tail,
                          training_parent_rule="continue the latest healthy child unless development shows a clear regression",
                          selection_rule="compare child against the selected reference; positive paired development mean selects child",
                          additional_reference_panel_seed_offset=50_000,
                          confirmation_seed_offset=100_000,
                          strict_training_health=True)
        if self.source_semantics_certificate_sha256 is not None:
            result.update(protocol=result["protocol"] + "-source-semantics-v1",
                          source_semantics_certificate_sha256=self.source_semantics_certificate_sha256,
                          historical_source_policy="complete original TRAIN lineage certified against current physics; no diagnostic samples")
        return result


def _write_checked(path, payload):
    _write_immutable(path, canonical_json({"payload": payload, "sha256": content_hash(payload)}).encode("utf-8"))


def _read_checked(path):
    envelope = json.loads(path.read_text(encoding="utf-8"))
    return _checked_json(canonical_json(envelope["payload"]), envelope["sha256"])


def _race_coverage(store, runs):
    windows = selected = prior_only = unseen = actions = 0
    for run in runs:
        for row in store._game_rows(run["run_id"], Purpose.EVALUATION):
            if row["status"] != "completed":
                continue
            records = store._load_records(row)
            # The version under evaluation is the only mean-mode seat. The
            # three fixed opponents stay Thompson throughout this protocol.
            for decision in records["decisions"]:
                audit = decision.get("model_a_audit", {})
                if decision["scope"] != "race" or audit.get("mode") != "mean" or len(audit["action_keys"]) < 2:
                    continue
                counts = [item["observations"] for item in audit["estimates"]]
                windows += 1
                selected += counts[audit["selected_index"]] > 0
                prior_only += not any(counts)
                unseen += sum(count == 0 for count in counts)
                actions += len(counts)
    return {"multi_action_windows": windows, "selected_with_evidence": selected,
            "selected_with_evidence_fraction": selected / windows if windows else None,
            "entirely_prior_only_windows": prior_only, "candidate_actions": actions,
            "unseen_candidate_actions": unseen,
            "scope": "descriptive completed focal race windows, not independent games or a strength metric"}


def _stage_decision(report, before_coverage, after_coverage, previous_plateaus):
    paired = report["paired_evaluation"]
    gain = paired.get("mean_paired_total_difference")
    interval = paired.get("paired_bootstrap_95_percent_interval")
    before = before_coverage["windows_with_any_evidence_fraction"]
    after = after_coverage["windows_with_any_evidence_fraction"]
    coverage_gain = after - before if before is not None and after is not None else None
    unseen_gain = before_coverage["unseen_candidate_fraction"] - after_coverage["unseen_candidate_fraction"]
    healthy = report["status"] == "completed" and all(item["verified_score_updates"] for item in report["learning_health"])
    plateau = (gain is not None and gain <= 0.1 and coverage_gain is not None and coverage_gain <= 0.02
               and unseen_gain <= 0.02)
    plateaus = previous_plateaus + 1 if plateau else 0
    stop = "incomplete_or_unhealthy_stage" if not healthy else None
    if healthy and interval and interval[1] < 0:
        stop = "development_regression_needs_review"
    elif healthy and plateaus >= 2:
        stop = "two_stages_without_measured_score_or_coverage_gain"
    return {"healthy": healthy, "development_mean_gain": gain, "race_evidence_coverage_gain": coverage_gain,
            "fixed_probe_unseen_candidate_fraction_reduction": unseen_gain,
            "consecutive_plateau_stages": plateaus, "accept_child": healthy and gain is not None and gain > 0,
            "stop_reason": stop,
            "interpretation": "compute-allocation rule, not a convergence or global-optimality test"}


def _probe_requests(store, runs):
    requests = []
    for run in runs:
        spec = store.run_spec(run["run_id"])
        for row in store._game_rows(run["run_id"], Purpose.EVALUATION):
            if row["status"] != "completed":
                continue
            for decision in store._load_records(row)["decisions"]:
                audit = decision.get("model_a_audit", {})
                if decision["scope"] == "race" and audit.get("mode") == "mean" and len(audit["action_keys"]) > 1:
                    requests.append({"request": decision["request"], "race_index": decision["race_index"],
                                     "context_id": opponent_context_id(spec, decision["player_id"])})
    return requests


def _probe_coverage(model, requests):
    learned, unseen, candidates = 0, 0, 0
    for request in requests:
        encoded = encode_request(request["request"], race_index=request["race_index"])
        audit = model.select(request["context_id"], encoded, mode="mean", rng=random.Random(0)).audit
        counts = [estimate["observations"] for estimate in audit["estimates"]]
        learned += any(counts)
        unseen += sum(count == 0 for count in counts)
        candidates += len(counts)
    return {"multi_action_windows": len(requests), "windows_with_any_evidence": learned,
            "windows_with_any_evidence_fraction": learned / len(requests) if requests else None,
            "unseen_candidate_fraction": unseen / candidates if candidates else 1.0,
            "scope": "one fixed set of public development requests; value-coverage diagnostic, not replayed games or a strength estimate"}


def run_campaign(store, plan, *, workers=2, progress=None):
    with local_work_environment(store):
        return _run_campaign(store, plan, workers=workers, progress=progress)


def _run_campaign(store, plan, *, workers=2, progress=None):
    expected_source = (GameEngine._code_fingerprint(), infrastructure_fingerprint())

    def guard(phase, records=0):
        return check_resources(store, plan.resource_limits, records, f"{plan.campaign_id}:{phase}",
                               expected_source=expected_source)

    # Identity is bound before choosing the hash directory. A changed budget
    # cannot reopen a completed campaign after seeing its final confirmation.
    _write_checked(store.root / "model-a-campaigns" / "identities" / f"{plan.campaign_id}.json", plan.to_dict())
    directory = store.root / "model-a-campaigns" / content_hash(plan.to_dict())
    _write_checked(directory / "plan.json", plan.to_dict())
    if (directory / "report.json").exists():
        report = _read_checked(directory / "report.json")
        store.read_artifact(report["selected_artifact_sha256"])
        if plan.source_semantics_certificate_sha256 is not None:
            from .source_compatibility import certified_source_conditions

            artifacts = {report["selected_artifact_sha256"], report.get("training_tail_artifact_sha256", report["selected_artifact_sha256"])}
            certified_source_conditions(store, tuple(FrozenAModel.from_bytes(store.read_artifact(item)) for item in artifacts),
                                        plan.source_semantics_certificate_sha256)
        return {**report, "report_path": str(directory / "report.json")}
    guard("before-campaign", min(plan.first_stage_games, plan.maximum_new_training_games)
          + 8 * (plan.development_games_per_seat + plan.confirmation_games_per_seat))
    source_archive = _archive_sources(store)
    if plan.parent_artifact_sha256 is None:
        original = FrozenAModel.empty(AConfig())
        opponent = model_snapshot(store, original, f"{plan.campaign_id}-a0", "thompson")
    else:
        original = FrozenAModel.from_bytes(store.read_artifact(plan.parent_artifact_sha256))
        opponent = store.read_snapshot(plan.opponent_snapshot_id)
    if (opponent.model_type != "A" or not opponent.frozen or not opponent.artifact_sha256
            or json.loads(opponent.parameters_json).get("mode") != "thompson"
            or opponent.training_games != 0):
        raise ExperimentError("campaign confirmation requires the exact frozen zero-training A0 Thompson opponent")
    opponent_model = FrozenAModel.from_bytes(store.read_artifact(opponent.artifact_sha256))
    if opponent_model.training_games or opponent_model.config != original.config:
        raise ExperimentError("campaign A0 model/configuration does not match its parent")
    # A fresh first stage immediately enters run_pilot, whose parent gate
    # authenticates the complete certificate before any evaluation or TRAIN
    # Game. Partial pilot recovery enters that same gate. Recovered stages
    # can bypass the pilot, so they must retain this campaign-level check.
    first_pilot_will_verify_parent = (
        plan.maximum_stages >= 1
        and min(plan.first_stage_games, plan.maximum_new_training_games) >= 16
        and not (directory / "stage-1.json").exists()
    )
    if plan.source_semantics_certificate_sha256 is not None and not first_pilot_will_verify_parent:
        from .source_compatibility import certified_source_conditions

        certified_source_conditions(store, (original,), plan.source_semantics_certificate_sha256)
    opponent_snapshot_id = opponent.snapshot_id
    selected = original.artifact_sha256
    training_tail = selected
    continuous = plan.continue_healthy_training_tail
    stages, total_elapsed, new_games, plateaus = [], 0.0, 0, 0
    stop_reason = None
    for index in range(plan.maximum_stages):
        desired = plan.first_stage_games if index == 0 else plan.later_stage_games
        requested = min(desired, plan.maximum_new_training_games - new_games)
        if requested < 16:
            stop_reason = "maximum_new_training_games_reached"
            break
        stage_parent = training_tail if continuous else selected
        extra_reference_records = 8 * plan.development_games_per_seat if continuous and stage_parent != selected else 0
        if stages:
            seconds_per_attempt = max(stage["budget_charged_seconds"] / stage["retained_game_records"] for stage in stages[-2:])
            estimated_confirmation = seconds_per_attempt * 8 * plan.confirmation_games_per_seat
            estimated_next = seconds_per_attempt * (requested + 8 * plan.development_games_per_seat + extra_reference_records)
            if total_elapsed + 1.2 * (estimated_next + estimated_confirmation) > plan.wall_budget_seconds:
                stop_reason = "measured_time_budget_reserves_final_confirmation"
                break
        stage_path = directory / f"stage-{index + 1}.json"
        if stage_path.exists():
            stage = _read_checked(stage_path)
            if stage["parent_artifact_sha256"] != stage_parent:
                raise ExperimentError("campaign stage parent differs from the committed training ancestry")
            if continuous and stage["selection_reference_artifact_sha256"] != selected:
                raise ExperimentError("campaign stage selection reference differs from the committed selection")
            report_path = store.root / stage["report_relative_path"]
            report_bytes = report_path.read_bytes()
            if hashlib.sha256(report_bytes).hexdigest() != stage["report_sha256"]:
                raise ExperimentError("campaign stage report checksum mismatch")
            report = json.loads(report_bytes)
            if stage.get("reference_comparison_report_relative_path"):
                reference_path = store.root / stage["reference_comparison_report_relative_path"]
                if hashlib.sha256(reference_path.read_bytes()).hexdigest() != stage["reference_comparison_report_sha256"]:
                    raise ExperimentError("campaign reference comparison report checksum mismatch")
            if progress:
                progress(f"Recovered stage {index + 1}; no new games scheduled")
        else:
            guard(f"before-stage-{index + 1}", requested
                  + 8 * (plan.development_games_per_seat + plan.confirmation_games_per_seat) + extra_reference_records)
            parent = FrozenAModel.from_bytes(store.read_artifact(stage_parent))
            checkpoint_batches, games_per_seat = ((requested // 128, 32)
                                                  if continuous and requested >= 128 and requested % 128 == 0
                                                  else (4, requested // 16))
            pilot = PilotPlan(f"{plan.campaign_id}-s{index + 1}", plan.master_seed + index,
                              checkpoint_batches, games_per_seat, plan.development_games_per_seat,
                              parent.config, pilot_config(), stage_parent, opponent_snapshot_id,
                              strict_training_health=continuous or plan.parent_artifact_sha256 is None,
                              source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256)
            attempts = directory / f"stage-{index + 1}-invocations"
            attempt_id = uuid.uuid4().hex
            _write_checked(attempts / f"{attempt_id}-start.json", {"stage": index + 1, "attempt_id": attempt_id})
            started = time.perf_counter()
            reference_comparison = None
            try:
                report = run_pilot(store, pilot, workers=workers, progress=progress,
                                   resource_limits=plan.resource_limits)
                if (extra_reference_records and report["status"] == "completed"
                        and all(item["verified_score_updates"] for item in report["learning_health"])):
                    guard(f"before-reference-comparison-{index + 1}", extra_reference_records)
                    reference_plan = FrozenEvaluationPlan(
                        f"{plan.campaign_id}-ref-s{index + 1}", selected, report["model"]["artifact_sha256"],
                        opponent_snapshot_id, plan.development_games_per_seat,
                        plan.master_seed + 50_000 + index, "development", pilot_config(),
                        source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256)
                    reference_comparison = run_frozen_evaluation(store, reference_plan, workers=workers, progress=progress)
                    guard(f"after-reference-comparison-{index + 1}")
            finally:
                _write_checked(attempts / f"{attempt_id}-end.json", {"stage": index + 1, "attempt_id": attempt_id,
                                                                      "elapsed_seconds": time.perf_counter() - started})
            elapsed = sum(_read_checked(path)["elapsed_seconds"] for path in attempts.glob("*-end.json"))
            guard(f"after-stage-{index + 1}")
            unfinished_attempts = sum(not path.with_name(path.name.replace("-start.json", "-end.json")).exists()
                                      for path in attempts.glob("*-start.json"))
            before = _race_coverage(store, report["initial_evaluation_runs"])
            after = _race_coverage(store, report["final_evaluation_runs"])
            if not stages:
                probe = _probe_requests(store, report["initial_evaluation_runs"])
                probe_artifact = store.put_artifact(canonical_json(probe).encode("utf-8"))
            else:
                probe_artifact = stages[0]["fixed_development_probe_artifact"]
                probe = json.loads(store.read_artifact(probe_artifact))
            child = FrozenAModel.from_bytes(store.read_artifact(report["model"]["artifact_sha256"]))
            reference = FrozenAModel.from_bytes(store.read_artifact(selected)) if continuous else parent
            probe_before, probe_after = _probe_coverage(reference, probe), _probe_coverage(child, probe)
            selection_report = report
            if reference_comparison is not None:
                selection_report = {**report, "paired_evaluation": reference_comparison["paired_evaluation"],
                                    "status": "completed" if reference_comparison["status"] == "completed" else "completed_with_failures"}
            decision = _stage_decision(selection_report, probe_before, probe_after, plateaus)
            path = Path(report["report_path"])
            stage = {"stage": index + 1, "parent_artifact_sha256": stage_parent,
                     "child_artifact_sha256": report["model"]["artifact_sha256"],
                     "new_training_games": report["new_training_games"],
                     "planned_new_training_games": requested,
                     "retained_game_records": report["retained_game_records"]
                        + (reference_comparison["retained_game_records"] if reference_comparison else 0), "elapsed_seconds": elapsed,
                     "unfinished_timing_attempts": unfinished_attempts,
                     "budget_charged_seconds": elapsed + unfinished_attempts * plan.wall_budget_seconds,
                     "evaluation_panel_role": "development", "evaluation_used_for_selection": True,
                     "report_relative_path": path.relative_to(store.root).as_posix(),
                     "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "race_coverage_before": before, "race_coverage_after": after, "decision": decision}
            stage.update(fixed_development_probe_artifact=probe_artifact,
                         fixed_probe_coverage_before=probe_before, fixed_probe_coverage_after=probe_after)
            if continuous:
                stage.update(selection_reference_artifact_sha256=selected,
                             selection_paired_evaluation=selection_report["paired_evaluation"],
                             training_tail_cumulative_training_games=child.training_games,
                             reference_comparison_retained_game_records=reference_comparison["retained_game_records"] if reference_comparison else 0)
                if reference_comparison:
                    reference_path = Path(reference_comparison["report_path"])
                    stage.update(reference_comparison_report_relative_path=reference_path.relative_to(store.root).as_posix(),
                                 reference_comparison_report_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
            _write_checked(stage_path, stage)
        stages.append(stage)
        new_games += stage["planned_new_training_games"]
        total_elapsed += stage["budget_charged_seconds"]
        plateaus = stage["decision"]["consecutive_plateau_stages"]
        if stage["decision"]["accept_child"]:
            selected = stage["child_artifact_sha256"]
        if (continuous and stage["decision"]["healthy"]
                and stage["decision"]["stop_reason"] != "development_regression_needs_review"):
            training_tail = stage["child_artifact_sha256"]
        elif not continuous:
            training_tail = selected
        if progress:
            progress(f"Stage {index + 1}: new TRAIN={stage['new_training_games']}, elapsed={stage['elapsed_seconds']:.1f}s; "
                     f"development gain={stage['decision']['development_mean_gain']}; selected={selected}")
        if stage["decision"]["stop_reason"]:
            stop_reason = stage["decision"]["stop_reason"]
            break
    stop_reason = stop_reason or "maximum_stages_reached"
    confirmation = None
    confirmation_elapsed = 0.0
    if all(stage["decision"]["healthy"] for stage in stages) and stages:
        # Selection is already closed. These outcomes cannot alter the selected
        # checkpoint or increase the campaign's training allocation.
        guard("before-final-confirmation", 8 * plan.confirmation_games_per_seat)
        frozen_plan = FrozenEvaluationPlan(f"{plan.campaign_id}-confirm", original.artifact_sha256, selected,
                                           opponent_snapshot_id, plan.confirmation_games_per_seat,
                                           plan.master_seed + 100_000, "confirmation", pilot_config(),
                                           source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256)
        started = time.perf_counter()
        confirmation = run_frozen_evaluation(store, frozen_plan, workers=workers, progress=progress)
        confirmation_elapsed = time.perf_counter() - started
        guard("after-final-confirmation")
    selected_model = FrozenAModel.from_bytes(store.read_artifact(selected))
    report = {"plan": plan.to_dict(), "source_archive_artifact_sha256": source_archive,
              "stages": stages, "stop_reason": stop_reason, "selected_artifact_sha256": selected,
              "inherited_training_games": original.training_games,
              "new_complete_training_games_across_all_branches": sum(stage["new_training_games"] for stage in stages),
              "selected_model_cumulative_training_games": selected_model.training_games,
              "training_and_development_elapsed_seconds": sum(stage["elapsed_seconds"] for stage in stages),
              "training_and_development_budget_charged_seconds": total_elapsed,
              "unknown_interrupted_timing_policy": "charge a full campaign budget per unfinished invocation; measured durations remain separate",
              "confirmation_elapsed_seconds_this_invocation": confirmation_elapsed,
              "development_panels_used_for_selection": True,
              "training_and_development_retained_game_records": sum(stage["retained_game_records"] for stage in stages),
              "confirmation_report_relative_path": str(Path(confirmation["report_path"]).relative_to(store.root)) if confirmation else None,
              "confirmation_paired_evaluation": confirmation["paired_evaluation"] if confirmation else None,
              "confirmation_used_for_training_or_selection": False,
              "limitation": "one exact frozen A0 opponent distribution; no general strength or convergence claim",
              "status": "completed" if confirmation and confirmation["status"] == "completed" else "needs_review"}
    if plan.parent_artifact_sha256 is None:
        report.update(fresh_start=True, initial_artifact_sha256=original.artifact_sha256,
                      fixed_opponent_snapshot_id=opponent_snapshot_id)
    if continuous:
        tail_model = FrozenAModel.from_bytes(store.read_artifact(training_tail))
        report.update(training_tail_artifact_sha256=training_tail,
                      training_tail_cumulative_training_games=tail_model.training_games,
                      training_tail_is_selected_reference=training_tail == selected,
                      training_tail_rule="healthy training ancestry is retained separately from development selection")
    if plan.source_semantics_certificate_sha256 is not None:
        report.update(source_semantics_certificate_sha256=plan.source_semantics_certificate_sha256,
                      semantic_diagnostic_games_used_as_training_samples=0,
                      historical_training_sources_inherited_under_current_semantics=bool(original.training_games))
    guard("before-campaign-report")
    _write_checked(directory / "report.json", report)
    return {**report, "report_path": str(directory / "report.json")}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Measured, bounded model-A continuation and separate frozen confirmation")
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--parent-artifact-sha256", help="Omit with opponent for a fresh current-engine series")
    parser.add_argument("--opponent-snapshot-id", help="Exact A0 snapshot; must accompany an explicit parent")
    parser.add_argument("--source-semantics-certificate-sha256", help="Whole-parent exact current-source semantic certificate")
    parser.add_argument("--master-seed", type=int, default=20261110)
    parser.add_argument("--first-stage-games", type=int, default=128)
    parser.add_argument("--later-stage-games", type=int, default=256)
    parser.add_argument("--maximum-new-training-games", type=int, default=1024)
    parser.add_argument("--maximum-stages", type=int, default=4)
    parser.add_argument("--development-games-per-seat", type=int, default=16)
    parser.add_argument("--confirmation-games-per-seat", type=int, default=32)
    parser.add_argument("--wall-budget-seconds", type=int, default=3600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-project-free-gib", type=int, default=30)
    parser.add_argument("--minimum-system-free-gib", type=int, default=5)
    parser.add_argument("--record-budget-mb", type=int, default=5)
    parser.add_argument("--continue-healthy-training-tail", action="store_true", default=None,
                        help="Keep healthy accumulated learning separate from checkpoint selection (fresh default)")
    args = parser.parse_args(argv)
    try:
        values = vars(args).copy()
        root, workers = values.pop("root"), values.pop("workers")
        values["resource_limits"] = ResourceLimits(
            minimum_project_free_bytes=values.pop("minimum_project_free_gib") * 1024**3,
            minimum_system_free_bytes=values.pop("minimum_system_free_gib") * 1024**3,
            bytes_per_planned_record=values.pop("record_budget_mb") * 1_000_000)
        report = run_campaign(ExperimentStore(root), CampaignPlan(**values), workers=workers,
                              progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Model A campaign error: {exc}\n")
    print(json.dumps({key: report[key] for key in ("status", "selected_artifact_sha256", "selected_model_cumulative_training_games",
                     "stop_reason", "confirmation_paired_evaluation", "report_path")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
