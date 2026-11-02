"""Optional twelve-Game pipeline example; not a strategy-strength experiment.

Run after installing the package: python examples/model_a_smoke.py --output smoke-output
All generated Games, snapshots and update evidence remain in the new output
directory. This starts from an empty prior, not the local research checkpoint.
"""

import argparse
import json
from pathlib import Path

from magical_athlete.experiment_store import ExperimentStore
from magical_athlete.training_a import PilotPlan, ResourceLimits, run_pilot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private experiment directory")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must not exist; this example never resumes or overwrites a run")
    plan = PilotPlan(
        "release-smoke-v1", master_seed=20261102, batches=1,
        games_per_seat=1, evaluation_games_per_seat=1,
        strict_training_health=True,
        repetition_guard="public-turn-local-duel-fixed-keep-uniform-retry-v3",
        evaluation_guard_scheme="public-turn-local-duel-retry-fish-keep-v4",
    )
    limits = ResourceLimits(
        minimum_project_free_bytes=256 * 1024**2,
        minimum_system_free_bytes=256 * 1024**2,
        bytes_per_planned_record=5_000_000,
        temporary_reserve_bytes=64 * 1024**2,
    )
    report = run_pilot(ExperimentStore(args.output), plan, workers=1, resource_limits=limits)
    if not (report["status"] == "completed" and report["training_game_records"] == 4
            and report["evaluation_game_records"] == 8 and report["retained_game_records"] == 12
            and report["inherited_training_games"] == 0 and report["new_training_games"] == 4
            and len(report["updates"]) == 1 and report["failed_games_used_for_updates"] == 0
            and report["evaluation_games_used_for_updates"] == 0):
        raise RuntimeError("the twelve-Game smoke run did not complete; retained evidence must be inspected")
    summary = {
        "kind": "model-a-twelve-game-pipeline-smoke-v1", "completed": True,
        "master_seed": plan.master_seed, "pilot_id": plan.pilot_id,
        "training_games": 4, "evaluation_games": 8, "retained_games": 12,
        "model_updates": 1, "inherited_research_training_games": 0,
        "training_guard": plan.repetition_guard, "evaluation_guard": plan.evaluation_guard_scheme,
        "all_learning_health_verified": all(row["verified_score_updates"] is True for row in report["learning_health"]),
        "failed_or_evaluation_games_used_for_updates": 0,
        "report": Path(report["report_path"]).relative_to(args.output.resolve()).as_posix(),
        "interpretation": "Only demonstrates persistence, a real update, and frozen evaluation; too small for a performance claim.",
    }
    with (args.output / "smoke-summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
