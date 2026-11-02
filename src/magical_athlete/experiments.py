"""Run, resume, audit and export purpose-isolated four-race experiments.

The CLI constructs baselines. The programmatic runner also executes genuine
frozen model-A artifacts; training_a owns updates between audited batches.
"""

import argparse
import json
import sys
import traceback
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import replace
from pathlib import Path

from .decisions import KeepPolicy, RandomPolicy
from .experiment_spec import GameSeeds, PolicySnapshot, Purpose, RunSpec, SeedPlan, canonical_json
from .experiment_store import ExperimentError, ExperimentStore, _write_immutable, infrastructure_fingerprint, verify_export
from .game import FirstLegalGamePolicy, GameDiagnosticError, GameEngine, RandomGamePolicy, official_config
from .game_simulation import fixture_config
from .policy_runtime_router import build_policies, require_implemented, snapshot_artifacts


def baseline_snapshot(kind: str) -> PolicySnapshot:
    if kind == "first_legal_memory":
        from .first_legal_memory_v1 import MODEL_VERSION, baseline_parameters
        return PolicySnapshot("baseline", kind, MODEL_VERSION, canonical_json(baseline_parameters()))
    if kind not in {"random", "first_legal"}:
        raise ExperimentError("only implemented random/first_legal baselines can execute")
    return PolicySnapshot("baseline", kind, f"baseline-{kind}-v1")


def make_run_spec(run_id: str, study_id: str, purpose: Purpose, master_seed: int,
                  lineup=("random",) * 4, *, config=None) -> RunSpec:
    return RunSpec(
        run_id, study_id, purpose, tuple(baseline_snapshot(kind) for kind in lineup),
        SeedPlan(master_seed, purpose, study_id), config if config is not None else official_config(),
    )


def _simulate_one(spec_data: dict, seed_data: dict, artifacts=None):
    spec = RunSpec.from_dict(spec_data)
    seeds = GameSeeds.from_dict(seed_data)
    for policy in spec.policies:
        require_implemented(policy)
    game = None
    try:
        game_policies, race_policies = build_policies(spec, seeds, artifacts or {})
        game = GameEngine(spec.game_config, seeds.game_seed, game_policies, race_policies)
        game.run()
        return game.trace(), None
    except GameDiagnosticError as exc:
        return exc.trace, {"kind": "diagnostic", "exception_type": type(exc).__name__, "message": exc.reason}
    except Exception as exc:
        return game.trace() if game is not None else None, {
            "kind": "error", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
        }


def _timed_simulate_one(spec_data, seed_data, artifacts):
    started = time.perf_counter()
    result = _simulate_one(spec_data, seed_data, artifacts)
    return result, time.perf_counter() - started


def run_experiment(store: ExperimentStore, spec: RunSpec, game_indices, *, workers=1) -> dict:
    """Retain runtime measurements separately from deterministic game evidence.

    Worker durations are summed elapsed times, not CPU time or wall throughput.
    Persistence includes replay, action auditing and SQLite/artifact writes.
    A recovered invocation has its own record; timings never change a run plan.
    """
    started = time.perf_counter()
    timing = {"kind": "experiment-runtime-v1", "run_id": spec.run_id,
              "workers": workers, "simulation_elapsed_sum_seconds": 0.0,
              "persistence_elapsed_seconds": 0.0, "returned_worker_results": 0,
              "previously_committed_games": 0, "newly_attempted_games": 0}
    try:
        result = _run_experiment(store, spec, game_indices, workers=workers, timing=timing)
        timing["status"] = result["status"]
        return result
    except BaseException as exc:
        timing["exception_type"] = type(exc).__name__
        raise
    finally:
        timing["wall_elapsed_seconds"] = time.perf_counter() - started
        _write_immutable(store.root / "runtime-invocations" / spec.run_id / (uuid.uuid4().hex + ".json"),
                         canonical_json(timing).encode("utf-8"))


def _run_experiment(store: ExperimentStore, spec: RunSpec, game_indices, *, workers, timing) -> dict:
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ExperimentError("workers must be an integer from 1 through 64")
    artifacts = snapshot_artifacts(store, spec)
    manifest = store.create_run(spec, game_indices)
    expected_infrastructure = manifest["infrastructure_fingerprint"]
    already = store.completed_indices(spec.run_id)
    if already:
        store.audit_run(spec.run_id, replay=False)
    remaining = [index for index in manifest["game_indices"] if index not in already]
    timing["previously_committed_games"] = len(already)
    timing["newly_attempted_games"] = len(remaining)

    def persist(index, measured_result):
        result, simulation_elapsed = measured_result
        timing["simulation_elapsed_sum_seconds"] += simulation_elapsed
        timing["returned_worker_results"] += 1
        persist_started = time.perf_counter()
        if infrastructure_fingerprint() != expected_infrastructure:
            raise ExperimentError("experiment source changed during execution; committed records are preserved")
        store.manifest(spec.run_id)  # Recheck immutable policy artifacts after execution too.
        trace, failure = result
        try:
            if trace is None:
                raise ExperimentError("simulation failed before a trace was available")
            store.record_game(spec.run_id, index, trace, failure=failure)
        except (ValueError, RuntimeError, KeyError, TypeError) as exc:
            store.quarantine_game(spec.run_id, index, trace, {
                "kind": "logging_or_validation_error", "exception_type": type(exc).__name__,
                "message": str(exc), "simulation_failure": failure,
            })
        timing["persistence_elapsed_seconds"] += time.perf_counter() - persist_started

    if workers == 1:
        for index in remaining:
            persist(index, _timed_simulate_one(spec.to_dict(), store.seeds_for(spec.run_id, index).to_dict(), artifacts))
    elif remaining:
        # Keep the outstanding result window bounded; full traces may be large.
        indices = iter(remaining)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {}

            def submit():
                index = next(indices, None)
                if index is not None:
                    future = executor.submit(_timed_simulate_one, spec.to_dict(), store.seeds_for(spec.run_id, index).to_dict(), artifacts)
                    pending[future] = index

            for _ in range(min(len(remaining), 2 * workers)):
                submit()
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index = pending.pop(future)
                    persist(index, future.result())
                    submit()
    return {**store.summary(spec.run_id), "previously_committed_games": len(already),
            "newly_attempted_games": len(remaining)}


def _parser():
    parser = argparse.ArgumentParser(description="Purpose-isolated four-race experiment logs; no model training")
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"), help="Managed SQLite/artifact store")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume an immutable plan with baseline policies")
    run.add_argument("--run-id", required=True)
    run.add_argument("--study-id", required=True, help="Explicit seed/pairing panel namespace")
    run.add_argument("--purpose", choices=tuple(p.value for p in Purpose), required=True)
    run.add_argument("--master-seed", type=int, default=0)
    run.add_argument("--games", type=int, default=4)
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--lineup", default="random,random,random,random", help="Four seats: random or first_legal")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--fixture", action="store_true", help="Only allowed for diagnostic purpose")
    run.add_argument("--max-turns-per-race", type=int)
    run.add_argument("--max-events-per-race", type=int)
    for command in ("inspect", "audit"):
        child = commands.add_parser(command)
        child.add_argument("--run-id", required=True)
        if command == "audit":
            child.add_argument("--no-replay", action="store_true", help="Check hashes/records without re-executing games")
    export = commands.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--purpose", choices=tuple(p.value for p in Purpose), required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--model-type", choices=("A", "B"))
    export.add_argument("--snapshot-id")
    verify = commands.add_parser("verify-export")
    verify.add_argument("path", type=Path)
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify-export":
            result = verify_export(args.path)
        else:
            store = ExperimentStore(args.root)
            if args.command == "run":
                if args.games < 1 or args.start_index < 0:
                    raise ExperimentError("games must be positive and start-index nonnegative")
                config = fixture_config() if args.fixture else official_config()
                overrides = {name: value for name in ("max_turns_per_race", "max_events_per_race")
                             if (value := getattr(args, name)) is not None}
                if overrides:
                    config = replace(config, **overrides)
                spec = make_run_spec(args.run_id, args.study_id, Purpose(args.purpose), args.master_seed,
                                     tuple(args.lineup.split(",")), config=config)
                result = run_experiment(store, spec, range(args.start_index, args.start_index + args.games), workers=args.workers)
            elif args.command == "inspect":
                result = {"summary": store.summary(args.run_id), "manifest": store.manifest(args.run_id),
                          "failures": store.failures(args.run_id)}
            elif args.command == "audit":
                result = store.audit_run(args.run_id, replay=not args.no_replay)
            else:
                result = store.export(args.run_id, args.output, purpose=Purpose(args.purpose),
                                      model_type=args.model_type, snapshot_id=args.snapshot_id)
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Experiment error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "run" and result["status"] != "completed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
