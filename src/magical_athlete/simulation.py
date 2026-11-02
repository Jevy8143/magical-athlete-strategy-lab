if __package__ in (None, ""):
    # Direct file execution needs the src directory on the import path.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Mapping

from magical_athlete.decisions import KeepPolicy, Policy, RandomPolicy
from magical_athlete.engine import D6, DiagnosticLimitExceeded, RaceEngine, replay_trace
from magical_athlete.racer import Racer
from magical_athlete.track import MILD_MILE, SYNTHETIC_TRACK, WILD_WILDS, Track


def make_demo_engine(
    seed: int | None = None, track: Track = SYNTHETIC_TRACK,
    policies: Mapping[int, Policy] | None = None,
) -> RaceEngine:

    racers = [
        Racer(0, "P1"),
        Racer(1, "P2"),
        Racer(2, "P3"),
        Racer(3, "P4"),
    ]

    return RaceEngine(
        track=track,
        racers=racers,
        turn_order=[0, 1, 2, 3],
        dice=D6(seed=seed),
        policies=policies,
    )


def make_scoocher_demo_engine(
    seed: int | None = None, track: Track = MILD_MILE,
    policies: Mapping[int, Policy] | None = None,
) -> RaceEngine:
    """An explicit mid-race fixture to inspect H10/H11, not a drafted game."""
    return RaceEngine(
        track,
        [Racer(0, "Blimp", 16), Racer(1, "Gunk", 2), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)],
        [0, 1, 2, 3],
        D6(seed),
        policies=policies,
    )


def make_decision_demo_engine(
    seed: int | None = None, track: Track = MILD_MILE,
    policies: Mapping[int, Policy] | None = None,
) -> RaceEngine:
    """Single-race fixture; use game_simulation for drafting and four-race scores."""
    return RaceEngine(
        track,
        [Racer(0, "Alchemist"), Racer(1, "Rocket Scientist"),
         Racer(2, "Magician"), Racer(3, "Dicemonger")],
        [0, 1, 2, 3], D6(seed), policies=policies,
    )

def run_one_race(seed: int | None = None):
    return make_demo_engine(seed).run()


def main() -> int:
    parser = argparse.ArgumentParser(description="Incremental single-race engine demo (not a complete game)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--max-turns", type=int, default=10_000)
    parser.add_argument("--track", choices=("synthetic", "mild", "wild"), default="synthetic")
    parser.add_argument("--scenario", choices=("baseline", "scoocher", "decisions"), default="baseline")
    parser.add_argument("--policy", choices=("keep", "random"), default="keep")
    parser.add_argument("--policy-seed", type=int, default=0)
    args = parser.parse_args()
    if args.replay:
        engine = replay_trace(json.loads(args.replay.read_text(encoding="utf-8")))
        print(f"Replay verified: {len(engine.turns)} turns, {len(engine.events)} events")
        return 0
    track = {"mild": MILD_MILE, "wild": WILD_WILDS, "synthetic": SYNTHETIC_TRACK}[args.track]
    # One independent policy stream per seat; the dice stream uses --seed.
    policies = {
        rid: RandomPolicy(args.policy_seed + rid) if args.policy == "random" else KeepPolicy()
        for rid in range(4)
    }
    if args.scenario == "scoocher":
        if args.track == "synthetic":
            track = Track("Scoocher synthetic demo", 30, 15, frozenset({6}))
        engine = make_scoocher_demo_engine(args.seed, track, policies)
    elif args.scenario == "decisions":
        engine = make_decision_demo_engine(args.seed, track, policies)
    else:
        engine = make_demo_engine(args.seed, track, policies)
    try:
        result = engine.run(max_turns=args.max_turns)
        trace = engine.trace()
    except DiagnosticLimitExceeded as exc:
        trace = exc.trace
        output = args.output or Path(f".cache/diagnostics/race-seed-{args.seed}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Diagnostic failure: incomplete race. Trace: {output.resolve()}")
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Trace: {args.output.resolve()}")
    print(f"Single-race demo on {track.name}; use game_simulation for drafting and four-race scoring.")
    if args.scenario == "scoocher":
        print("Scoocher scenario starts from an explicit mid-race fixture.")
    print(f"Finish order: {result.finish_order}")
    print(f"Final positions: {result.final_positions}")
    print(f"Turns: {len(result.turns)}; events: {len(result.events)}")
    print(f"Decisions: {len(engine.decisions)}; policy: {args.policy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
