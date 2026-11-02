"""Official four-race game runner with explicit fixture and replay modes."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .decisions import RandomPolicy
from .game import BASE_DECK, GameConfig, GameDiagnosticError, GameEngine, RacerCard, RandomGamePolicy, official_config, replay_game_trace
from .scoring import AwardSchedule
from .track import MILD_MILE, Track


def fixture_config() -> GameConfig:
    """Declared test values, deliberately not a transcription of official Wild."""
    return GameConfig(
        MILD_MILE,
        Track(
            "TEST ONLY: synthetic second track", 35, 18, frozenset({7, 22}),
            source="Synthetic workflow fixture; NOT official Wild Wilds",
            first_corner_position=14, data_version="synthetic-workflow-v1",
            star_positions=frozenset({5, 12, 26}), arrows=((3, 2), (14, -3), (24, 2)),
        ),
        AwardSchedule(((2, 1), (6, 4), (5, 2), (11, 8)),
                      source="TEST ONLY arbitrary placement awards; NOT official chips", verified=False),
        deck=BASE_DECK, official=False,
        source="Original 36-card draft with synthetic second track/awards; diagnostic fixture only",
    )


def load_config(path: Path) -> GameConfig:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return GameConfig(
        mild_track=Track(**data["mild_track"]), wild_track=Track(**data["wild_track"]),
        award_schedule=AwardSchedule(**data["award_schedule"]),
        deck=tuple(RacerCard(**card) for card in data["deck"]) if "deck" in data else BASE_DECK,
        official=data.get("official", False), source=data["source"],
        max_turns_per_race=data.get("max_turns_per_race", 10_000),
        max_rolloff_rounds=data.get("max_rolloff_rounds", 100),
        max_events_per_race=data.get("max_events_per_race", 100_000),
    )


def write_trace(path: Path, trace: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Draft and four-race workflow with auditable points and replay")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--official", action="store_true", help="Use verified original-box tracks, 8 prize chips and 36 racers")
    source.add_argument("--fixture", action="store_true", help="Explicitly use synthetic second-track and prize data")
    source.add_argument("--config", type=Path, help="Explicit JSON track/award configuration")
    source.add_argument("--replay", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--game-policy-seed", type=int, default=100)
    parser.add_argument("--race-policy-seed", type=int, default=200)
    parser.add_argument("--max-turns-per-race", type=int)
    parser.add_argument("--max-events-per-race", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replay:
        game = replay_game_trace(json.loads(args.replay.read_text(encoding="utf-8-sig")))
        print(f"Four-race replay verified: {len(game.races)} races; scores={game.ledger.totals()}")
        return 0
    try:
        config = official_config() if args.official else fixture_config() if args.fixture else load_config(args.config)
        overrides = {}
        if args.max_turns_per_race is not None:
            overrides["max_turns_per_race"] = args.max_turns_per_race
        if args.max_events_per_race is not None:
            overrides["max_events_per_race"] = args.max_events_per_race
        if overrides:
            config = replace(config, **overrides)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    game = GameEngine(
        config, args.seed,
        game_policies={p: RandomGamePolicy(args.game_policy_seed + p) for p in range(4)},
        race_policies={p: RandomPolicy(args.race_policy_seed + p) for p in range(4)},
    )
    kind = "official" if config.official else "fixture"
    output = args.output or Path(f".cache/game-{kind}-seed-{args.seed}.json")
    try:
        result = game.run()
    except GameDiagnosticError as exc:
        write_trace(output, exc.trace)
        print(f"DIAGNOSTIC FAILURE: {exc.reason}; incomplete game; trace={output.resolve()}")
        return 2
    write_trace(output, game.trace())
    if config.official:
        print("OFFICIAL ORIGINAL-BOX GAME: verified Mild/Wild and all 8 prize chips; project HOUSE rulings apply.")
    else:
        print("DIAGNOSTIC FIXTURE: custom/restricted data; NOT an official complete game.")
    for race in result.races:
        print(f"Race {race.race_index} ({race.track_kind} schedule slot): start=P{race.starting_player + 1}; "
              f"places={race.finish_order}; race_points={race.race_scores}; totals={race.scores_after}")
    print(f"Final scores: {result.scores}; co-winners={result.winners}")
    print(f"Trace: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
