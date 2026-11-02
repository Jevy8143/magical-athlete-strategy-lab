"""A database-free, seeded original-box game and strict replay demonstration.

Run ``python -m magical_athlete.demo --seed 1 --output demo-output`` after
installing this package. All four agents choose only from public legal menus.
The demo does not train a model or make a performance claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .decisions import RandomPolicy
from .engine import replay_trace
from .game import GameDiagnosticError, GameEngine, RandomGamePolicy, official_config, replay_game_trace
from .scoring import ScoreLedger


DEFAULT_SEED = 1


def canonical_json(value) -> bytes:
    """Canonical JSON also distinguishes true/1 while normalizing tuples/lists."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def seed_streams(seed: int) -> dict:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "scheme": "demo-offset-seeds-v1",
        "game_seed": seed,
        "dice_seed": seed + 1,
        "game_policy_seeds": [seed + 99 + player for player in range(4)],
        "race_policy_seeds": [seed + 199 + player for player in range(4)],
    }


def verify_trace(trace: dict) -> dict:
    """Replay each Race and the complete Game, including all score entries."""
    if type(trace) is not dict:
        raise ValueError("trace must be a JSON object")
    races = trace.get("race_traces")
    if trace.get("completed") is not True or type(races) is not list or len(races) != 4:
        raise ValueError("strict replay requires a completed four-race Game")
    ledger = ScoreLedger.from_trace(trace["scores"])
    if canonical_json(ledger.trace()) != canonical_json(trace["scores"]):
        raise ValueError("score ledger differs from its actual entries")
    for race in trace["race_traces"]:
        if canonical_json(replay_trace(race).trace()) != canonical_json(race):
            raise ValueError("strict Race replay diverged")
    if canonical_json(replay_game_trace(trace).trace()) != canonical_json(trace):
        raise ValueError("strict Game replay diverged")
    return {"score_ledger_verified": True, "individual_races_verified": 4, "whole_game_verified": True}


def summarize_trace(trace: dict, verification: dict) -> dict:
    """Keep placement chips, Star points and skill deltas separately visible."""
    ledger = trace["scores"]
    races = []
    for race in trace["races"]:
        race_index = race["race_index"]
        entries = [entry for entry in ledger["entries"] if entry["race_index"] == race_index]
        breakdown = []
        for player in range(4):
            own = [entry for entry in entries if entry["player_id"] == player]
            placement = sum(entry["delta"] for entry in own if entry["reason"].startswith("placement_"))
            star = sum(entry["delta"] for entry in own if entry["reason"] == "star")
            skills = sum(entry["delta"] for entry in own if not entry["reason"].startswith("placement_") and entry["reason"] != "star")
            total = sum(entry["delta"] for entry in own)
            if total != race["race_scores"][player] or placement + star + skills != total:
                raise ValueError("race score breakdown does not reconcile")
            breakdown.append({"player_id": player, "placement": placement, "star": star, "skills": skills, "total": total})
        races.append({
            "race_index": race_index, "track": race["track_kind"],
            "entries": [card["card_id"] for card in race["entries"]],
            "finish_order": race["finish_order"], "scores": race["race_scores"],
            "scores_after": race["scores_after"], "score_breakdown": breakdown,
        })
    return {
        "schema_version": 1, "purpose": "reproducible_demo", "completed": trace["completed"],
        "official_original_box": trace["config"]["official"], "rule_version": trace["rule_version"],
        "seed": trace["seed"], "dice_seed": trace["dice_seed"],
        "game_policies": trace["game_policies"], "race_policies": trace["race_traces"][0]["policies"],
        "races": races, "total_scores": [ledger["totals"][str(player)] for player in range(4)],
        "winners": trace["winners"], "score_ledger": ledger,
        "verification": verification, "code_fingerprint": trace["code_fingerprint"],
        "trace_sha256": hashlib.sha256(canonical_json(trace)).hexdigest(),
        "interpretation": "One seeded demonstration; no training or evidence of strategy strength.",
    }


def run_demo(seed: int = DEFAULT_SEED) -> tuple[dict, dict]:
    """Use separate deterministic game, dice and four-seat policy streams."""
    seeds = seed_streams(seed)
    engine = GameEngine(
        official_config(), seeds["game_seed"],
        game_policies={player: RandomGamePolicy(seeds["game_policy_seeds"][player]) for player in range(4)},
        race_policies={player: RandomPolicy(seeds["race_policy_seeds"][player]) for player in range(4)},
    )
    engine.run()
    trace = engine.trace()
    summary = summarize_trace(trace, verify_trace(trace))
    summary["seed_streams"] = seeds
    return summary, trace


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="nonnegative seed (default: 1)")
    parser.add_argument("--output", type=Path, help="new or empty evidence directory (default for a new game: demo-output)")
    parser.add_argument("--replay", type=Path, help="strictly replay an existing trace instead of choosing new actions")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    output = args.output if args.output is not None else (None if args.replay else Path("demo-output"))
    if output:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            parser.error("--output must name a new or empty directory; existing evidence is never overwritten")
        output.mkdir(parents=True, exist_ok=True)
    try:
        if args.replay:
            with args.replay.open(encoding="utf-8") as stream:
                trace = json.load(stream)
            summary = summarize_trace(trace, verify_trace(trace))
        else:
            summary, trace = run_demo(args.seed)
    except GameDiagnosticError as error:
        failure = {"completed": False, "diagnostic_failure": error.reason,
                   "interpretation": "Incomplete diagnostic; no result or placement was invented."}
        if output:
            _write_json(output / "trace.json", error.trace)
            _write_json(output / "summary.json", failure)
        else:
            failure["trace"] = error.trace
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True, indent=2))
        return 2
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"demo: {error}\n")
    if output:
        _write_json(output / "trace.json", trace)
        _write_json(output / "summary.json", summary)
    display = {"seed": summary["seed"], "completed": True,
        "tracks": [race["track"] for race in summary["races"]],
        "race_scores": [race["scores"] for race in summary["races"]],
        "total_scores": summary["total_scores"], "winners": summary["winners"],
        "strict_replay": summary["verification"]}
    if output:
        display["files"] = {"summary": (output / "summary.json").as_posix(), "trace": (output / "trace.json").as_posix()}
    print(json.dumps(display, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
