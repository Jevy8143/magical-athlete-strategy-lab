"""Recompute three frozen paired-Game score comparisons using only stdlib.

No engine, database, policy, checkpoint or update is loaded. Input rows contain
the actual ten seeds and both arms' four-Race, four-seat official score deltas.
The other three seats are not counted as additional independent Games.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics


ROLES = {
    "native": "primary_native_incremental_learning",
    "random": "exploratory_transfer",
    "public-memory": "exploratory_transfer",
}
PARENT_896 = "1e256714cd662938a812467bfe7d157f7654d3b13b6ad2af1044012635e89243"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def require(value, message):
    if not value:
        raise ValueError(message)


def derive_seeds(master, namespace, index):
    """The existing sha256-domain-v1 EVALUATION stream convention."""
    require(type(master) is int and 0 <= master <= (1 << 63)-2, "invalid master seed")
    require(type(index) is int and index >= 0, "invalid pair index")
    require(type(namespace) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", namespace), "invalid study identifier")
    base = {"master_seed": master, "purpose": "evaluation", "namespace": namespace, "scheme": "sha256-domain-v1"}
    used = set()

    def stream(label, paired=False):
        attempt = 0
        while True:
            raw = {**base, "game_index": index, "stream": label, "collision_attempt": attempt}
            local = int(hashlib.sha256(canonical(raw)).hexdigest(), 16) & ((1 << 60)-1)
            if paired:
                local &= ~1
            value = (1 << 60) | local
            claims = {value, value+1} if paired else {value}
            if not claims.intersection(used):
                used.update(claims)
                return value
            attempt += 1

    game = stream("game-shuffle", paired=True)
    return {"game_seed": game, "dice_seed": game+1,
            "game_policy_seeds": [stream(f"game-policy-seat-{seat}") for seat in range(4)],
            "race_policy_seeds": [stream(f"race-policy-seat-{seat}") for seat in range(4)]}


def paired_statistics(rows, per_seat, master, study):
    """Equal-seat paired bootstrap, matching the frozen evaluation convention."""
    strata = {}
    by_seat = {}
    for seat in range(4):
        own = sorted((row for row in rows if row["focal_seat"] == seat), key=lambda row: row["pair_index"])
        require(len(own) == per_seat, "missing predeclared pairs")
        left = [sum(race[seat] for race in row["left_race_scores"]) for row in own]
        right = [sum(race[seat] for race in row["right_race_scores"]) for row in own]
        strata[seat] = [r-l for l, r in zip(left, right)]
        by_seat[str(seat)] = {"pairs": len(own), "left_mean": statistics.fmean(left),
                             "right_mean": statistics.fmean(right), "mean_difference": statistics.fmean(strata[seat])}
    identity = {"statistic": "seat-stratified-paired-game-bootstrap-v1", "master_seed": master, "study_id": study}
    bootstrap_seed = int(hashlib.sha256(canonical(identity)).hexdigest(), 16) & ((1 << 63)-1)
    rng = random.Random(bootstrap_seed)
    replicates = sorted(statistics.fmean(statistics.fmean(rng.choices(values, k=len(values)))
                                        for values in strata.values()) for _ in range(2000))
    return {"paired_games": 4*per_seat, "left_mean": statistics.fmean(row["left_mean"] for row in by_seat.values()),
            "right_mean": statistics.fmean(row["right_mean"] for row in by_seat.values()),
            "mean_difference_right_minus_left": statistics.fmean(statistics.fmean(values) for values in strata.values()),
            "pointwise_bootstrap_95_percent_interval": [replicates[49], replicates[1949]],
            "bootstrap_replicates": 2000, "bootstrap_seed": bootstrap_seed, "by_seat": by_seat}


def summarize(data):
    require(type(data) is dict and set(data) == {"schema_version", "purpose", "models", "panels", "pairs"}, "unknown dataset fields")
    require(type(data["schema_version"]) is int and data["schema_version"] == 1
            and data["purpose"] == "frozen_eval_score_export", "unsupported score-export schema")
    require(set(data["models"]) == {"left", "right"}, "exact two frozen arms required")
    for arm, count in (("left", 896), ("right", 1024)):
        model = data["models"][arm]
        require(set(model) == {"artifact_sha256", "training_games"} and type(model["training_games"]) is int
                and model["training_games"] == count and type(model["artifact_sha256"]) is str
                and re.fullmatch(r"[0-9a-f]{64}", model["artifact_sha256"]), "invalid frozen model provenance")
    require(data["models"]["left"]["artifact_sha256"] == PARENT_896, "left arm is not the predeclared frozen896 parent")
    require(set(data["panels"]) == set(ROLES), "three declared panels required; panels must not be pooled")
    for panel, setup in data["panels"].items():
        require(type(setup) is dict and set(setup) == {"inference_role", "pairs_per_seat", "master_seed", "study_id", "opponent_snapshot_ids", "actual_context_id", "lookup_context_id", "mean_guard"}, "unknown panel metadata fields")
        identities = [*setup["opponent_snapshot_ids"], setup["actual_context_id"], setup["lookup_context_id"]]
        require(type(setup["opponent_snapshot_ids"]) is list and len(setup["opponent_snapshot_ids"]) == 3
                and all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in identities), "exact opponent and context identities required")
        require(canonical(setup["mean_guard"]) == canonical({"scheme": "public-turn-local-duel-retry-fish-keep-v4", "implementation_sha256": "548f19b411b29924e103344e05310327abd38cbd60d4c363d71c7eb385dbe644"}), "both frozen mean arms must declare the actual V4 binding")
    require(type(data["pairs"]) is list and len(data["pairs"]) == 384, "all three panels need 128 complete pairs")
    seen, all_streams = set(), set()
    for row in data["pairs"]:
        require(type(row) is dict and set(row) == {"panel", "focal_seat", "pair_index", "seeds", "left_race_scores", "right_race_scores"}, "unknown paired-row fields")
        panel, seat, index = row["panel"], row["focal_seat"], row["pair_index"]
        require(panel in ROLES and type(seat) is int and seat in range(4)
                and type(index) is int and seat*32 <= index < (seat+1)*32, "invalid declared pair coordinate")
        identity = panel, seat, index
        require(identity not in seen, "duplicate pair cannot become an extra independent sample")
        seen.add(identity)
        setup = data["panels"][panel]
        require(setup["inference_role"] == ROLES[panel] and type(setup["pairs_per_seat"]) is int
                and setup["pairs_per_seat"] == 32, "predeclared inference role or size changed")
        require(canonical(row["seeds"]) == canonical(derive_seeds(setup["master_seed"], setup["study_id"], index)), "actual ten seeds differ from predeclared stream derivation")
        seeds = row["seeds"]
        values = [seeds["game_seed"], seeds["dice_seed"], *seeds["game_policy_seeds"], *seeds["race_policy_seeds"]]
        require(len(set(values)) == 10 and not all_streams.intersection(values), "distinct paired coordinates must not reuse streams")
        all_streams.update(values)
        for arm in ("left", "right"):
            races = row[arm+"_race_scores"]
            require(type(races) is list and len(races) == 4 and all(type(race) is list and len(race) == 4
                    and all(type(score) is int for score in race) for race in races), "official scores must be four Races by four seats, signed integers")
            require(all(sum(race[player] for race in races) >= 0 for player in range(4)), "final official balances cannot be negative")
    results = {}
    for panel, role in ROLES.items():
        setup = data["panels"][panel]
        own = [row for row in data["pairs"] if row["panel"] == panel]
        results[panel] = {"inference_role": role, **paired_statistics(own, 32, setup["master_seed"], setup["study_id"])}
    return {"schema_version": 1, "models": data["models"], "panels": results,
            "estimand": "Change from frozen 896 to 1024 Games in expected official four-Race focal total, with equal seat weights, separately for each declared opponent panel.",
            "independent_unit": "One paired Game seed and focal seat per row; three other seats are not additional samples.",
            "multiplicity": "95% pointwise intervals; no multiplicity correction. Random and public-memory panels are exploratory.",
            "panel_pooling": False, "Game_or_model_execution": False,
            "scope": "Recomputes the supplied score table; source and official-ledger authenticity are established by the separate private read-only export review."}


def self_test():
    # These are explicitly synthetic arithmetic controls, never exported Game data.
    rows = [{"focal_seat": seat, "pair_index": seat*2+index,
             "left_race_scores": [[-1, -1, -1, -1], [2, 2, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
             "right_race_scores": [[0, 0, 0, 0], [2, 2, 2, 2], [1, 1, 1, 1], [0, 0, 0, 0]]}
            for seat in range(4) for index in range(2)]
    stat = paired_statistics(rows, 2, 11, "synthetic-arithmetic-self-test")
    require(stat["mean_difference_right_minus_left"] == 2 and stat["pointwise_bootstrap_95_percent_interval"] == [2, 2], "constant paired contrast failed")
    require(len(set([derive_seeds(11, "synthetic-seeds", 0)["game_seed"], derive_seeds(11, "synthetic-seeds", 1)["game_seed"]])) == 2, "stream derivation failed")
    return {"synthetic_controls_passed": True, "actual_Game_data_read": False, "Game_database_or_model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        require(args.input is None and args.output is None, "self-test cannot claim actual Game data")
        print(json.dumps(self_test(), sort_keys=True))
        return 0
    if args.input is None:
        parser.error("--input must name a real completed score export")
    raw = args.input.read_bytes()
    result = summarize(json.loads(raw))
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
