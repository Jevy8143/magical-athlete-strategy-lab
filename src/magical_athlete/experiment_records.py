"""Turn auditable game traces into raw records and player-local transitions.

Raw events remain an audit source; model inputs below are copied only from the
requests the engine actually delivered. Successor states belong to the same
player, across other players, automatic effects and race boundaries. Extraction
does not authenticate a trace: the experiment store must verify/replay it first.
"""

import json
from dataclasses import fields

from .decisions import Action, DecisionRequest, Observation, PublicRacer
from .game import DraftPick, DraftRequest, EntryRequest, PublicGameObservation, RacerCard
from .scoring import ScoreLedger
from .track import Track


RECORD_SCHEMA_VERSION = 1


def _plain(value):
    """Detach every nested container without accepting non-JSON payloads."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _known_fields(value, cls, name):
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
        raise ValueError(f"unexpected fields in {name}; observation schema must be explicit")


def _decision(raw, scope, race_index):
    request = _plain(raw["request"])
    if scope == "race":
        _known_fields(request, DecisionRequest, "race decision request")
        _known_fields(request["observation"], Observation, "race observation")
        legal = request["legal_actions"]
        action = _plain(raw["action"])
        # Reject unknown/invalid action fields as well as an illegal selection.
        _known_fields(action, Action, "selected action")
        Action(**action)
        for choice in legal:
            _known_fields(choice, Action, "legal action")
            Action(**choice)
        for racer in request["observation"]["racers"]:
            _known_fields(racer, PublicRacer, "public racer")
        totals = request["observation"]["score_totals"]
        if len(totals) != 4 or {pair[0] for pair in totals} != set(range(4)):
            raise ValueError("a game observation must contain four public score totals")
        scores = dict(totals)
        event_sequence = _integer(raw["event_sequence"], "event sequence", minimum=1)
    else:
        _known_fields(request, DraftRequest if scope == "draft" else EntryRequest, "game decision request")
        _known_fields(request["observation"], PublicGameObservation, "game observation")
        for card in request["hand"] + (request["available"] if scope == "draft" else []):
            _known_fields(card, RacerCard, "visible card")
        for pick in request["observation"]["draft_history"]:
            _known_fields(pick, DraftPick, "public draft pick")
            _known_fields(pick["card"], RacerCard, "public drafted card")
        if request["observation"]["track"] is not None:
            _known_fields(request["observation"]["track"], Track, "public track")
        legal = request["legal_card_ids"]
        action = raw["card_id"]
        scores = request["observation"]["scores"]
        if len(scores) != 4:
            raise ValueError("a game observation must contain four public score totals")
        event_sequence = None
    player = _integer(request["player_id"], "player ID", maximum=3)
    local_id = _integer(request["decision_id"], "decision ID", minimum=1)
    if not legal or action not in legal:
        raise ValueError("decision action is not in its recorded legal actions")
    score = _integer(scores[player], "score before decision")
    return {
        "decision_seq": None, "scope": scope, "race_index": race_index,
        "local_decision_id": local_id, "player_id": player,
        "request": request, "legal_actions": _plain(legal), "action": _plain(action),
        "event_sequence": event_sequence, "score_before": score,
        "batch_id": f"entry-race-{race_index}" if scope == "entry" else None,
        "behavior_probability": None,
        "behavior_probability_source": "unavailable_from_engine_trace",
    }


def _ordered_decisions(trace, race_traces):
    decisions = []
    game_ids = []
    for raw in trace["decisions"]:
        observation = raw["request"]["observation"]
        phase = observation["phase"]
        if phase == "draft":
            scope, race_index = "draft", 0
            if observation["race_index"] != 0:
                raise ValueError("draft decisions must precede race 1")
        elif phase == "entry_selection":
            scope = "entry"
            race_index = _integer(observation["race_index"], "entry race index", minimum=1, maximum=4)
        else:
            raise ValueError("unknown game decision phase")
        row = _decision(raw, scope, race_index)
        game_ids.append(row["local_decision_id"])
        decisions.append(row)
    if game_ids != list(range(1, len(game_ids) + 1)):
        raise ValueError("game decision IDs must be contiguous in call order")
    for race_index, raw in race_traces.items():
        local = [_decision(choice, "race", race_index) for choice in raw["decisions"]]
        if [row["local_decision_id"] for row in local] != list(range(1, len(local) + 1)):
            raise ValueError("race decision IDs must be contiguous in call order")
        decisions.extend(local)
    decisions.sort(key=lambda row: (
        0 if row["scope"] == "draft" else 2 * row["race_index"] - (row["scope"] == "entry"),
        row["local_decision_id"],
    ))
    for seq, row in enumerate(decisions, 1):
        row["decision_seq"] = seq
    return decisions


def _transitions(decisions, final_scores):
    by_player = {player: [] for player in range(4)}
    for decision in decisions:
        by_player[decision["player_id"]].append(decision)
    transitions = []
    for player, player_decisions in by_player.items():
        for index, current in enumerate(player_decisions):
            following = player_decisions[index + 1] if index + 1 < len(player_decisions) else None
            next_score = following["score_before"] if following is not None else final_scores[player]
            transitions.append({
                "decision_seq": current["decision_seq"], "player_id": player,
                "scope": current["scope"], "race_index": current["race_index"],
                "request": _plain(current["request"]),
                "legal_actions": _plain(current["legal_actions"]), "action": _plain(current["action"]),
                "score_before": current["score_before"], "next_score": next_score,
                "next_decision_seq": following["decision_seq"] if following is not None else None,
                "next_request": _plain(following["request"]) if following is not None else None,
                "next_legal_actions": _plain(following["legal_actions"]) if following is not None else [],
                "reward": next_score - current["score_before"], "done": following is None,
                "return_to_go": final_scores[player] - current["score_before"],
                "final_total": final_scores[player],
            })
    return sorted(transitions, key=lambda row: row["decision_seq"])


def build_game_records(trace: dict) -> dict:
    """Extract JSON records; incomplete games never acquire terminal labels.

    Entry choices share a public snapshot and batch ID. Their call order is
    retained solely for audit ordering, without exposing pending choices in the
    request. Unknown request/observation fields fail closed instead of becoming
    accidental model features. Purpose, model lineage and probabilities are
    attached by the outer experiment layer, which knows the actual policies.
    """
    trace = _plain(trace)
    if type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
        raise ValueError("unsupported game trace schema")
    if type(trace.get("completed")) is not bool:
        raise ValueError("trace completed flag must be boolean")
    ledger = ScoreLedger.from_trace(trace["scores"])
    if ledger.player_ids != (0, 1, 2, 3):
        raise ValueError("experiment records require the four standard seats")
    completed_races = {race["race_index"]: race for race in trace["races"]}
    if len(completed_races) != len(trace["races"]) or sorted(completed_races) != list(range(1, len(completed_races) + 1)):
        raise ValueError("completed races must have unique contiguous indices")
    race_traces = {}
    for race_trace in trace["race_traces"]:
        race_index = _integer(race_trace["race_index"], "race index", minimum=1, maximum=4)
        if race_index in race_traces:
            raise ValueError("duplicate completed race trace")
        race_traces[race_index] = race_trace
    if set(race_traces) != set(completed_races):
        raise ValueError("completed race summaries and traces do not match")
    diagnostic = trace.get("diagnostic_failure") or trace.get("race_diagnostic_reason")
    # A constructor can fail before active_race is assigned. Prefer the caught
    # failure trace, including a partially resolved before-race power.
    partial = trace.get("race_diagnostic") or trace.get("active_race")
    if partial is not None:
        race_index = _integer(partial["race_index"], "partial race index", minimum=1, maximum=4)
        if race_index in race_traces:
            raise ValueError("active race duplicates a completed race")
        race_traces[race_index] = partial
        diagnostic = diagnostic or partial.get("diagnostic_failure")
    complete = trace["completed"] and diagnostic is None and partial is None
    if trace["completed"] and not (
        complete and len(completed_races) == 4
        and all(raw["completed"] for raw in race_traces.values())
    ):
        raise ValueError("completed game must contain four complete races without diagnostics")
    decisions = _ordered_decisions(trace, race_traces)
    events, races = [], []
    for race_index, raw in sorted(race_traces.items()):
        record = completed_races.get(race_index)
        for expected_seq, event in enumerate(raw["events"], 1):
            if event["sequence"] != expected_seq:
                raise ValueError("race event sequences must be contiguous")
            if event["parent_sequence"] is not None and not 0 < event["parent_sequence"] < expected_seq:
                raise ValueError("event parent must reference an earlier event in the same race")
            events.append({"race_index": race_index, **event})
        before = ScoreLedger.from_trace(raw["initial_score_ledger"])
        after = [sum(entry.delta for entry in ledger.entries if entry.player_id == p and entry.race_index <= race_index)
                 for p in range(4)]
        positions = record["final_positions"] if record is not None else [
            next(r["position"] for r in raw["racers"] if r["racer_id"] == p) for p in range(4)
        ]
        races.append({
            "race_index": race_index, "status": "completed" if record is not None else "diagnostic",
            "track_kind": "Mild" if race_index % 2 else "Wild", "track": _plain(raw["track"]),
            "entries": _plain(record["entries"] if record is not None else trace["active_entries"]),
            "starting_player": record["starting_player"] if record is not None else raw["turn_order"][0],
            "turn_order": _plain(raw["turn_order"]),
            "finish_order": _plain(record["finish_order"] if record is not None else raw["finish_order"]),
            "final_positions": _plain(positions),
            "race_scores": [ledger.race_total(p, race_index) for p in range(4)],
            "scores_before": [before.total(p) for p in range(4)], "scores_after": after,
            "termination_reason": record["termination_reason"] if record is not None else raw["termination_reason"],
            "turn_count": len(raw["turns"]), "event_count": len(raw["events"]),
            "decision_count": len(raw["decisions"]),
            "diagnostic_reason": None if record is not None else diagnostic or "unfinished_race",
        })
    final_scores = [ledger.total(p) for p in range(4)] if complete else None
    if complete:
        if len([row for row in decisions if row["scope"] == "draft"]) != 16:
            raise ValueError("completed game must contain all sixteen draft choices")
        for race_index in range(1, 5):
            choices = [row for row in decisions if row["scope"] == "entry" and row["race_index"] == race_index]
            if sorted(row["player_id"] for row in choices) != list(range(4)):
                raise ValueError("completed race must contain one entry choice per player")
            if any(row["request"]["observation"] != choices[0]["request"]["observation"] for row in choices):
                raise ValueError("simultaneous entry choices must share one public snapshot")
        if list(trace["winners"]) != list(ledger.winners()):
            raise ValueError("recorded game winners disagree with actual score totals")
    scores = _plain(trace["scores"]["entries"])
    game = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "status": "completed" if complete else "diagnostic", "completed": complete,
        "official": trace["config"]["official"], "seed": trace["seed"], "dice_seed": trace["dice_seed"],
        "rule_version": trace["rule_version"], "code_fingerprint": trace["code_fingerprint"],
        "role_data_versions": sorted({raw["role_data_version"] for raw in race_traces.values()}),
        "track_data_versions": {kind: trace["config"][name]["data_version"]
                                for kind, name in (("Mild", "mild_track"), ("Wild", "wild_track"))},
        "chip_data_version": trace["config"]["award_schedule"].get("data_version", "custom-v1"),
        "observed_scores": [ledger.total(p) for p in range(4)], "final_scores": final_scores,
        "race_scores": [race["race_scores"] for race in races],
        "scores_after_races": [race["scores_after"] for race in races],
        "winners": list(ledger.winners()) if complete else [],
        "champion_counting": "Every highest-scoring player counts as a co-winner; games remain the sampling unit",
        "diagnostic_reason": None if complete else diagnostic or "unfinished_game",
        "counts": {"completed_races": len(completed_races), "logged_races": len(races),
                   "decisions": len(decisions), "events": len(events), "score_entries": len(scores)},
    }
    return {
        "game": game, "races": races, "decisions": decisions,
        "events": events, "scores": scores,
        "transitions": _transitions(decisions, final_scores) if complete else [],
    }
