"""Versioned observable state groups and legal action identities for model A.

This is a deliberately lossy three-level abstraction, not a Markov-state claim.
Every input comes from an actual player request. Event/decision counters and
physical-card IDs are not strategy features; target seats are actor-relative.
"""

import json
from dataclasses import asdict, dataclass, fields

from .decisions import Action, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .experiment_spec import canonical_json
from .game import DraftPick, DraftRequest, EntryRequest, PublicGameObservation, RacerCard
from .state import RacerStatus
from .track import Track

FEATURE_VERSION = "model-a-observable-groups-v2"
FEATURE_CONFIG = {
    "levels": ["fine", "middle", "coarse"],
    "position_bucket_width": 5,
    "score_bucket_width": 5,
    "gap_bucket_width": 3,
    "progress_bucket_width": 10,
    "exact_finish_distance_through": 6,
    "visible_trip_neighborhood": 6,
    "seat_normalization": "clockwise_offset_from_deciding_player",
}


def _plain(value):
    if isinstance(value, (DraftRequest, EntryRequest, DecisionRequest, Action)):
        value = asdict(value)
    return json.loads(canonical_json(value))


def _known(value, cls, name):
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(cls)}:
        raise ValueError(f"unknown or missing fields in {name}")


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}")
    return value


def _optional_integer(value, name, *, minimum=0, maximum=None):
    return None if value is None else _integer(value, name, minimum=minimum, maximum=maximum)


def _scope(raw):
    if not isinstance(raw, dict):
        raise ValueError("model A requires a typed request or its JSON object")
    return "race" if "window" in raw else "draft" if "round_index" in raw else "entry"


def _card(card):
    _known(card, RacerCard, "visible card")
    rebuilt = RacerCard(**card)
    if not isinstance(rebuilt.card_id, str) or not isinstance(rebuilt.name, str):
        raise ValueError("visible card names and IDs must be strings")


def _normalize_action(action):
    action = _plain(action)
    _known(action, Action, "action")
    rebuilt = Action(**action)
    for name in ("source_id", "target_id"):
        _optional_integer(getattr(rebuilt, name), name, maximum=3)
    return _plain(rebuilt)


def normalize_request(request, *, race_index=None) -> dict:
    """Detach and validate only the public/own-private request schema.

    Structural normalization permits an omitted race context; actual state or
    action encoding requires it for race requests, which lack race_index.
    """
    raw = _plain(request)
    scope = _scope(raw)
    cls = {"draft": DraftRequest, "entry": EntryRequest, "race": DecisionRequest}[scope]
    _known(raw, cls, "request")
    _integer(raw["decision_id"], "decision ID", minimum=1)
    player = _integer(raw["player_id"], "player ID", maximum=3)
    observation = raw["observation"]
    if scope == "race":
        _known(observation, Observation, "race observation")
        DecisionWindow(raw["window"])
        _integer(observation["current_player_id"], "current player", maximum=3)
        _integer(observation["turn_number"], "turn number")
        finish = _integer(observation["finish_position"], "finish", minimum=1)
        _optional_integer(observation["second_corner_position"], "second corner", minimum=1, maximum=finish - 1)
        for position in observation["trip_positions"]:
            _integer(position, "Trip space", maximum=finish - 1)
        racers = observation["racers"]
        if len(racers) != 4 or {racer["racer_id"] for racer in racers} != set(range(4)):
            raise ValueError("model A requires the four standard public racers")
        for racer in racers:
            _known(racer, PublicRacer, "public racer")
            _integer(racer["racer_id"], "racer ID", maximum=3)
            _integer(racer["position"], "racer position", maximum=finish)
            RacerStatus(racer["status"])
            if type(racer["tripped"]) is not bool:
                raise ValueError("tripped must be boolean")
            _optional_integer(racer["finish_place"], "finish place", minimum=1, maximum=2)
            _optional_integer(racer["copied_target"], "copied target", maximum=3)
        totals = observation["score_totals"]
        if len(totals) != 4 or {pair[0] for pair in totals} != set(range(4)):
            raise ValueError("four public score balances are required")
        for seat, balance in totals:
            _integer(seat, "score seat", maximum=3)
            _integer(balance, "score balance")
        for seat in observation["finish_order"]:
            _integer(seat, "finish-order seat", maximum=3)
        _optional_integer(observation["latest_roll"], "latest roll", minimum=1, maximum=6)
        if observation["current_move_value"] is not None and type(observation["current_move_value"]) is not int:
            raise ValueError("move value must be an integer or None")
        for name in ("magician_rerolls_remaining", "dicemonger_rerolls_remaining"):
            _integer(observation[name], name)
        choices = raw["legal_actions"]
        if not choices:
            raise ValueError("there must be a legal action")
        raw["legal_actions"] = [_normalize_action(action) for action in choices]
        if race_index is not None:
            _integer(race_index, "race index", minimum=1, maximum=4)
    else:
        _known(observation, PublicGameObservation, "game observation")
        expected_phase = "draft" if scope == "draft" else "entry_selection"
        if observation["phase"] != expected_phase:
            raise ValueError("request phase differs from its schema")
        actual_race = observation["race_index"]
        _integer(actual_race, "race index", maximum=4)
        if (scope == "draft" and actual_race != 0) or (scope == "entry" and actual_race == 0):
            raise ValueError("request phase differs from its race index")
        if race_index is not None and (type(race_index) is not int or race_index != actual_race):
            raise ValueError("explicit race index differs from the public game request")
        if scope == "draft":
            _integer(raw["round_index"], "draft round", minimum=1, maximum=2)
        for collection in ("scores", "hand_sizes", "used_cards"):
            if len(observation[collection]) != 4:
                raise ValueError("four public player entries are required")
        for balance in observation["scores"]:
            _integer(balance, "score balance")
        for size in observation["hand_sizes"]:
            _integer(size, "hand size", maximum=4)
        _optional_integer(observation["starting_player"], "starting player", maximum=3)
        if observation["current_awards"] is not None:
            if len(observation["current_awards"]) != 2:
                raise ValueError("gold and silver awards are required")
            for value in observation["current_awards"]:
                _integer(value, "placement award")
        if observation["track"] is not None:
            _known(observation["track"], Track, "public track")
            # JSON container ordering is not printed board information. A
            # typed Track and its hand-written JSON geometry normalize alike.
            observation["track"] = json.loads(canonical_json(Track(**observation["track"])))
        cards = raw["hand"] + (raw["available"] if scope == "draft" else [])
        for card in cards:
            _card(card)
        for pick in observation["draft_history"]:
            _known(pick, DraftPick, "public draft pick")
            _integer(pick["round_index"], "public draft round", minimum=1, maximum=2)
            _integer(pick["player_id"], "public drafting player", maximum=3)
            _card(pick["card"])
        legal = raw["legal_card_ids"]
        visible = raw["available"] if scope == "draft" else raw["hand"]
        if not legal or len(legal) != len(set(legal)) or not set(legal).issubset({card["card_id"] for card in visible}):
            raise ValueError("legal card choices must be distinct visible cards")
        for order in observation["previous_finish_orders"]:
            for seat in order:
                _integer(seat, "previous finisher", maximum=3)
    return raw


def request_scope(request) -> str:
    return _scope(normalize_request(request))


def legal_choices(request) -> tuple:
    """JSON-compatible choices in original legal order, detached from caller."""
    raw = normalize_request(request)
    return tuple(raw["legal_actions"] if "window" in raw else raw["legal_card_ids"])


def _race_index(raw, scope, race_index):
    if scope == "race":
        if race_index is None:
            raise ValueError("race requests require an explicit race_index")
        return _integer(race_index, "race index", minimum=1, maximum=4)
    return raw["observation"]["race_index"]


def _relative(seat, player):
    return None if seat is None else (seat - player) % 4


def _card_semantics(card):
    return {"name": card["name"], "ability": card["ability"]}


def _cards_semantics(cards):
    return sorted((_card_semantics(card) for card in cards), key=canonical_json)


def _target(raw, seat):
    if seat is None:
        return None
    racer = next(racer for racer in raw["observation"]["racers"] if racer["racer_id"] == seat)
    return {"seat": _relative(seat, raw["player_id"]), "name": racer["name"], "ability": racer["ability"]}


def _action_identity(raw, scope, action):
    if scope == "race":
        return {"kind": action["kind"], "source": _target(raw, action["source_id"]),
                "target": _target(raw, action["target_id"]), "value": action["value"]}
    cards = raw["available"] if scope == "draft" else raw["hand"]
    card = next(card for card in cards if card["card_id"] == action)
    return {"kind": "choose_card", **_card_semantics(card)}


def action_key(request, action, *, race_index=None) -> str:
    raw = normalize_request(request, race_index=race_index)
    scope = _scope(raw)
    _race_index(raw, scope, race_index)
    choice = _normalize_action(action) if scope == "race" else action
    legal = raw["legal_actions"] if scope == "race" else raw["legal_card_ids"]
    if choice not in legal:
        raise ValueError("model A cannot encode an illegal action")
    return FEATURE_VERSION + "|action|" + canonical_json(_action_identity(raw, scope, choice))


def _gap(value):
    sign = -1 if value < 0 else 1
    return sign * min(5, abs(value) // FEATURE_CONFIG["gap_bucket_width"])


def _relation(position, reference):
    return "ahead" if position > reference else "behind" if position < reference else "same"


def _distance_band(distance):
    """Keep immediate interaction distances exact, aggregate distant gaps."""
    if abs(distance) <= FEATURE_CONFIG["exact_finish_distance_through"]:
        return {"exact": distance}
    return {"band": _gap(distance)}


def _without_balances(value):
    """Geometry repetition may continue while an opponent farms more points.

    This projection is for policy memory only. Value state groups retain all
    actual score features; no engine effect or legal window is suppressed.
    """
    if isinstance(value, dict):
        return {key: _without_balances(item) for key, item in value.items()
                if key not in {"scores", "score_band", "score_gap"}}
    if isinstance(value, list):
        return [_without_balances(item) for item in value]
    return value


def _public_scores(raw, scope):
    player = raw["player_id"]
    observation = raw["observation"]
    totals = dict(observation["score_totals"]) if scope == "race" else observation["scores"]
    return tuple(totals[(player + offset) % 4] for offset in range(4))


def _public_game_history(observation, cards, player):
    known = {card["card_id"]: _card_semantics(card) for card in cards}
    for pick in observation["draft_history"]:
        known[pick["card"]["card_id"]] = _card_semantics(pick["card"])

    def resolve(card_id):
        if card_id not in known:
            raise ValueError("public history references a card without visible semantics")
        return known[card_id]

    return {
        "used": [[resolve(card) for card in observation["used_cards"][(player + offset) % 4]] for offset in range(4)],
        "previous_entries": [[resolve(card) for card in entries[player:] + entries[:player]]
                             for entries in observation["previous_entries"]],
        "previous_finishes": [[_relative(seat, player) for seat in order]
                              for order in observation["previous_finish_orders"]],
        "drafted": [{"seat": _relative(pick["player_id"], player), "card": _card_semantics(pick["card"])}
                    for pick in observation["draft_history"]],
    }


def _groups(raw, scope, race_index):
    observation, player = raw["observation"], raw["player_id"]
    if scope != "race":
        scores = observation["scores"]
        track = observation["track"]
        coarse = {"scope": scope, "race_index": race_index,
                  "draft_round": raw.get("round_index"), "track": track["name"] if track else None}
        middle = {**coarse, "hand": _cards_semantics(raw["hand"]),
                  "hand_sizes": [observation["hand_sizes"][(player + offset) % 4] for offset in range(4)],
                  "score_band": scores[player] // FEATURE_CONFIG["score_bucket_width"],
                  "score_gap": _gap(scores[player] - max(scores)),
                  "starting_offset": _relative(observation["starting_player"], player),
                  "awards": observation["current_awards"]}
        visible = raw["available"] if scope == "draft" else raw["hand"]
        fine = {**middle, "available": _cards_semantics(visible),
                "scores": [scores[(player + offset) % 4] for offset in range(4)],
                "legal_cards": sorted((_action_identity(raw, scope, choice) for choice in raw["legal_card_ids"]), key=canonical_json),
                "history": _public_game_history(observation, raw["hand"] + visible, player),
                "track_data": None if track is None else {key: value for key, value in track.items()
                                                         if key not in {"source", "data_version"}}}
        return fine, middle, coarse
    racers = {racer["racer_id"]: racer for racer in observation["racers"]}
    own, finish = racers[player], observation["finish_position"]
    totals = dict(observation["score_totals"])
    coarse = {"scope": scope, "race_index": race_index, "track": observation["track_name"],
              "window": raw["window"], "self_role": own["name"], "self_ability": own["ability"],
              "roll": observation["latest_roll"], "move": observation["current_move_value"],
              "magician_remaining": observation["magician_rerolls_remaining"],
              "dicemonger_remaining": observation["dicemonger_rerolls_remaining"],
              # Sparse fallback must not merge standing/recovering, isolated/
              # shared spaces, or an unplaced pack with a race awaiting silver.
              "self_tripped": own["tripped"], "self_status": own["status"],
              "finished_count": len(observation["finish_order"]),
              "sharing": any(r["status"] == "active" and r["position"] == own["position"]
                             for rid, r in racers.items() if rid != player)}
    copy_targets = None
    if raw["window"] == "copy_choice":
        candidates = {action["target_id"] for action in raw["legal_actions"] if action["kind"] == "copy_target"}
        copy_targets = sorted((
            {"role": racers[seat]["name"], "ability": racers[seat]["ability"],
             "relation": _relation(racers[seat]["position"], own["position"])}
            for seat in candidates
        ), key=canonical_json)
        coarse["can_keep_copy"] = any(action["kind"] == "keep" for action in raw["legal_actions"])
    middle = {**coarse, "position_band": own["position"] // FEATURE_CONFIG["position_bucket_width"],
              "current_player_offset": _relative(observation["current_player_id"], player),
              "progress_band": own["position"] // FEATURE_CONFIG["progress_bucket_width"],
              "near_finish": finish - own["position"] <= FEATURE_CONFIG["exact_finish_distance_through"],
              "space": {
                  "active_opponents": sum(r["status"] == "active" for rid, r in racers.items() if rid != player),
                  "ahead": sum(r["status"] == "active" and r["position"] > own["position"]
                               for rid, r in racers.items() if rid != player),
                  "same": sum(r["status"] == "active" and r["position"] == own["position"]
                              for rid, r in racers.items() if rid != player),
                  "tripped_opponents": sum(r["status"] == "active" and r["tripped"]
                                           for rid, r in racers.items() if rid != player),
              },
              "finish_distance_band": (finish - own["position"]) // FEATURE_CONFIG["position_bucket_width"],
              "score_band": totals[player] // FEATURE_CONFIG["score_bucket_width"],
              "score_gap": _gap(totals[player] - max(totals.values())),
              "active_count": sum(racer["status"] == "active" for racer in racers.values()),
              "position_gaps": [_gap(racers[(player + offset) % 4]["position"] - own["position"])
                                for offset in range(1, 4)],
              "finish_distance": _distance_band(finish - own["position"]),
              "on_trip": own["position"] in observation["trip_positions"],
              "trip_offsets": sorted(position - own["position"] for position in observation["trip_positions"]
                                     if abs(position - own["position"]) <= FEATURE_CONFIG["visible_trip_neighborhood"]),
              "second_corner_side": None if observation["second_corner_position"] is None else
                                    _relation(own["position"], observation["second_corner_position"]),
              "copied_target": _relative(own["copied_target"], player),
              "opponents": [
                  {"role": racers[(player + offset) % 4]["name"],
                   "ability": racers[(player + offset) % 4]["ability"],
                   "status": racers[(player + offset) % 4]["status"],
                   "tripped": racers[(player + offset) % 4]["tripped"],
                   "relation": _relation(racers[(player + offset) % 4]["position"], own["position"]),
                   "distance": _distance_band(racers[(player + offset) % 4]["position"] - own["position"])}
                  for offset in range(1, 4)
              ]}
    if copy_targets is not None:
        middle["copy_targets"] = copy_targets
    public_racers = []
    for offset in range(4):
        racer = racers[(player + offset) % 4]
        public_racers.append({key: value for key, value in racer.items() if key not in {"racer_id", "copied_target"}}
                             | {"seat": offset, "copied_target": _relative(racer["copied_target"], player)})
    fine = {**middle, "racers": public_racers,
            "scores": [totals[(player + offset) % 4] for offset in range(4)],
            "finish_order": [_relative(seat, player) for seat in observation["finish_order"]],
            "finish": finish, "second_corner": observation["second_corner_position"],
            "trip_positions": sorted(observation["trip_positions"]),
            "legal_actions": sorted((_action_identity(raw, scope, action) for action in raw["legal_actions"]), key=canonical_json)}
    return fine, middle, coarse


def state_keys(request, *, race_index=None) -> tuple[str, ...]:
    raw = normalize_request(request, race_index=race_index)
    scope = _scope(raw)
    index = _race_index(raw, scope, race_index)
    return tuple(FEATURE_VERSION + "|" + level + "|" + canonical_json(group)
                 for level, group in zip(FEATURE_CONFIG["levels"], _groups(raw, scope, index)))


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    scope: str
    race_index: int
    state_keys: tuple[str, ...]
    actions: tuple[str | Action, ...]
    action_keys: tuple[str, ...]
    public_scores: tuple[int, ...]
    repetition_key: str


def encode_request(request, *, race_index=None) -> EncodedDecision:
    raw = normalize_request(request, race_index=race_index)
    scope = _scope(raw)
    index = _race_index(raw, scope, race_index)
    choices = raw["legal_actions"] if scope == "race" else raw["legal_card_ids"]
    actions = tuple(Action(**choice) for choice in choices) if scope == "race" else tuple(choices)
    abstractions = _groups(raw, scope, index)
    groups = tuple(FEATURE_VERSION + "|" + level + "|" + canonical_json(group)
                   for level, group in zip(FEATURE_CONFIG["levels"], abstractions))
    identities = tuple(FEATURE_VERSION + "|action|" + canonical_json(_action_identity(raw, scope, choice))
                       for choice in choices)
    repetition = FEATURE_VERSION + "|repetition|" + canonical_json(_without_balances(abstractions[0]))
    return EncodedDecision(scope, index, groups, actions, identities, _public_scores(raw, scope), repetition)
