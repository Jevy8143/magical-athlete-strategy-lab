"""Read-only authentication of original powers and real Egg/Twin acquisition.

This is provenance verification, not a way to copy an inherited skill. Only
the original carrier's real BEFORE_RACE decisions and immediately following
inherit events can grant a static power. A Copy Cat remains a Copy Cat even
when its current borrowed ability happens to equal the requested power.
"""

from dataclasses import asdict
import json

from .decisions import Action, ActionKind, DecisionWindow, Observation, PublicRacer
from .events import RaceEvent
from .racer import Ability


_CARRIERS = frozenset({Ability.EGG, Ability.TWIN})
_NORMAL_TURN_EVENTS = frozenset({"roll", "roll_locked", "main_move_fixed", "main_move_resolved",
                                 "main_move_skipped", "recover", "prediction", "next_turn_requested"})
_NORMAL_TURN_WINDOWS = frozenset({DecisionWindow.REROLL, DecisionWindow.MAIN_MOVE_MODIFIER,
                                  DecisionWindow.BEFORE_MAIN_MOVE, DecisionWindow.TURN_START})


def _require(condition):
    if not condition:
        raise ValueError("static inheritance evidence differs")


def _roster(engine):
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    _require(len(initial) == len(engine.initial_racers) == len(engine.racers) == 4
             and set(initial) == set(engine.racers)
             and len(engine.turn_order) == 4 and set(engine.turn_order) == set(initial))
    raw = []
    for rid, actor in engine.racers.items():
        first = initial[rid]
        _require(type(rid) is int and actor.racer_id == rid and isinstance(actor.ability, Ability)
                 and Ability(first["ability"]) == actor.ability and first["name"] == actor.name)
        raw.append(actor.ability)
    _require(len(set(raw)) == 4)
    return initial


def _acquisition(engine, initial):
    """Rebuild all actual carrier choices in global Egg draw order."""
    physical = {actor.ability for actor in engine.racers.values()}
    deck = list(engine.initial_remaining_abilities)
    previous = tuple(engine.previous_winners)
    _require(all(isinstance(power, Ability) for power in (*deck, *previous))
             and len(set(deck)) == len(deck) and len(set(previous)) == len(previous)
             and not physical.intersection(deck) and not physical.intersection(previous)
             and not set(deck).intersection(previous))
    states = {rid: dict(item) for rid, item in initial.items()}
    leaves = {rid: actor.ability for rid, actor in engine.racers.items()}
    inherited = {rid: None for rid in initial}
    visited = {rid: set() for rid in initial}
    stopped, expected_inherits, choices, inherit_events, drawn = set(), {}, [], [], []
    for rid, item in initial.items():
        _require(item["inherited_ability"] is None)
        if leaves[rid] in _CARRIERS:
            _require(item["pre_race_resolved"] is False)
    records = {}
    for index, record in enumerate(engine.decisions, 1):
        _require(record.request.decision_id == index and type(record.event_sequence) is int
                 and 1 <= record.event_sequence <= len(engine.events)
                 and record.event_sequence not in records)
        records[record.event_sequence] = record
    score_ids = tuple(engine.initial_score_ledger["player_ids"])
    totals = {int(rid): value for rid, value in engine.initial_score_ledger["totals"].items()}
    _require(set(score_ids) == set(initial) == set(totals))
    finish = []

    def public_observation():
        racers = []
        for rid in engine.turn_order:
            item = states[rid]
            power = leaves[rid]
            if power == Ability.COPY_CAT:
                power = item["copied_ability"]
            racers.append(PublicRacer(rid, item["name"], item["position"], str(item["status"]),
                                     power.value if isinstance(power, Ability) else power,
                                     item["tripped"], item["finish_place"], item["copied_target"]))
        return Observation(engine.turn_order[0], 0, engine.track.name, engine.track.finish_position,
                           engine.track.second_corner_position, tuple(sorted(engine.track.trip_positions)),
                           tuple(racers), tuple(finish), None, None,
                           score_totals=tuple((rid, totals[rid]) for rid in score_ids))

    normal_started = False
    for index, event in enumerate(engine.events, 1):
        _require(isinstance(event, RaceEvent) and event.sequence == index)
        if (event.kind in _NORMAL_TURN_EVENTS
                or event.kind in {"ability_effect", "move", "warp"} and event.ability == Ability.PARTY_ANIMAL.value
                or event.kind == "score" and event.reason == "alone_last_at_turn_start"):
            normal_started = True
        if event.kind == "decision" and index in records:
            record = records[index]
            if record.request.window in _NORMAL_TURN_WINDOWS:
                normal_started = True
            if record.request.window == DecisionWindow.BEFORE_RACE:
                _require(not normal_started)
                rid = record.request.player_id
                _require(rid in initial and type(rid) is int and rid == event.racer_id
                         and engine.racers[rid].ability in _CARRIERS and rid not in stopped)
                power = leaves[rid]
                _require(power in _CARRIERS and power not in visited[rid])
                visited[rid].add(power)
                if power == Ability.EGG:
                    # The current engine requires three actual cards; no
                    # partial draw is manufactured to authenticate a trace.
                    _require(len(deck) >= 3)
                    cards, deck = tuple(deck[:3]), deck[3:]
                    drawn.extend(cards)
                    legal = tuple(Action(ActionKind.EGG_CHOOSE, rid, rid, value=card.value) for card in cards)
                else:
                    legal = (Action(ActionKind.KEEP), *(Action(ActionKind.TWIN_CHOOSE, rid, rid, value=card.value)
                                                       for card in previous))
                    _require(len(legal) > 1)
                action = record.action
                _require(record.request.legal_actions == legal and action in legal
                         and record.request.observation == public_observation()
                         and event == RaceEvent(index, "decision", rid, source_id=action.source_id,
                                                target_id=action.target_id, action_kind=action.kind.value,
                                                decision_id=record.request.decision_id))
                choices.append(record)
                if action.kind == ActionKind.KEEP:
                    _require(power == Ability.TWIN)
                    stopped.add(rid)
                else:
                    following = Ability(action.value)
                    expected = RaceEvent(index + 1, "inherit", rid, source_id=rid,
                                         ability=following.value, parent_sequence=index)
                    _require(index < len(engine.events) and engine.events[index] == expected)
                    expected_inherits[index + 1] = (rid, following)
        if event.kind == "inherit":
            _require(not normal_started and index in expected_inherits)
            rid, following = expected_inherits[index]
            _require(event.racer_id == rid)
            leaves[rid], inherited[rid] = following, following
            states[rid]["inherited_ability"] = following
            inherit_events.append(event)
        elif event.kind == "copy_changed":
            _require(event.racer_id in states)
            states[event.racer_id]["copied_target"] = event.target_id
            states[event.racer_id]["copied_ability"] = event.ability
        elif event.kind in {"move", "warp", "stop"}:
            _require(event.racer_id in states)
            states[event.racer_id]["position"] = event.after
        elif event.kind in {"trip", "recover"}:
            states[event.racer_id]["tripped"] = event.kind == "trip"
        elif event.kind == "finish":
            item = states[event.racer_id]
            item["status"] = "finished"
            if item["finish_place"] is None:
                item["finish_place"] = event.amount
            finish.append(event.racer_id)
        elif event.kind == "eliminate":
            states[event.racer_id]["status"] = "eliminated"
        elif event.kind == "score":
            _require(event.racer_id in totals and event.before == totals[event.racer_id])
            totals[event.racer_id] = event.after
    _require(len(expected_inherits) == len(inherit_events)
             and tuple(drawn) == tuple(engine.egg_drawn) and tuple(deck) == tuple(engine.remaining_abilities))
    for rid, actor in engine.racers.items():
        _require(actor.inherited_ability == inherited[rid])
        if actor.ability in _CARRIERS and actor.pre_race_resolved:
            _require(leaves[rid] != Ability.EGG
                     and (leaves[rid] != Ability.TWIN or not previous or rid in stopped))
    return leaves, (tuple(choices), tuple(inherit_events))


def static_inheritance_evidence(engine, actor, wanted):
    """Immutable acquisition/deck binding, empty for native, or None on refusal.

    This authenticates a static source, including an actual acquisition of
    Copy Cat itself. It never follows Copy Cat's borrowed ability to another
    wanted power. Callers still enforce their own geometry and active status.
    """
    try:
        wanted = Ability(wanted)
        initial = _roster(engine)
        rid = actor.racer_id
        _require(engine.racers.get(rid) is actor and actor.pre_race_resolved is True
                 and initial[rid]["inherited_ability"] is None)
        if actor.ability == wanted and actor.inherited_ability is None:
            _require(not any(event.kind == "inherit" and event.racer_id == rid for event in engine.events))
            if wanted not in _CARRIERS:
                _require(not any(record.request.window == DecisionWindow.BEFORE_RACE
                                 and record.request.player_id == rid for record in engine.decisions))
            if wanted != Ability.COPY_CAT:
                _require(actor.copied_target is None and actor.copied_ability is None)
            return ()
        _require(actor.ability in _CARRIERS and isinstance(actor.inherited_ability, Ability))
        leaves, (choices, inheritance) = _acquisition(engine, initial)
        _require(leaves[rid] == wanted and actor.inherited_ability == wanted)
        if wanted != Ability.COPY_CAT:
            _require(actor.copied_target is None and actor.copied_ability is None)
        binding = {"actor": rid, "original": actor.ability.value, "wanted": wanted.value,
                   "initial_racers": engine.initial_racers, "initial_scores": engine.initial_score_ledger,
                   "initial_remaining_abilities": engine.initial_remaining_abilities,
                   "remaining_abilities": engine.remaining_abilities, "egg_drawn": engine.egg_drawn,
                   "previous_winners": engine.previous_winners,
                   "static_entities": [(key, value.name, value.ability, value.inherited_ability, value.pre_race_resolved)
                                       for key, value in sorted(engine.racers.items())],
                   "choices": [asdict(record) for record in choices],
                   "inherit_events": [asdict(event) for event in inheritance]}
        return (json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration):
        return None


def authenticated_static_power(engine, actor, wanted):
    """True only for its original power or a verified static Egg/Twin chain."""
    return static_inheritance_evidence(engine, actor, wanted) is not None


def resolved_inert_twin_evidence(engine, actor):
    """Authenticate an original Twin that actually kept its original power.

    Unlike the native-power fast path, this checks the complete preparation
    history. A nonempty winner list requires its one real BEFORE_RACE KEEP;
    an empty list requires no window, as in the engine's automatic case.
    Egg->Twin and any acquired or dynamically copied power remain outside
    this narrow family. Callers still verify status, geometry and controls.
    """
    try:
        initial = _roster(engine)
        rid = actor.racer_id
        _require(engine._preparing is False and engine.racers.get(rid) is actor and actor.ability == Ability.TWIN
                 and actor.pre_race_resolved is True and initial[rid]["pre_race_resolved"] is False
                 and initial[rid]["inherited_ability"] is None and actor.inherited_ability is None
                 and initial[rid]["copied_target"] is None and initial[rid]["copied_ability"] is None
                 and actor.copied_target is None and actor.copied_ability is None)
        leaves, (choices, inheritance) = _acquisition(engine, initial)
        own = tuple(record for record in choices if record.request.player_id == rid)
        _require(leaves[rid] == Ability.TWIN
                 and not any(event.racer_id == rid for event in inheritance))
        if engine.previous_winners:
            _require(len(own) == 1 and own[0].action == Action(ActionKind.KEEP))
            resolution = "before-race-keep"
        else:
            _require(not own)
            resolution = "no-previous-winners-no-window"
        binding = {"actor": rid, "original": Ability.TWIN.value, "resolution": resolution,
                   "initial_racers": engine.initial_racers, "initial_scores": engine.initial_score_ledger,
                   "initial_remaining_abilities": engine.initial_remaining_abilities,
                   "remaining_abilities": engine.remaining_abilities, "egg_drawn": engine.egg_drawn,
                   "previous_winners": engine.previous_winners,
                   "static_entities": [(key, value.name, value.ability, value.inherited_ability,
                                        value.pre_race_resolved) for key, value in sorted(engine.racers.items())],
                   "choices": [asdict(record) for record in choices],
                   "inherit_events": [asdict(event) for event in inheritance]}
        return (json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration):
        return None
