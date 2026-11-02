"""Two separate proofs for native Fish/Scoocher/Baby on Wild Arrow11.

The synchronous proof stops only a repeated Arrow step and keeps every FIFO
qualification. The FIFO proof waits for all original jobs, including real
silent Baba Trips, to return before stopping only their fresh descendants.
Exact domains: original I5/current S after its real main6; original Hypnotist15;
or original Baba11 with F/S tripped at
the FIFO cut. Local Arrow cuts allow either F/S Trip flag: Trips are deferred.
"""
from collections import deque
from dataclasses import asdict, dataclass, field, fields
import json
import hashlib

from .decisions import Action, ActionKind, DecisionRecord, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .fish_baby_arrow11_cycle import _typed_request, _Reject as _RequestReject
from .romantic_baby_shared12_cycle import _ENGINE_FIELDS, _EVENT_FIELDS, _RACER_FIELDS, _event, _known_pending, _ledger_valid
from .racer import Ability, Racer
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import WILD_WILDS

REASON = "native_fish_scoocher_baby_arrow11_fifo_cycle_proved"
ARROW_REASON = "native_fish_scoocher_baby_arrow11_choice_cycle_proved"
CACHE_TAG = "native-fish-scoocher-arrow11-choice-v1"
_ALLOWED = _ENGINE_FIELDS | {"_arrow_visits", "_arrow_queue_visits"}
_ARROW_PROOF = "KEEP returns the single state; FOLLOW reaches the forced shared state; all FIFO qualifications remain owed."


class _Reject(Exception):
    pass


def _need(value):
    if not value:
        raise _Reject


def _actors(engine):
    _need(type(engine.racers) is dict and len(engine.racers) == 4)
    by = {actor.ability: actor for actor in engine.racers.values()}
    base = {Ability.SUCKERFISH, Ability.SCOOCHER, Ability.HUGE_BABY}
    _need(set(by) in (base | {Ability.HYPNOTIST}, base | {Ability.BABA_YAGA}, base | {Ability.INCHWORM}))
    observer = by.get(Ability.HYPNOTIST, by.get(Ability.BABA_YAGA, by.get(Ability.INCHWORM)))
    return by[Ability.SUCKERFISH], observer, by[Ability.HUGE_BABY], by[Ability.SCOOCHER]


def _digest(events, choices):
    value = [[asdict(event) for event in events], [asdict(record) for record in choices]]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _parent(engine, event):
    index = event.parent_sequence
    _need(type(index) is int and 1 <= index < event.sequence <= len(engine.events))
    parent = engine.events[index - 1]
    _need(type(parent) is RaceEvent and parent.sequence == index)
    return parent


def _batch_before(engine, sequence):
    return max((event.arrival_batch_id for event in engine.events[:sequence - 1]
                if type(event.arrival_batch_id) is int), default=0)


def _choice_source(engine, event, parent, fish, target, *, hypnotist=False):
    decision = engine.events[event.sequence - 2]
    if not hypnotist and parent.kind == "ability_effect":
        decision = engine.events[parent.sequence - 2]
        expected_parent = parent.parent_sequence
    else:
        expected_parent = None if hypnotist else parent.sequence
    kind = ActionKind.HYPNOTIZE if hypnotist else ActionKind.SUCKERFISH_FOLLOW
    _need(_event(decision, "decision", fish, source_id=fish, target_id=target,
                 action_kind=kind.value, decision_id=decision.decision_id, parent_sequence=expected_parent)
          and type(decision.decision_id) is int and 1 <= decision.decision_id <= len(engine.decisions))
    record = engine.decisions[decision.decision_id - 1]
    try:
        _typed_request(record)
    except _RequestReject:
        raise _Reject
    action = Action(kind, fish, target)
    _need(record.event_sequence == decision.sequence and record.action == action and action in record.request.legal_actions
          and record.request.player_id == fish
          and record.request.window == (DecisionWindow.TURN_START if hypnotist else DecisionWindow.MOVE_START))
    if hypnotist:
        _need(event.parent_sequence == decision.sequence)
    if not hypnotist:
        _need(record.request.legal_actions == (Action(ActionKind.KEEP), action)
              and record.request.observation.current_move_value == 1)


def _auth_cause(engine, event, actors, seen=(), checked=None):
    """Walk literal earlier native causes; never accept an arbitrary parent ID."""
    _need(event.sequence not in seen and len(seen) < 250)
    checked = set() if checked is None else checked
    if event.sequence in checked:
        return
    seen = (*seen, event.sequence)
    f, o, b, s = (actor.racer_id for actor in actors)
    if event.kind == "ability_effect" and event.ability == "Scoocher":
        _need(_event(event, "ability_effect", s, source_id=s, ability="Scoocher", amount=1, parent_sequence=event.parent_sequence))
        _auth_notification(engine, _parent(engine, event), actors, seen, checked)
    elif event.kind == "ability_effect" and event.ability == "Suckerfish":
        _need(_event(event, "ability_effect", f, source_id=f, target_id=s, ability="Suckerfish", amount=1, parent_sequence=event.parent_sequence))
        parent = _parent(engine, event)
        _need((parent.kind == "ability_effect" and parent.ability == "Scoocher")
              or parent.kind == "stop" and parent.racer_id == s and parent.after in {11, 16})
        _choice_source(engine, event, parent, f, s)
        _auth_cause(engine, parent, actors, seen, checked)
    elif event.kind == "ability_effect" and event.ability == "Huge Baby":
        _need(type(event.arrival_batch_id) is int and event.arrival_batch_id > 0
              and _event(event, "ability_effect", b, source_id=b, ability="Huge Baby",
                         parent_sequence=event.parent_sequence, arrival_batch_id=event.arrival_batch_id))
        parent = _parent(engine, event)
        prior = engine.events[:event.sequence - 1]
        arrivals = [item for item in prior if item.kind == "move" and item.arrival_batch_id == event.arrival_batch_id]
        _need(event.arrival_batch_id == max((item.arrival_batch_id for item in prior if type(item.arrival_batch_id) is int), default=0)
              and not any(item.kind == "ability_effect" and item.ability == "Huge Baby"
                          and item.arrival_batch_id == event.arrival_batch_id for item in prior)
              and any(item.racer_id in {f, s, b} and item.after == 12 and item.parent_sequence == parent.sequence for item in arrivals))
        _need(parent.kind == "main_move_resolved"
              and (parent.racer_id == b or actors[1].ability == Ability.INCHWORM and parent.racer_id == s)
              or parent.kind == "ability_effect" and parent.ability == "Scoocher"
              or parent.kind == "stop" and parent.racer_id in {f, s, b} and parent.after in {11, 16})
        for arrival in arrivals:
            _need(arrival.racer_id in ({f, s, b, o} if actors[1].ability == Ability.BABA_YAGA else {f, s, b}) and arrival.after == 12)
            _auth_cause(engine, arrival, actors, seen, checked)
        _auth_cause(engine, parent, actors, seen, checked)
    elif event.kind == "ability_effect" and event.ability == "Hypnotist":
        _need(actors[1].ability == Ability.HYPNOTIST and event.racer_id == o and event.source_id == o and event.target_id in {f, s, b}
              and _event(event, "ability_effect", o, source_id=o, target_id=event.target_id,
                         ability="Hypnotist", parent_sequence=event.parent_sequence))
        _choice_source(engine, event, event, o, event.target_id, hypnotist=True)
    elif event.kind == "main_move_resolved":
        locked = _parent(engine, event)
        roll = _parent(engine, locked)
        actor = s if actors[1].ability == Ability.INCHWORM else b
        _need(event.racer_id == actor and type(event.after) is int and 1 <= event.after <= 6
              and (actors[1].ability != Ability.INCHWORM or event.after == 6)
              and _event(event, "main_move_resolved", actor, before=event.after, after=event.after, parent_sequence=locked.sequence)
              and _event(locked, "roll_locked", actor, amount=event.after, parent_sequence=roll.sequence)
              and _event(roll, "roll", actor, amount=event.after))
    elif event.kind == "stop":
        _need(event.racer_id in {f, s, b, o} and type(event.before) is int and type(event.after) is int
              and _event(event, "stop", event.racer_id, before=event.before, after=event.after,
                         parent_sequence=event.parent_sequence, arrival_batch_id=event.arrival_batch_id))
        parent = _parent(engine, event)
        if parent.kind == "move":
            _need((event.racer_id, event.before, event.after, event.arrival_batch_id)
                  == (parent.racer_id, parent.before, parent.after, parent.arrival_batch_id))
        elif parent.kind == "warp":
            _need((event.racer_id, event.before, event.after, event.arrival_batch_id)
                  == (parent.racer_id, parent.before, parent.after, _batch_before(engine, parent.sequence) + 1))
        else:
            _need(parent.kind == "ability_effect" and parent.ability == "Huge Baby"
                  and event.before == 12 and event.after == 11
                  and type(parent.arrival_batch_id) is int and event.arrival_batch_id == parent.arrival_batch_id + 1
                  and event.racer_id != b and _history_positions(engine, parent.sequence - 1)[event.racer_id] == 12)
        _auth_cause(engine, parent, actors, seen, checked)
    elif event.kind in {"move", "warp"}:
        _need(event.racer_id in ({f, s, b, o} if actors[1].ability == Ability.BABA_YAGA else {f, s, b})
              and (event.racer_id != o or event.kind == "move" and event.movement_kind == "track")
              and type(event.before) is int and type(event.after) is int)
        parent = _parent(engine, event)
        if event.kind == "move":
            _need(event.movement_kind in {"main", "track", "ability"} and type(event.amount) is int
                  and event.after - event.before == event.amount
                  and type(event.arrival_batch_id) is int and event.arrival_batch_id > 0)
            _need(event.before == _history_positions(engine, event.sequence - 1)[event.racer_id])
            values = {"before": event.before, "after": event.after, "amount": event.amount,
                      "movement_kind": event.movement_kind, "parent_sequence": parent.sequence,
                      "arrival_batch_id": event.arrival_batch_id}
            if event.movement_kind == "track":
                _need(parent.kind == "stop" and event.before == parent.after
                      and dict(WILD_WILDS.arrows).get(parent.after) == event.amount
                      and type(parent.arrival_batch_id) is int and event.arrival_batch_id == parent.arrival_batch_id + 1)
            elif event.movement_kind == "ability":
                ability = "Scoocher" if event.racer_id == s else "Suckerfish" if event.racer_id == f else None
                _need(parent.kind == "ability_effect" and parent.ability == ability
                      and parent.racer_id == parent.source_id == event.racer_id and event.amount == 1
                      and event.arrival_batch_id == _batch_before(engine, parent.sequence) + 1)
                values.update(source_id=event.racer_id, ability=ability)
            else:
                actor = s if actors[1].ability == Ability.INCHWORM else b
                _need(parent.kind == "main_move_resolved" and parent.racer_id == event.racer_id == actor
                      and (actors[1].ability != Ability.INCHWORM or (event.before, event.after, event.amount) == (6, 12, 6))
                      and event.amount == parent.after and event.arrival_batch_id == _batch_before(engine, parent.sequence) + 1)
            _need(_event(event, "move", event.racer_id, **values))
        else:
            _need(parent.kind == "ability_effect" and parent.ability == "Hypnotist"
                  and parent.source_id == o and parent.target_id == event.racer_id
                  and event.before == _history_positions(engine, event.sequence - 1)[event.racer_id]
                  and event.after == _history_positions(engine, event.sequence - 1)[o]
                  and event.sequence == parent.sequence + 2
                  and _event(event, "warp", event.racer_id, source_id=o, ability="Hypnotist",
                             before=event.before, after=event.after, parent_sequence=parent.sequence))
        _auth_cause(engine, parent, actors, seen, checked)
    elif event.kind == "trip":
        _need(actors[1].ability == Ability.BABA_YAGA and event.racer_id in {f, s}
              and _event(event, "trip", event.racer_id, source_id=o, ability="Baba Yaga", parent_sequence=event.parent_sequence))
        queued = _parent(engine, event)
        _auth_baba_qualification(engine, queued, actors)
        _need(queued.target_id == event.racer_id)
        flags = {row["racer_id"]: row["tripped"] for row in engine.initial_racers}
        _need(all(type(value) is bool for value in flags.values()))
        for prior in engine.events[:event.sequence - 1]:
            if prior.kind == "trip":
                flags[prior.racer_id] = True
            elif prior.kind == "recover":
                flags[prior.racer_id] = False
        _need(flags[event.racer_id] is False)
        _source_fifo(engine, actors, event.sequence)
        _auth_cause(engine, _parent(engine, queued), actors, seen, checked)
    else:
        raise _Reject
    checked.add(event.sequence)


def _auth_notification(engine, queued, actors, seen=(), checked=None):
    f, o, b, s = (actor.racer_id for actor in actors)
    _need(_event(queued, "effect_queued", s, source_id=queued.source_id, ability="Scoocher", parent_sequence=queued.parent_sequence))
    trigger = _parent(engine, queued)
    _need(queued.sequence == trigger.sequence + 1)
    allowed = {f: "Suckerfish", b: "Huge Baby"}
    if actors[1].ability != Ability.INCHWORM:
        allowed[o] = "Hypnotist" if actors[1].ability == Ability.HYPNOTIST else "Baba Yaga"
    _need(queued.source_id in allowed and trigger.ability == allowed[queued.source_id]
          and (trigger.kind == "trip" if trigger.ability == "Baba Yaga" else trigger.kind == "ability_effect"))
    _auth_cause(engine, trigger, actors, seen, checked)


def _history_positions(engine, boundary):
    positions = {row["racer_id"]: row["position"] for row in engine.initial_racers}
    for event in engine.events[:boundary]:
        if event.kind in {"move", "warp"}:
            _need(type(event.before) is int and type(event.after) is int and event.before == positions[event.racer_id])
            positions[event.racer_id] = event.after
        elif event.kind == "stop":
            _need(type(event.before) is int and type(event.after) is int and positions[event.racer_id] in {event.before, event.after})
            positions[event.racer_id] = event.after
        elif event.kind in {"inherit", "copy_changed", "finish", "eliminate"}:
            raise _Reject
    return positions


def _auth_baba_qualification(engine, queued, actors):
    f, o, _, s = (actor.racer_id for actor in actors)
    _need(actors[1].ability == Ability.BABA_YAGA and queued.target_id in {f, s}
          and _event(queued, "effect_queued", o, source_id=o, target_id=queued.target_id,
                     ability="Baba Yaga", parent_sequence=queued.parent_sequence))
    stop = _parent(engine, queued)
    _need(stop.racer_id in {queued.target_id, o} and type(stop.before) is int
          and _event(stop, "stop", stop.racer_id, before=stop.before, after=11,
                     parent_sequence=stop.parent_sequence, arrival_batch_id=stop.arrival_batch_id))
    positions = _history_positions(engine, stop.sequence)
    members = {event.racer_id for event in engine.events[:queued.sequence]
               if event.kind == "stop" and event.arrival_batch_id == stop.arrival_batch_id}
    _need(positions[o] == positions[queued.target_id] == 11 and not {o, queued.target_id} <= members)


def _source_fifo(engine, actors, boundary=None):
    """Rebuild only the current native turn's real queue, including inert Trips.

    This is not replay: it reads recorded births and real service events, never
    selects an action, rolls, moves a racer or consumes an engine task.
    """
    f, observer, b, s = actors
    end = len(engine.events) if boundary is None else boundary
    events = engine.events[:end]
    baba = observer.ability == Ability.BABA_YAGA
    inchworm = observer.ability == Ability.INCHWORM
    if baba or inchworm:
        roots = [event for event in events if event.kind == "main_move_resolved"]
        _need(roots and roots[-1].racer_id == (s.racer_id if inchworm else b.racer_id))
        start = roots[-1].sequence - 1
    else:
        roots = [event for event in events if event.kind == "ability_effect" and event.ability == "Hypnotist"
                 and event.racer_id == observer.racer_id]
        _need(roots)
        start = roots[-1].sequence - 1
    flags = {row["racer_id"]: row["tripped"] for row in engine.initial_racers}
    for event in engine.events[:start]:
        if event.kind == "trip":
            flags[event.racer_id] = True
        elif event.kind == "recover":
            flags[event.racer_id] = False
    fifo, born = deque(), {}

    def head(identity):
        while fifo and fifo[0].queued_sequence != identity and fifo[0].kind == "trip" and flags[fifo[0].target_id] is True:
            fifo.popleft()  # Native already-tripped call is genuinely silent.
        _need(fifo and fifo[0].queued_sequence == identity)
        return fifo.popleft()

    for event in engine.events[start:end]:
        if event.kind == "effect_queued":
            if event.ability == "Scoocher":
                _need(event.racer_id == s.racer_id
                      and event.source_id in ({f.racer_id, b.racer_id} if inchworm else {f.racer_id, observer.racer_id, b.racer_id})
                      and _event(event, "effect_queued", s.racer_id, source_id=event.source_id,
                                 ability="Scoocher", parent_sequence=event.parent_sequence))
                pending = PendingEffect(s.racer_id, event.source_id, event.parent_sequence, event.sequence)
            else:
                _need(baba and event.target_id in {f.racer_id, s.racer_id}
                      and _event(event, "effect_queued", observer.racer_id, source_id=observer.racer_id,
                                 target_id=event.target_id, ability="Baba Yaga", parent_sequence=event.parent_sequence))
                pending = PendingEffect(observer.racer_id, observer.racer_id, event.parent_sequence, event.sequence,
                                        "trip", 1, "Baba Yaga", event.target_id)
            fifo.append(pending)
            born[event.sequence] = pending
        elif event.kind == "ability_effect" and event.ability == "Scoocher":
            pending = head(event.parent_sequence)
            _need(pending.kind == "move" and pending.racer_id == event.racer_id == s.racer_id)
        elif event.kind == "trip":
            if event.ability == "Baba Yaga":
                pending = head(event.parent_sequence)
                _need(pending.kind == "trip" and pending.target_id == event.racer_id and flags[event.racer_id] is False)
            flags[event.racer_id] = True
        elif event.kind == "recover":
            flags[event.racer_id] = False
    if boundary is None:
        current = tuple(engine.pending_effects)
        while tuple(fifo) != current and fifo and fifo[0].kind == "trip" and flags[fifo[0].target_id] is True:
            fifo.popleft()
        _need(tuple(fifo) == current)
    return tuple(fifo)


def _pending(engine, actors, pending=None, *, unconsumed=True):
    f, o, b, s = (actor.racer_id for actor in actors)
    baba = actors[1].ability == Ability.BABA_YAGA
    consumed = {event.parent_sequence for event in engine.events if event.kind in {"ability_effect", "trip", "effect_cancelled"}}
    ids = []
    pending = engine.pending_effects if pending is None else pending
    for job in pending:
        _need(_known_pending(job) and 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events))
        queued, trigger = (engine.events[index - 1] for index in (job.queued_sequence, job.trigger_sequence))
        if job.kind == "move":
            _need((job.racer_id, job.kind, job.steps, job.ability, job.target_id) == (s, "move", 1, "Scoocher", None)
                  and job.source_id in ({f, b, o} if baba else {f, b}))
            _need(_event(queued, "effect_queued", s, source_id=job.source_id, ability="Scoocher", parent_sequence=trigger.sequence))
            _auth_notification(engine, queued, actors)
            if job.source_id == f:
                _need(_event(trigger, "ability_effect", f, source_id=f, target_id=s, ability="Suckerfish", amount=1,
                             parent_sequence=trigger.parent_sequence))
            elif job.source_id == b:
                _need(type(trigger.arrival_batch_id) is int and trigger.arrival_batch_id > 0
                      and _event(trigger, "ability_effect", b, source_id=b, ability="Huge Baby",
                                 parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id))
            else:
                _need(trigger.racer_id in {f, s} and _event(trigger, "trip", trigger.racer_id, source_id=o,
                      ability="Baba Yaga", parent_sequence=trigger.parent_sequence))
        else:
            _need(baba and (job.racer_id, job.source_id, job.kind, job.steps, job.ability) == (o, o, "trip", 1, "Baba Yaga")
                  and job.target_id in {f, s}
                  and _event(queued, "effect_queued", o, source_id=o, target_id=job.target_id,
                             ability="Baba Yaga", parent_sequence=trigger.sequence)
                  and trigger.racer_id in {job.target_id, o}
                  and _event(trigger, "stop", trigger.racer_id, before=trigger.before, after=11,
                             parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id))
            _auth_baba_qualification(engine, queued, actors)
            _auth_cause(engine, trigger, actors)
        _need(not unconsumed or job.queued_sequence not in consumed)
        ids.append(job.queued_sequence)
    _need(ids == sorted(set(ids)) and any(job.kind == "move" for job in pending))


def _control(engine, actors, *, arrow=False):
    f, observer, baby, s = actors
    _need(not set(vars(engine)) - _ALLOWED and engine.track == WILD_WILDS
          and engine.termination_reason is None and type(engine.pending_effects) is deque
          and type(engine.finish_order) is list and not engine.finish_order
          and type(engine._copy_locks) is set and not engine._copy_locks
          and engine._preparing is False and engine._turn_in_progress is True and engine._refreshing_copies is False
          and engine.next_turn_racer_id is None and engine._genius_prediction is None
          and getattr(engine, "_inert_duel_romantic", None) is None
          and type(getattr(engine, "_duel_offered_epochs", {})) is dict and not getattr(engine, "_duel_offered_epochs", {})
          and type(engine.score_ledger) is ScoreLedger and type(engine.policies) is dict and type(engine.policy_metadata) is dict)
    _need(all(type(getattr(engine, name)) is list for name in ("turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn"))
          and len(engine.turn_order) == 4 and set(engine.turn_order) == set(engine.racers)
          and all(type(rid) is int for rid in engine.turn_order)
          and type(engine.initial_remaining_abilities) is tuple and type(engine.previous_winners) is tuple
          and type(engine.turn_index) is int and 0 <= engine.turn_index < 4)
    _need(all(type(getattr(engine, name, None)) is int and getattr(engine, name) > 0
              for name in ("_resolution_depth", "_physical_resolution_depth", "_arrival_counter", "race_index", "max_events"))
          and type(getattr(engine, "_movement_depth", 0)) is int
          and (getattr(engine, "_movement_depth", 0) > 0 if arrow else getattr(engine, "_movement_depth", 0) == 0))
    _need((engine.seed is None or type(engine.seed) is int) and type(engine._score_start) is int and engine._score_start >= 0)
    if arrow:
        _need(type(engine._arrow_visits) is set and type(engine._arrow_queue_visits) is dict)
    else:
        _need(not hasattr(engine, "_arrow_visits") and not hasattr(engine, "_arrow_queue_visits"))
    for actor in actors:
        _need(type(actor) is Racer and set(vars(actor)) == _RACER_FIELDS
              and type(actor.racer_id) is int and engine.racers.get(actor.racer_id) is actor
              and type(actor.ability) is Ability and type(actor.status) is RacerStatus and actor.status == RacerStatus.ACTIVE
              and actor.pre_race_resolved is True and type(actor.position) is int and type(actor.tripped) is bool
              and type(actor.turns_taken) is int and actor.turns_taken >= 0
              and actor.inherited_ability is None and actor.copied_target is None and actor.copied_ability is None
              and actor.prediction is None and actor.finish_place is None and type(actor.finish_places) is tuple and not actor.finish_places)
    _need((f.position, s.position, baby.position) == (11, 11, 12))
    baba = observer.ability == Ability.BABA_YAGA
    inchworm = observer.ability == Ability.INCHWORM
    _need(observer.position == (5 if inchworm else 11 if baba else 15))
    _need(engine.turn_order[engine.turn_index] == (s.racer_id if inchworm else baby.racer_id if baba else observer.racer_id))
    _need(not inchworm or f.tripped is s.tripped is observer.tripped is False)
    _need(arrow or not baba or f.tripped is s.tripped is True)
    initial = {row["racer_id"]: row for row in engine.initial_racers}
    _need(len(initial) == 4 and set(initial) == set(engine.racers)
          and all(initial[actor.racer_id]["ability"] == actor.ability.value and initial[actor.racer_id]["name"] == actor.name for actor in actors))
    _need(all(type(event) is RaceEvent and set(vars(event)) == _EVENT_FIELDS and type(event.sequence) is int
              and event.sequence == index + 1 for index, event in enumerate(engine.events))
          and all(type(record) is DecisionRecord for record in engine.decisions)
          and [event.decision_id for event in engine.events if event.kind == "decision"] == list(range(1, len(engine.decisions) + 1)))
    _pending(engine, actors)
    _source_fifo(engine, actors)
    _need(_ledger_valid(engine, len(engine.events)))
    _need(not hasattr(engine, "max_action_events") or type(engine.max_action_events) is int and engine.max_action_events > 0)


def _state(engine, actors, *, arrow=False):
    _control(engine, actors, arrow=arrow)
    fixed = {"roster": [{**asdict(actor), "status": actor.status.value, "ability": actor.ability.value}
                         for _, actor in sorted(engine.racers.items())],
             "turns": [asdict(turn) for turn in engine.turns], "turn_order": engine.turn_order,
             "turn_index": engine.turn_index, "race_index": engine.race_index,
             "scores": asdict(engine.score_ledger.snapshot()), "initial_scores": engine.initial_score_ledger,
             "initial_racers": engine.initial_racers, "remaining": engine.remaining_abilities,
             "initial_remaining": engine.initial_remaining_abilities, "egg_drawn": engine.egg_drawn,
             "previous_winners": engine.previous_winners, "policy_metadata": engine.policy_metadata,
             "policy_seats": sorted(engine.policies), "score_start": engine._score_start,
             "fingerprint": engine.initial_code_fingerprint, "seed": engine.seed, "max_events": engine.max_events,
             "max_action_events": getattr(engine, "max_action_events", None), "known_fields": sorted(vars(engine))}
    if not arrow:
        fixed["depths"] = [engine._resolution_depth, engine._physical_resolution_depth]
    return json.dumps(fixed, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(slots=True)
class Snapshot:
    state: str
    event_start: int
    decision_start: int
    original_pending: tuple
    score_before: object
    arrival_counter: int
    event_prefix: tuple
    decision_prefix: tuple
    prefix_digest: str
    jobs: list = field(default_factory=list)
    in_flight: object = None
    invalid: bool = False


@dataclass(frozen=True, slots=True)
class ArrowSnapshot:
    state: str
    event_start: int
    decision_start: int
    original_pending: tuple
    score_before: object
    arrival_counter: int
    event_prefix: tuple
    decision_prefix: tuple
    depths: tuple
    stop_sequence: int
    original_visits: frozenset
    original_memos: tuple
    prefix_digest: str


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot, self.actors = engine, snapshot, actors
        self.f, self.o, self.b, self.s = (actor.racer_id for actor in actors)
        self.baba = actors[1].ability == Ability.BABA_YAGA
        self.positions = {actor.racer_id: actor.position for actor in actors}
        self.events, self.choices = engine.events[snapshot.event_start:], engine.decisions[snapshot.decision_start:]
        self.offset = self.choice_index = 0
        self.batch = snapshot.arrival_counter
        self.children, self.choice_ids = [], []

    def take(self, kind, actor, **values):
        _need(self.offset < len(self.events))
        event = self.events[self.offset]
        _need(_event(event, kind, actor, **values))
        self.offset += 1
        return event

    def choice(self, parent):
        _need(self.choice_index < len(self.choices))
        record = self.choices[self.choice_index]
        try:
            _typed_request(record)
        except _RequestReject:
            raise _Reject
        legal = (Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, self.f, self.s))
        public = Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            WILD_WILDS.name, 30, 15, tuple(sorted(WILD_WILDS.trip_positions)),
            tuple(PublicRacer(rid, self.engine.racers[rid].name, self.positions[rid], "active",
                             self.engine.racers[rid].ability.value, self.engine.racers[rid].tripped, None, None)
                  for rid in self.engine.turn_order), (), None, 1, score_totals=self.snapshot.score_before.totals)
        request = DecisionRequest(self.snapshot.decision_start + self.choice_index + 1, self.f,
                                  DecisionWindow.MOVE_START, public, legal)
        _need(record.request == request and record.action in legal)
        self.choice_index += 1
        action = record.action
        event = self.take("decision", self.f, source_id=action.source_id, target_id=action.target_id,
                          parent_sequence=parent, action_kind=action.kind.value, decision_id=request.decision_id)
        _need(record.event_sequence == event.sequence)
        self.choice_ids.append(event.sequence)
        return action.kind == ActionKind.SUCKERFISH_FOLLOW

    def queued(self, source, trigger):
        event = self.take("effect_queued", self.s, source_id=source, ability="Scoocher", parent_sequence=trigger.sequence)
        self.children.append(PendingEffect(self.s, source, trigger.sequence, event.sequence))

    def fish(self, parent):
        effect = self.take("ability_effect", self.f, source_id=self.f, target_id=self.s, ability="Suckerfish", amount=1, parent_sequence=parent)
        self.queued(self.f, effect)
        return effect

    def arrival(self, parent, *, follow, source="track", power=None):
        """Literal 11->12 direct batch, Baby, 12->11 batch, then Baba qualifications."""
        self.batch += 1
        first = self.take("move", self.s, before=11, after=12, amount=1, movement_kind=source,
                          source_id=self.s if source == "ability" else None,
                          ability="Scoocher" if source == "ability" else None,
                          parent_sequence=parent, arrival_batch_id=self.batch)
        if follow:
            self.take("move", self.f, before=11, after=12, amount=1, movement_kind="ability",
                      source_id=self.f, ability="Suckerfish", parent_sequence=power.sequence, arrival_batch_id=self.batch)
        return self.baby(parent, (self.s, self.f) if follow else (self.s,))

    def paired_arrow(self, stop):
        self.batch += 1
        order = [rid for rid in self.engine._clockwise_order() if rid in {self.f, self.s}]
        for rid in order:
            self.take("move", rid, before=11, after=12, amount=1, movement_kind="track",
                      parent_sequence=stop.sequence, arrival_batch_id=self.batch)
        returned = self.baby(stop.sequence, tuple(order))
        self.take("loop_stopped", order[0], reason="forced_arrow_cycle_with_deferred_queue_proved",
                  parent_sequence=returned.sequence)

    def baby(self, parent, members):
        power = self.take("ability_effect", self.b, source_id=self.b, ability="Huge Baby",
                          parent_sequence=parent, arrival_batch_id=self.batch)
        self.queued(self.b, power)
        self.batch += 1
        order = [rid for rid in self.engine._clockwise_order() if rid in members]
        stops = [self.take("stop", rid, before=12, after=11, parent_sequence=power.sequence, arrival_batch_id=self.batch) for rid in order]
        if self.baba:
            for stop in stops:
                queued = self.take("effect_queued", self.o, source_id=self.o, target_id=stop.racer_id,
                                   ability="Baba Yaga", parent_sequence=stop.sequence)
                self.children.append(PendingEffect(self.o, self.o, stop.sequence, queued.sequence, "trip", 1, "Baba Yaga", stop.racer_id))
        return stops[0]

    def arrow_keep(self, stop):
        _need(not self.choice(stop.sequence))
        returned = self.arrival(stop.sequence, follow=False)
        return returned

    def arrow_group_from_single(self, stop):
        capture_count = self.snapshot.event_start + self.offset
        original = self.queue_before + tuple(self.children)
        choice_begin = len(self.choice_ids)
        child_begin = len(self.children)
        follow = self.choice(stop.sequence)
        fish = self.fish(stop.sequence) if follow else None
        returned = self.arrival(stop.sequence, follow=follow, power=fish)
        if follow:
            self.paired_arrow(returned)
        else:
            loop = self.take("loop_stopped", self.s, reason=ARROW_REASON, parent_sequence=returned.sequence,
                             operation=self.events[self.offset].operation if self.offset < len(self.events) else None)
            packet = json.loads(loop.operation)
            expected = _arrow_packet(capture_count, returned.sequence, original,
                                     self.children[child_begin:], self.choice_ids[choice_begin:])
            _need(_canonical(packet) == _canonical(expected))

    def job(self, pending):
        self.children = []
        if pending.kind == "trip":
            _need(self.baba and self.engine.racers[pending.target_id].tripped is True)
            return ()
        follow = self.choice(pending.queued_sequence)
        power = self.take("ability_effect", self.s, source_id=self.s, ability="Scoocher", amount=1, parent_sequence=pending.queued_sequence)
        fish = self.fish(power.sequence) if follow else None
        returned = self.arrival(power.sequence, follow=follow, source="ability", power=fish)
        if follow:
            self.paired_arrow(returned)
        else:
            self.arrow_group_from_single(returned)
        return tuple(self.children)


def _arrow_packet(event_start, stop_sequence, original, children, choices):
    return {"capture_event_count": event_start, "current_arrow_stop_sequence": stop_sequence,
            "original_pending_ids": [job.queued_sequence for job in original],
            "new_deferred_pending_ids": [job.queued_sequence for job in children],
            "fish_decision_event_ids": choices, "actual_choice": "keep",
            "original_and_new_qualifications_preserved": True, "only_repeated_arrow_step_stopped": True,
            "scores_unchanged": True, "all_choice_proof": _ARROW_PROOF}


def arrow_key(engine, group, position, stop):
    try:
        actors = _actors(engine)
        _need(position == 11 and group == [(actors[3].racer_id, 1)]
              and type(stop) is RaceEvent and type(stop.sequence) is int)
        # Baba qualifications can follow the stop before this hook is called.
        _need(_event(stop, "stop", actors[3].racer_id, before=12, after=11,
                     parent_sequence=stop.parent_sequence, arrival_batch_id=stop.arrival_batch_id)
              and engine.events[stop.sequence - 1] == stop)
        return _state(engine, actors, arrow=True)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, _Reject):
        return None


def arrow_capture(engine, state, stop):
    _need(arrow_key(engine, [(stop.racer_id, 1)], 11, stop) == state)
    return ArrowSnapshot(state, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
        engine.score_ledger.snapshot(), engine._arrival_counter, tuple(engine.events), tuple(engine.decisions),
        (engine._resolution_depth, engine._physical_resolution_depth, engine._movement_depth), stop.sequence,
        frozenset(engine._arrow_visits), tuple(engine._arrow_queue_visits.items()), _digest(engine.events, engine.decisions))


def arrow_completed(engine, snapshot, stop):
    try:
        _need(type(snapshot) is ArrowSnapshot and type(snapshot.original_pending) is tuple
              and all(_known_pending(job) for job in snapshot.original_pending))
        actors = _actors(engine)
        _need(arrow_key(engine, [(actors[3].racer_id, 1)], 11, stop) == snapshot.state
              and tuple(engine.events[:snapshot.event_start]) == snapshot.event_prefix
              and tuple(engine.decisions[:snapshot.decision_start]) == snapshot.decision_prefix
              and engine.score_ledger.snapshot() == snapshot.score_before
              and (engine._resolution_depth, engine._physical_resolution_depth, engine._movement_depth)
                  == (snapshot.depths[0] + 3, snapshot.depths[1] + 3, snapshot.depths[2] + 1)
              and snapshot.original_visits <= engine._arrow_visits
              and all(engine._arrow_queue_visits.get(key) == value for key, value in snapshot.original_memos)
              and _digest(engine.events[:snapshot.event_start], engine.decisions[:snapshot.decision_start]) == snapshot.prefix_digest)
        _pending(engine, actors, snapshot.original_pending, unconsumed=False)
        reader = _Reader(engine, snapshot, actors)
        parent = engine.events[snapshot.stop_sequence - 1]
        returned = reader.arrow_keep(parent)
        _need(reader.offset == len(reader.events) and reader.choice_index == len(reader.choices)
              and reader.batch == engine._arrival_counter and returned == stop
              and tuple(engine.pending_effects) == snapshot.original_pending + tuple(reader.children))
        return _arrow_packet(snapshot.event_start, stop.sequence, snapshot.original_pending, reader.children, reader.choice_ids)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, _Reject):
        return False


def key(engine):
    try:
        return _state(engine, _actors(engine))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, _Reject):
        return None


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual:
        return None
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
        engine.score_ledger.snapshot(), engine._arrival_counter, tuple(engine.events), tuple(engine.decisions), _digest(engine.events, engine.decisions))


def job_started(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid:
        return
    if snapshot.in_flight is not None or not queue_before or queue_before[0] != pending:
        snapshot.invalid = True
    else:
        snapshot.in_flight = (pending, queue_before, event_start)


def job_finished(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid:
        return
    if snapshot.in_flight != (pending, queue_before, event_start):
        snapshot.invalid = True
    else:
        snapshot.jobs.append((pending, queue_before, event_start, len(engine.events), tuple(engine.pending_effects)))
        snapshot.in_flight = None


def completed(engine, snapshot):
    try:
        _need(type(snapshot) is Snapshot and type(snapshot.original_pending) is tuple
              and all(_known_pending(job) for job in snapshot.original_pending)
              and not snapshot.invalid and snapshot.in_flight is None and key(engine) == snapshot.state
              and tuple(engine.events[:snapshot.event_start]) == snapshot.event_prefix
              and tuple(engine.decisions[:snapshot.decision_start]) == snapshot.decision_prefix
              and engine.score_ledger.snapshot() == snapshot.score_before
              and _digest(engine.events[:snapshot.event_start], engine.decisions[:snapshot.decision_start]) == snapshot.prefix_digest)
        _pending(engine, _actors(engine), snapshot.original_pending, unconsumed=False)
        reader, fifo = _Reader(engine, snapshot, _actors(engine)), deque(snapshot.original_pending)
        original = [job.queued_sequence for job in fifo]
        executed, silent = [], []
        for pending, before, start, end, after in snapshot.jobs:
            _need(fifo and tuple(fifo) == before and fifo.popleft() == pending
                  and start == snapshot.event_start + reader.offset)
            reader.queue_before = tuple(fifo)
            children = reader.job(pending)
            fifo.extend(children)
            _need(end == snapshot.event_start + reader.offset and tuple(fifo) == after)
            executed.append(pending.queued_sequence)
            if pending.kind == "trip":
                silent.append(pending.queued_sequence)
        fresh = [job.queued_sequence for job in fifo]
        _need(reader.offset == len(reader.events) and reader.choice_index == len(reader.choices)
              and reader.batch == engine._arrival_counter and tuple(fifo) == tuple(engine.pending_effects)
              and [identity for identity in executed if identity in original] == original
              and len(executed) == len(set(executed)) and not set(original).intersection(fresh)
              and any(job.kind == "move" for job in fifo) and all(identity > snapshot.event_start for identity in fresh))
        return {"capture_event_count": snapshot.event_start, "original_pending_ids": original,
                "executed_pending_ids": executed, "executed_silent_trip_ids": silent, "new_pending_ids": fresh,
                "fish_decision_event_ids": reader.choice_ids, "original_jobs_really_returned": True,
                "mechanical_state_returned": True, "scores_unchanged": True,
                "all_choice_proof": "S remains nonempty: initial FOLLOW->3 S; initial KEEP plus one real local KEEP cycle->2 S; Baba T is a genuinely served inert job at this all-tripped FIFO cut."}
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, _Reject):
        return False
