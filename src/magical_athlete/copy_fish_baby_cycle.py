"""Authenticate the all-choice Copy/Fish/Baby FIFO with a fixed Fish leader.

The original Fish stays strictly beyond p+1, while Copy (borrowing that
original Fish) and Scoocher return to p and Baby stays at p+1. Every actual
KEEP replaces a Scoocher notification by [Baby]; every actual FOLLOW replaces
it by [Copy, Baby]. Both substitutions are nonempty for every legal choice.
All captured original jobs, choices, effects and returned stops must really
run before the caller may stop only their freshly qualified descendants.
This module only recognizes and authenticates; it executes or removes nothing.
"""

from collections import deque
from dataclasses import dataclass, fields
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow
from .events import PendingEffect, RaceEvent
from .fish_baby_star_cycle import _Reader, _Reject, _event
from .racer import Ability, Racer
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS


REASON = "copy_fish_baby_choice_queue_cycle_proved"
_ROLES = (Ability.COPY_CAT, Ability.SUCKERFISH, Ability.HUGE_BABY, Ability.SCOOCHER)
# A new engine control must be reviewed before this deliberately small proof
# can project it away. Policy internals may evolve on real choices; dice are
# never consulted in the authenticated body. Neither is inspected here.
_ENGINE_FIELDS = frozenset({
    "track", "max_events", "_turn_in_progress", "_preparing", "initial_remaining_abilities",
    "remaining_abilities", "previous_winners", "egg_drawn", "_copy_locks", "_refreshing_copies",
    "race_index", "score_ledger", "initial_score_ledger", "_score_start", "racers", "turn_order",
    "dice", "seed", "policies", "policy_metadata", "decisions", "finish_order", "turns", "events",
    "pending_effects", "turn_index", "next_turn_racer_id", "termination_reason", "initial_racers",
    "initial_code_fingerprint", "_genius_prediction", "_resolution_depth", "_physical_resolution_depth",
    "_arrival_counter", "_movement_depth", "_inert_duel_romantic", "_duel_offered_epochs",
    "max_action_events",
})
_RACER_FIELDS = frozenset(field.name for field in fields(Racer))
_EVENT_FIELDS = frozenset(field.name for field in fields(RaceEvent))


def _actors(engine):
    if len(engine.racers) != 4:
        return None
    by_power = {actor.ability: actor for actor in engine.racers.values()}
    if set(by_power) != set(_ROLES):
        return None
    return tuple(by_power[power] for power in _ROLES)


def _copy_grant(engine, actors):
    copy, anchor, _, _ = actors
    if (copy.copied_target != anchor.racer_id or type(copy.copied_target) is not int
            or copy.copied_ability != Ability.SUCKERFISH):
        return None
    initial = {row["racer_id"]: row for row in engine.initial_racers}
    if (len(initial) != 4 or set(initial) != set(engine.racers)
            or any(initial[actor.racer_id]["ability"] != power.value
                   for actor, power in zip(actors, _ROLES))):
        return None
    grants = [event for event in engine.events
              if event.kind == "copy_changed" and event.racer_id == copy.racer_id]
    if not grants:
        return None
    grant = grants[-1]
    if (not 1 <= grant.sequence <= len(engine.events)
            or engine.events[grant.sequence - 1] != grant
            or not _event(grant, "copy_changed", copy.racer_id, source_id=copy.racer_id,
                          target_id=anchor.racer_id, ability=Ability.SUCKERFISH.value)):
        return None
    return grant.sequence


def _qualified(engine, pending, actors, grant, *, require_unconsumed=True, authenticate_birth=True):
    copy, _, baby, scoocher = actors
    if not pending or any(type(job) is not PendingEffect or set(vars(job)) != {
            field.name for field in fields(PendingEffect)} for job in pending):
        return False
    identities = [job.queued_sequence for job in pending]
    if (any(type(identity) is not int for identity in identities)
            or identities != sorted(set(identities))):
        return False
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}}
    if require_unconsumed and consumed.intersection(identities):
        return False
    for job in pending:
        if (any(type(value) is not int for value in (job.racer_id, job.source_id, job.steps,
                                                       job.trigger_sequence, job.queued_sequence))
                or (job.racer_id, job.kind, job.steps, job.ability, job.target_id) != (
                    scoocher.racer_id, "move", 1, Ability.SCOOCHER.value, None)
                or job.source_id not in {copy.racer_id, baby.racer_id}
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)):
            return False
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if (type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                or not _event(queued, "effect_queued", scoocher.racer_id, source_id=job.source_id,
                      ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)):
            return False
        if job.source_id == copy.racer_id:
            if (trigger.sequence <= grant
                    or not _event(trigger, "ability_effect", copy.racer_id, source_id=copy.racer_id,
                                  target_id=scoocher.racer_id, ability=Ability.SUCKERFISH.value,
                                  amount=1, parent_sequence=trigger.parent_sequence)):
                return False
        elif (type(trigger.arrival_batch_id) is not int or not 1 <= trigger.arrival_batch_id <= engine._arrival_counter
                or not _event(trigger, "ability_effect", baby.racer_id, source_id=baby.racer_id,
                              ability=Ability.HUGE_BABY.value, parent_sequence=trigger.parent_sequence,
                              arrival_batch_id=trigger.arrival_batch_id)):
            return False
    return not authenticate_birth or _births_valid(engine, pending, actors, grant)


def key(engine):
    """Complete mechanical state at a clean, source-authenticated FIFO cut."""
    try:
        if (set(vars(engine)) - _ENGINE_FIELDS or engine.track != WILD_WILDS
                or engine.termination_reason is not None or engine.finish_order
                or engine._preparing is not False or engine._turn_in_progress is not True
                or engine._copy_locks or engine._refreshing_copies is not False
                or getattr(engine, "_movement_depth", 0) != 0
                or getattr(engine, "_inert_duel_romantic", None) is not None
                or getattr(engine, "_duel_offered_epochs", {})
                or type(getattr(engine, "_resolution_depth", None)) is not int or engine._resolution_depth < 1
                or type(getattr(engine, "_physical_resolution_depth", None)) is not int
                or engine._physical_resolution_depth < 1
                or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
                or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
                or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
                or engine.next_turn_racer_id is not None or engine._genius_prediction is not None):
            return None
        actors = _actors(engine)
        if actors is None:
            return None
        if any(type(event) is not RaceEvent or set(vars(event)) != _EVENT_FIELDS
               or type(event.sequence) is not int or event.sequence != index + 1
               for index, event in enumerate(engine.events)):
            return None
        copy, anchor, baby, scoocher = actors
        p = scoocher.position
        if (type(p) is not int or not 1 <= p < p + 1 < anchor.position < engine.track.finish_position
                or copy.position != p or baby.position != p + 1
                or p in engine.track.star_positions | engine.track.trip_positions
                or p in dict(engine.track.arrows)):
            return None
        for actor in actors:
            if (type(actor) is not Racer or set(vars(actor)) != _RACER_FIELDS
                    or type(actor.racer_id) is not int or engine.racers.get(actor.racer_id) is not actor
                    or type(actor.position) is not int or not 0 <= actor.position < engine.track.finish_position
                    or actor.status != RacerStatus.ACTIVE or actor.pre_race_resolved is not True
                    or actor.inherited_ability is not None or actor.finish_place is not None or actor.finish_places
                    or actor.prediction is not None or type(actor.tripped) is not bool
                    or type(actor.turns_taken) is not int or actor.turns_taken < 0
                    or actor is not copy and (actor.copied_target is not None or actor.copied_ability is not None)):
                return None
        grant = _copy_grant(engine, actors)
        if grant is None or not _qualified(engine, tuple(engine.pending_effects), actors, grant):
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                              "policy_metadata": engine.policy_metadata}, sort_keys=True, separators=(",", ":"))
        state = (engine.track, roster, tuple(engine.turns), engine.turn_index, tuple(engine.turn_order),
                 engine.race_index, tuple(engine.remaining_abilities), tuple(engine.initial_remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._preparing, engine._turn_in_progress,
                 engine._resolution_depth, engine._physical_resolution_depth, initial, engine._score_start,
                 engine.initial_code_fingerprint, engine.max_events, getattr(engine, "max_action_events", None),
                 engine.seed, tuple(sorted(engine.policies)), grant, tuple(sorted(vars(engine))))
        hash(state)
        return state
    except (AttributeError, KeyError, ValueError, TypeError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple[PendingEffect, ...]
    score_before: ScoreSnapshot
    arrival_counter: int
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        raise ValueError("not a certified original-Fish/Copy/Baby FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine.score_ledger.snapshot(), engine._arrival_counter,
                    tuple(engine.events), tuple(engine.decisions))


class _OrdinaryReader(_Reader):
    def job(self, pending):
        record = self.choices[self.choice_index]
        if type(record.request.player_id) is not int or type(record.event_sequence) is not int:
            raise _Reject
        return super().job(pending)

    def star(self, actor, stop):
        # p is a verified ordinary square. Any actual score event is therefore
        # left unconsumed and rejected by the exact body parser below.
        pass


def _source_reader(engine, actors, event_start, decision_start, arrival_counter):
    snapshot = Snapshot((), event_start, decision_start, (), engine.score_ledger.snapshot(), arrival_counter, (), ())
    return _OrdinaryReader(engine, snapshot, actors)


def _root_children(engine, actors, parent):
    """Authenticate a real current main move, including its complete Baby batch."""
    mimic, _, baby, scoocher = actors
    mover, amount = parent.racer_id, parent.after
    if (mover not in {mimic.racer_id, scoocher.racer_id}
            or mover != engine.turn_order[engine.turn_index] or type(amount) is not int
            or not 1 <= amount <= 6 or mover == scoocher.racer_id and amount != 1
            or type(parent.parent_sequence) is not int or parent.parent_sequence != parent.sequence - 1
            or not _event(parent, "main_move_resolved", mover, before=amount, after=amount,
                          parent_sequence=parent.parent_sequence)):
        raise _Reject
    locked = engine.events[parent.parent_sequence - 1]
    if (type(locked.parent_sequence) is not int or locked.parent_sequence != locked.sequence - 1
            or not _event(locked, "roll_locked", mover, amount=amount, parent_sequence=locked.parent_sequence)
            or not _event(engine.events[locked.parent_sequence - 1], "roll", mover, amount=amount)):
        raise _Reject
    p, children = scoocher.position, []
    if engine.racers[mover].tripped is not False or not 0 <= p + 1 - amount < p + 1:
        raise _Reject
    direct = next((event for event in engine.events[parent.sequence:]
                   if event.kind == "move" and event.racer_id == mover), None)
    if (direct is None or type(direct.arrival_batch_id) is not int or direct.arrival_batch_id < 1
            or direct.arrival_batch_id + 1 > engine._arrival_counter):
        raise _Reject
    batch = direct.arrival_batch_id
    next_event = engine.events[parent.sequence]
    decision_start = next_event.decision_id - 1 if mover == scoocher.racer_id and type(next_event.decision_id) is int else 0
    reader = _source_reader(engine, actors, parent.sequence, decision_start, batch - 1)
    follow = False
    if mover == scoocher.racer_id:
        record = reader.choices[0]
        request, action = record.request, record.action
        legal = (Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, mimic.racer_id, scoocher.racer_id))
        if (type(request.decision_id) is not int or type(request.player_id) is not int
                or type(record.event_sequence) is not int or action not in legal
                or request != DecisionRequest(decision_start + 1, mimic.racer_id, DecisionWindow.MOVE_START,
                    reader.observe(), legal)):
            raise _Reject
        decision = reader.take("decision", mimic.racer_id, source_id=action.source_id, target_id=action.target_id,
                               parent_sequence=parent.sequence, action_kind=action.kind.value,
                               decision_id=request.decision_id)
        if decision.sequence != record.event_sequence:
            raise _Reject
        follow = action.kind == ActionKind.SUCKERFISH_FOLLOW
        if follow:
            fish = reader.take("ability_effect", mimic.racer_id, source_id=mimic.racer_id,
                               target_id=scoocher.racer_id, ability=Ability.SUCKERFISH.value,
                               amount=1, parent_sequence=parent.sequence)
            queued = reader.take("effect_queued", scoocher.racer_id, source_id=mimic.racer_id,
                                 ability=Ability.SCOOCHER.value, parent_sequence=fish.sequence)
            children.append(PendingEffect(scoocher.racer_id, mimic.racer_id, fish.sequence, queued.sequence))
    move = reader.take("move", mover, before=p + 1 - amount, after=p + 1, amount=amount,
                       movement_kind="main", parent_sequence=parent.sequence, arrival_batch_id=batch)
    if mover == mimic.racer_id and amount > 1:
        reader.take("passing", mover, target_id=scoocher.racer_id, parent_sequence=move.sequence,
                    arrival_batch_id=batch)
    if follow:
        reader.take("move", mimic.racer_id, source_id=mimic.racer_id, ability=Ability.SUCKERFISH.value,
                    before=p, after=p + 1, amount=1, movement_kind="ability",
                    parent_sequence=fish.sequence, arrival_batch_id=batch)
    power = reader.take("ability_effect", baby.racer_id, source_id=baby.racer_id,
                        ability=Ability.HUGE_BABY.value, parent_sequence=parent.sequence, arrival_batch_id=batch)
    queued = reader.take("effect_queued", scoocher.racer_id, source_id=baby.racer_id,
                         ability=Ability.SCOOCHER.value, parent_sequence=power.sequence)
    children.append(PendingEffect(scoocher.racer_id, baby.racer_id, power.sequence, queued.sequence))
    returned = {mover, mimic.racer_id} if follow else {mover}
    for rid in engine._clockwise_order():
        if rid in returned:
            reader.take("stop", rid, before=p + 1, after=p, parent_sequence=power.sequence,
                        arrival_batch_id=batch + 1)
    return tuple(children)


def _births_valid(engine, pending, actors, grant):
    """Trace actual complete birth bodies back to the real locked main roll.

    Every edge goes to an older queued ID. An iterative worklist avoids
    recursion proportional to a genuine long queue's saved ancestry.
    """
    work, checked, births = list(pending), set(), {}
    try:
        while work:
            job = work.pop()
            if job.queued_sequence in checked:
                continue
            if not _qualified(engine, (job,), actors, grant, require_unconsumed=False, authenticate_birth=False):
                return False
            trigger = engine.events[job.trigger_sequence - 1]
            parent = engine.events[trigger.parent_sequence - 1]
            if parent.sequence <= grant:
                return False
            if parent.sequence not in births:
                if parent.kind == "main_move_resolved":
                    births[parent.sequence] = _root_children(engine, actors, parent)
                elif parent.kind == "ability_effect":
                    scoocher = actors[3].racer_id
                    if (type(parent.parent_sequence) is not int
                            or not 1 <= parent.parent_sequence < parent.sequence):
                        return False
                    queued = engine.events[parent.parent_sequence - 1]
                    ancestor = PendingEffect(scoocher, queued.source_id, queued.parent_sequence, queued.sequence)
                    if not _qualified(engine, (ancestor,), actors, grant, require_unconsumed=False, authenticate_birth=False):
                        return False
                    decision = engine.events[parent.sequence - 2]
                    if (decision.kind != "decision" or type(decision.decision_id) is not int
                            or decision.decision_id < 1):
                        return False
                    following = decision.action_kind == ActionKind.SUCKERFISH_FOLLOW.value
                    move = engine.events[parent.sequence + (2 if following else 0)]
                    if type(move.arrival_batch_id) is not int or move.arrival_batch_id < 1:
                        return False
                    reader = _source_reader(engine, actors, decision.sequence - 1,
                                            decision.decision_id - 1, move.arrival_batch_id - 1)
                    births[parent.sequence] = reader.job(ancestor)
                    work.append(ancestor)
                else:
                    return False
            if job not in births[parent.sequence]:
                return False
            checked.add(job.queued_sequence)
        return True
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False


def completed(engine, snapshot):
    try:
        if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
                or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix
                or engine.score_ledger.snapshot() != snapshot.score_before):
            return False
        actors = _actors(engine)
        grant = _copy_grant(engine, actors)
        if not _qualified(engine, snapshot.original_pending, actors, grant, require_unconsumed=False):
            return False
        events = engine.events[snapshot.event_start:]
        if not events or any(type(event.sequence) is not int or event.sequence != snapshot.event_start + i + 1
                             for i, event in enumerate(events)):
            return False
        reader, fifo = _OrdinaryReader(engine, snapshot, actors), deque(snapshot.original_pending)
        while reader.offset < len(events):
            if not fifo:
                return False
            fifo.extend(reader.job(fifo.popleft()))
        original = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        if (reader.choice_index != len(reader.choices) or reader.batch != engine._arrival_counter
                or [identity for identity in reader.executed if identity in original] != original
                or len(reader.executed) != len(set(reader.executed)) or not fifo
                or set(original).intersection(fresh) or any(identity <= snapshot.event_start for identity in fresh)
                or tuple(fifo) != tuple(engine.pending_effects)):
            return False
        copy, anchor, _, scoocher = actors
        return {"capture_event_count": snapshot.event_start,
                "original_pending_ids": original, "executed_pending_ids": reader.executed,
                "new_pending_ids": fresh, "queue_before": len(original), "queue_after": len(fresh),
                "suckerfish_decision_event_ids": reader.choice_ids,
                "keep_jobs": len(reader.executed) - reader.follow_count, "follow_jobs": reader.follow_count,
                "return_position": scoocher.position, "fixed_original_fish_position": anchor.position,
                "copy_grant_sequence": grant,
                "copy_target_binding": {"copy_racer_id": copy.racer_id, "target_racer_id": anchor.racer_id,
                                        "original_ability": Ability.SUCKERFISH.value,
                                        "borrowed_ability": Ability.SUCKERFISH.value},
                "all_original_jobs_executed": True, "mechanical_state_returned": True,
                "only_fresh_pending_descendants_stopped": True, "score_ledger_unchanged": True,
                "proof": "fixed unique original Fish leader; every legal copied-Fish choice returns complete mechanics and nonempty FIFO: KEEP -> Baby; FOLLOW -> Copy,Baby"}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False
