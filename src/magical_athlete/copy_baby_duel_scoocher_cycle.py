"""All-choice S/D10 FIFO between native Baby12 and Copy/Baby11.

S -> S_C,D. KEEP consumes only D. Each of the36 actual Duel dice pairs
returns its winner10->12->11->10 and emits S_D,S_B,S_C,D. Thus an existing
S never disappears. Only real KEEP cuts, after all captured jobs executed,
may stop fresh descendants; existing Duel generators return naturally.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from functools import wraps
import json

from .decisions import Action, ActionKind, DecisionRecord, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import WILD_WILDS
from .romantic_baby_shared12_cycle import _ENGINE_FIELDS, _EVENT_FIELDS, _RACER_FIELDS, _event, _known_pending, _ledger_valid
from .scoocher_baby_duel22_cycle import Frame, _epoch, _past_digest, _mechanics as _base_mechanics, _Reader as _BaseReader, _Reject as _BaseReject
from .fish_baby_arrow11_cycle import _typed_request, _Reject as _RequestReject

REASON = "native_copy_baby_duel_scoocher10_fifo_cycle_proved"
ATTRIBUTE = "_copy_baby_duel_scoocher_scope"


class _Reject(Exception):
    pass


def _need(value):
    if not value:
        raise _Reject


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mechanics(engine, actors):
    # The old Party proof only allows an empty Copy lock. This new direct
    # Copy-turn domain must retain its actual lock in the returned mechanics.
    return _json({"base": _base_mechanics(engine, actors), "copy_locks": sorted(engine._copy_locks)})


def _actors(engine):
    _need(type(engine.racers) is dict and len(engine.racers) == 4)
    by_power = {actor.ability: actor for actor in engine.racers.values()}
    _need(set(by_power) == {Ability.COPY_CAT, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.DUELIST})
    return tuple(by_power[power] for power in (Ability.COPY_CAT, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.DUELIST))


def _base(engine, actors, *, stable):
    c, s, b, d = actors
    _need(not (set(vars(engine)) - (_ENGINE_FIELDS | {ATTRIBUTE})))
    _need(engine.track == WILD_WILDS and engine.termination_reason is None and engine.finish_order == []
          and type(engine.finish_order) is list and type(engine.pending_effects) is deque
          and engine._preparing is False and engine._turn_in_progress is True
          and type(engine._copy_locks) is set
          and engine._copy_locks in (set(), {c.racer_id}) and engine._refreshing_copies is False
          and getattr(engine, "_inert_duel_romantic", None) is None and engine.next_turn_racer_id is None
          and engine._genius_prediction is None and type(engine.score_ledger) is ScoreLedger)
    _need(type(engine.policies) is dict and type(engine.policy_metadata) is dict)
    _need(all(type(getattr(engine, name)) is list for name in
              ("turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn")))
    _need(type(engine.initial_remaining_abilities) is tuple and type(engine.previous_winners) is tuple)
    _need(type(engine.turn_index) is int and 0 <= engine.turn_index < 4
          and sorted(engine.turn_order) == sorted(engine.racers)
          and all(type(rid) is int for rid in engine.turn_order)
          and engine.turn_order[engine.turn_index] in {actor.racer_id for actor in actors})
    _need(not engine._copy_locks or engine.turn_order[engine.turn_index] == c.racer_id)
    _need(type(engine.race_index) is int and 1 <= engine.race_index <= 4
          and type(engine.max_events) is int and engine.max_events > 0
          and type(engine._score_start) is int and engine._score_start >= 0)
    _need(not hasattr(engine, "max_action_events") or type(engine.max_action_events) is int and engine.max_action_events > 0)
    _need(all(type(getattr(engine, name, None)) is int and getattr(engine, name) > 0
              for name in ("_resolution_depth", "_physical_resolution_depth"))
          and type(getattr(engine, "_arrival_counter", 0)) is int and getattr(engine, "_arrival_counter", 0) >= 0
          and type(getattr(engine, "_movement_depth", 0)) is int and getattr(engine, "_movement_depth", 0) == 0)
    offered = getattr(engine, "_duel_offered_epochs", {})
    _need(type(offered) is dict and set(offered) <= {d.racer_id}
          and all(type(key) is int and type(value) is tuple and len(value) == 2
                  and all(type(n) is int and n >= 0 for n in value) for key, value in offered.items()))
    for actor in actors:
        _need(type(actor) is Racer and set(vars(actor)) == _RACER_FIELDS
              and type(actor.racer_id) is int and engine.racers.get(actor.racer_id) is actor
              and type(actor.ability) is Ability and type(actor.status) is RacerStatus
              and actor.status == RacerStatus.ACTIVE and actor.pre_race_resolved is True
              and type(actor.position) is int and type(actor.tripped) is bool and not actor.tripped
              and type(actor.turns_taken) is int and actor.turns_taken >= 0
              and actor.inherited_ability is None and actor.prediction is None and actor.finish_place is None
              and type(actor.finish_places) is tuple and not actor.finish_places)
        if actor is not c:
            _need(actor.copied_target is None and actor.copied_ability is None)
    if stable:
        _need(tuple(actor.position for actor in actors) == (11, 10, 12, 10)
              and c.copied_target == b.racer_id and c.copied_ability is Ability.HUGE_BABY)
    _need(all(type(event) is RaceEvent and type(event.sequence) is int and event.sequence == i + 1
              and set(vars(event)) == _EVENT_FIELDS for i, event in enumerate(engine.events)))
    _need([event.decision_id for event in engine.events if event.kind == "decision"] == list(range(1, len(engine.decisions) + 1))
          and all(type(record) is DecisionRecord for record in engine.decisions)
          and all(_known_pending(job) for job in engine.pending_effects))


def _qualification(engine, actors, job):
    """Exact captured qualification; old started C/S instances remain real."""
    c, s, b, d = (actor.racer_id for actor in actors)
    _need(_known_pending(job) and 0 < job.trigger_sequence < job.queued_sequence <= len(engine.events))
    trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
    if job.racer_id == s:
        _need(job.source_id in {c, b, d} and job.kind == "move" and job.steps == 1
              and job.ability == "Scoocher" and job.target_id is None)
        _need(_event(queued, "effect_queued", s, source_id=job.source_id, ability="Scoocher", parent_sequence=trigger.sequence))
        if job.source_id == d:
            _need(_event(trigger, "ability_effect", d, source_id=d, target_id=s, ability="Duelist", parent_sequence=trigger.parent_sequence))
        elif trigger.ability == "Huge Baby":
            _need(type(trigger.arrival_batch_id) is int and trigger.arrival_batch_id > 0
                  and _event(trigger, "ability_effect", job.source_id, source_id=job.source_id, ability="Huge Baby",
                             arrival_batch_id=trigger.arrival_batch_id, parent_sequence=trigger.parent_sequence))
        else:
            _need(job.source_id == c and _event(trigger, "ability_effect", c, source_id=c, ability="Scoocher", amount=1,
                                             parent_sequence=trigger.parent_sequence))
            _need(type(trigger.parent_sequence) is int and 0 < trigger.parent_sequence < trigger.sequence)
            earlier = engine.events[trigger.parent_sequence - 1]
            _need(earlier.source_id in {s, b} and _event(earlier, "effect_queued", c, source_id=earlier.source_id,
                                                       ability="Scoocher", parent_sequence=earlier.parent_sequence))
        _need(type(trigger.parent_sequence) is int and 0 < trigger.parent_sequence < trigger.sequence)
        return
    _need((job.racer_id, job.source_id, job.kind, job.steps, job.ability, job.target_id)
          == (d, d, "duel", 0, "Duelist", d))
    _need(_event(queued, "effect_queued", d, source_id=d, target_id=d, ability="Duelist", parent_sequence=trigger.sequence)
          and trigger.racer_id in {c, s, d} and type(trigger.before) is int and type(trigger.after) is int
          and type(trigger.arrival_batch_id) is int and type(trigger.parent_sequence) is int
          and 0 < trigger.parent_sequence < trigger.sequence
          and _event(trigger, "stop", trigger.racer_id, before=trigger.before, after=trigger.after,
                     arrival_batch_id=trigger.arrival_batch_id, parent_sequence=trigger.parent_sequence))


class _Reader(_BaseReader):
    def __init__(self, engine, actors, start, batch):
        super().__init__(engine, actors, start, batch, {actor.racer_id: actor.position for actor in actors})
        self.c = self.p

    def choice(self, parent, targets):
        _need(self.offset < len(self.engine.events))
        event = self.engine.events[self.offset]
        _need(type(event.decision_id) is int and 1 <= event.decision_id <= len(self.engine.decisions))
        record = self.engine.decisions[event.decision_id - 1]
        _typed_request(record)
        actors = {actor.racer_id: actor for actor in self.actors}
        public = tuple(PublicRacer(rid, actors[rid].name, self.positions[rid], "active",
                                   "Huge Baby" if rid == self.c else actors[rid].ability.value, False, None,
                                   self.b if rid == self.c else None) for rid in self.engine.turn_order)
        observation = Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1, self.engine.track.name, 30, 15,
            tuple(sorted(self.engine.track.trip_positions)), public, (), None, None,
            score_totals=tuple(self.engine.score_ledger.totals().items()))
        legal = (Action(ActionKind.KEEP), Action(ActionKind.DUEL, self.d, self.s))
        _need(record.request == DecisionRequest(event.decision_id, self.d, DecisionWindow.DUEL, observation, legal)
              and record.action in legal and record.event_sequence == event.sequence)
        action = record.action
        self.take("decision", self.d, source_id=action.source_id, target_id=action.target_id,
                  action_kind=action.kind.value, decision_id=event.decision_id, parent_sequence=parent)
        if action.kind == ActionKind.KEEP:
            self.keeps.append(event.sequence)
        return action

    def pushed(self, actor, parent, source, batch, *, intermediate=False):
        self.take("ability_effect", source, source_id=source, ability="Huge Baby", parent_sequence=parent,
                  arrival_batch_id=batch)
        effect = self.engine.events[self.offset - 1]
        child = self.queued_s(source, effect)
        self.batch += 1
        if intermediate:
            second, returned = self.pushed(actor, effect.sequence, self.c, self.batch)
            return (child, *second), returned
        returned = self.take("stop", actor, before=11, after=10, parent_sequence=effect.sequence, arrival_batch_id=self.batch)
        return (child,), returned

    def duel(self, pending):
        start = self.offset
        identity = pending.queued_sequence
        self.duel_starts[identity] = start
        action = self.choice(identity, self.s)
        if action.kind == ActionKind.KEEP:
            self.counts["D_KEEP"] += 1
            return ()
        effect = self.take("ability_effect", self.d, source_id=self.d, target_id=self.s, ability="Duelist",
                           parent_sequence=self.engine.events[start].sequence)
        first = self.queued_s(self.d, effect)
        dice = []
        for actor in (self.d, self.s):
            value = self.engine.events[self.offset].amount
            _need(type(value) is int and 1 <= value <= 6)
            dice.append(value)
            self.take("duel_roll", actor, source_id=self.d, ability="Duelist", amount=value, parent_sequence=effect.sequence)
        winner = self.d if dice[0] >= dice[1] else self.s
        self.batch += 1
        self.take("move", winner, before=10, after=12, amount=2, source_id=self.d, ability="Duelist",
                  movement_kind="ability", parent_sequence=effect.sequence, arrival_batch_id=self.batch)
        self.take("passing", winner, target_id=self.c, parent_sequence=self.engine.events[self.offset - 1].sequence,
                  arrival_batch_id=self.batch)
        babies, returned = self.pushed(winner, effect.sequence, self.b, self.batch, intermediate=True)
        result = (first, *babies, self.queued_d(returned))
        self.duel_yields[identity] = self.offset
        self.counts["D_DUEL_winner_" + str(winner)] += 1
        return result

    def job(self, pending):
        self.executed.append(pending.queued_sequence)
        if pending.racer_id == self.s:
            _need(pending.source_id in {self.c, self.b, self.d} and pending.kind == "move" and pending.steps == 1
                  and pending.ability == "Scoocher" and pending.target_id is None)
            effect = self.take("ability_effect", self.s, source_id=self.s, ability="Scoocher", amount=1,
                               parent_sequence=pending.queued_sequence)
            self.batch += 1
            self.take("move", self.s, before=10, after=11, amount=1, source_id=self.s, ability="Scoocher",
                      movement_kind="ability", parent_sequence=effect.sequence, arrival_batch_id=self.batch)
            children, returned = self.pushed(self.s, effect.sequence, self.c, self.batch)
            self.counts["S"] += 1
            return (*children, self.queued_d(returned))
        _need((pending.racer_id, pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id)
              == (self.d, self.d, "duel", 0, "Duelist", self.d))
        return self.duel(pending)


@dataclass(frozen=True, slots=True)
class Snapshot:
    start: int
    decisions: int
    pending: tuple
    mechanics: str
    arrival: int
    owners: tuple
    depth: int
    prefix_sha256: str


@dataclass(slots=True)
class Scope:
    start: int
    root_pending: tuple
    drains: list = field(default_factory=list)
    duels: list = field(default_factory=list)
    snapshot: Snapshot | None = None
    closed: bool = False
    certificate: int | None = None
    expected_returns: list = field(default_factory=list)
    actual_returns: list = field(default_factory=list)
    direct_turn_start: bool = False


def _frames(engine, scope):
    _need(type(scope) is Scope and type(scope.start) is int and type(scope.root_pending) is tuple
          and all(_known_pending(job) for job in scope.root_pending)
          and len(scope.drains) == len(scope.duels) + (0 if scope.direct_turn_start else 1)
          and (scope.duels[0].start == scope.start if scope.direct_turn_start else scope.drains[0].start == scope.start)
          and scope.drains[-1].depth == engine._resolution_depth)
    for frame in (*scope.drains, *scope.duels):
        _need(type(frame) is Frame and type(frame.start) is int and type(frame.depth) is int
              and frame.physical == engine._physical_resolution_depth and type(frame.physical) is int
              and type(frame.yielded) is list and all(type(value) is int for value in frame.yielded))
    for index, owner in enumerate(scope.duels):
        child = scope.drains[index if scope.direct_turn_start else index + 1]
        direct = scope.direct_turn_start and index == 0
        parent_depth = (scope.drains[index - 1].depth if scope.direct_turn_start and index else
                        scope.drains[index].depth if not scope.direct_turn_start else 0)
        _need((owner.trigger is None if direct else type(owner.trigger) is int and owner.trigger > 0)
              and owner.depth == parent_depth + 1
              and owner.yielded == [child.start] and child.depth == owner.depth + 1
              and child.trigger == owner.trigger and child.source_id is None)


def capture(engine, scope):
    try:
        actors = _actors(engine)
        _base(engine, actors, stable=True)
        _frames(engine, scope)
        d, s = actors[3].racer_id, actors[1].racer_id
        last = engine.events[-1]
        _need(type(scope) is Scope and scope.drains and last.kind == "decision"
              and last.action_kind == "keep" and last.racer_id == d
              and engine._duel_offered_epochs.get(d) == _epoch(engine)
              and any(job.racer_id == s for job in engine.pending_effects)
              and _ledger_valid(engine, len(engine.events)))
        for job in engine.pending_effects:
            _qualification(engine, actors, job)
        _need(all(type(frame) is Frame and frame.physical == engine._physical_resolution_depth
                  and frame.source_id == d and len(frame.yielded) == 1 for frame in scope.duels))
        return Snapshot(len(engine.events), len(engine.decisions), tuple(engine.pending_effects), _mechanics(engine, actors),
            engine._arrival_counter, tuple(frame.trigger for frame in scope.duels), engine._resolution_depth,
            _past_digest(engine.events, engine.decisions))
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject, _BaseReject, _RequestReject):
        return None


def completed(engine, snapshot):
    try:
        actors = _actors(engine)
        _base(engine, actors, stable=True)
        scope = getattr(engine, ATTRIBUTE)
        _frames(engine, scope)
        _need(type(snapshot) is Snapshot and type(scope) is Scope and not scope.closed
              and type(snapshot.start) is int and type(snapshot.decisions) is int
              and type(snapshot.pending) is tuple and type(snapshot.owners) is tuple
              and all(type(identity) is int or scope.direct_turn_start and index == 0 and identity is None
                      for index, identity in enumerate(snapshot.owners))
              and type(snapshot.arrival) is int and snapshot.arrival > 0
              and type(snapshot.depth) is int and snapshot.depth > 0
              and len(engine.events) > snapshot.start and snapshot.start > scope.start
              and _past_digest(engine.events[:snapshot.start], engine.decisions[:snapshot.decisions]) == snapshot.prefix_sha256
              and _mechanics(engine, actors) == snapshot.mechanics)
        for job in snapshot.pending:
            _qualification(engine, actors, job)
        last = engine.events[-1]
        d = actors[3].racer_id
        _need(last.kind == "decision" and last.action_kind == "keep" and last.racer_id == d
              and engine._duel_offered_epochs.get(d) == _epoch(engine))
        reader = _Reader(engine, actors, snapshot.start, snapshot.arrival)
        fifo = deque(snapshot.pending)
        served = []
        while reader.offset < len(engine.events):
            _need(fifo)
            job = fifo.popleft()
            served.append(job.queued_sequence)
            fifo.extend(reader.job(job))
        originals = [job.queued_sequence for job in snapshot.pending]
        fresh = [job.queued_sequence for job in fifo]
        _need(reader.offset == len(engine.events) and reader.batch == engine._arrival_counter
              and reader.positions == {actor.racer_id: actor.position for actor in actors}
              and tuple(fifo) == tuple(engine.pending_effects)
              and [identity for identity in served if identity in originals] == originals
              and not set(originals).intersection(fresh) and fifo
              and scope.drains[-1].depth == engine._resolution_depth)
        owners = tuple(frame.trigger for frame in scope.duels)
        _need(owners[:len(snapshot.owners)] == snapshot.owners
              and engine._resolution_depth == snapshot.depth + 2 * (len(owners) - len(snapshot.owners)))
        for frame in scope.duels[len(snapshot.owners):]:
            _need(frame.trigger in served and frame.source_id == d and frame.start == reader.duel_starts[frame.trigger]
                  and frame.yielded == [reader.duel_yields[frame.trigger]]
                  and frame.physical == engine._physical_resolution_depth)
        return {"capture_event_count": snapshot.start, "original_pending_ids": originals,
                "executed_pending_ids": served, "new_pending_ids": fresh,
                "Duel_continuations_still_to_return": list(reversed(owners)),
                "original_Duel_body_yields": [{"pending_id": frame.trigger, "real_body_end": frame.yielded[0]}
                                              for frame in scope.duels[:len(snapshot.owners)]],
                "all_choice_invariant": "S->S_C,D; D KEEP->empty; D DUEL all36 dice->S_D,S_B,S_C,D; S never disappears",
                "typed_job_counts": dict(reader.counts), "actual_KEEP_epoch_equality": True,
                "epoch_mutated_by_certificate": False, "all_old_jobs_really_served": True,
                "mechanical_roster_and_full_ledger_returned": True,
                "current_actor_id": engine.turn_order[engine.turn_index],
                "phase": "direct_turn_start" if scope.direct_turn_start else "drain_with_existing_real_qualifications",
                "current_main_already_executed": False if scope.direct_turn_start else None,
                "current_main_die_and_move_still_owed": True if scope.direct_turn_start else None,
                "current_main_debt_not_changed_by_certificate": True,
                "current_turn_completion_still_owed": True}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject, _BaseReject, _RequestReject):
        return None


def probe(engine):
    scope = getattr(engine, ATTRIBUTE, None)
    if type(scope) is not Scope or scope.closed:
        return
    if scope.snapshot is None:
        scope.snapshot = capture(engine, scope)
        return
    result = completed(engine, scope.snapshot)
    if result is None:
        return
    scope.expected_returns = list(result["Duel_continuations_still_to_return"])
    engine._emit("loop_stopped", engine.turn_order[engine.turn_index], reason=REASON,
                 parent_sequence=scope.snapshot.start + 1, amount=len(engine.pending_effects), operation=_json(result))
    scope.closed, scope.certificate = True, len(engine.events)
    engine.pending_effects.clear()


def continuation_returned(engine, scope, trigger):
    if scope.closed:
        _need(scope.expected_returns and scope.expected_returns.pop(0) == trigger
              and len(engine.events) == scope.certificate and not engine.pending_effects
              and engine._duel_offered_epochs.get(_actors(engine)[3].racer_id) == _epoch(engine))
        scope.actual_returns.append(trigger)


def completion(engine, scope):
    if scope.closed:
        _need(not scope.expected_returns and not scope.drains and not scope.duels
              and not engine.pending_effects and len(engine.events) == scope.certificate)


def drain_scope(method):
    @wraps(method)
    def scoped(engine, *args, **kwargs):
        scope = getattr(engine, ATTRIBUTE, None)
        owns = False
        if scope is None:
            try:
                actors = _actors(engine)
                _base(engine, actors, stable=False)
                _need(actors[2].position == 12 and actors[0].copied_target == actors[2].racer_id
                      and actors[0].copied_ability is Ability.HUGE_BABY and engine.pending_effects)
                scope = Scope(len(engine.events), tuple(engine.pending_effects))
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, _Reject):
                yield from method(engine, *args, **kwargs)
                return
            setattr(engine, ATTRIBUTE, scope)
            owns = True
        frame = Frame(scope.duels[-1].trigger if scope.duels else None, len(engine.events),
                      engine._resolution_depth, engine._physical_resolution_depth)
        scope.drains.append(frame)
        failed = False
        try:
            yield from method(engine, *args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            _need(scope.drains and scope.drains[-1] is frame)
            scope.drains.pop()
            if owns:
                try:
                    if not failed:
                        completion(engine, scope)
                finally:
                    delattr(engine, ATTRIBUTE)
    return scoped


def duel_scope(method):
    @wraps(method)
    def scoped(engine, source, trigger_sequence):
        scope = getattr(engine, ATTRIBUTE, None)
        owns = False
        if scope is None and trigger_sequence is None:
            try:
                actors = _actors(engine)
                _base(engine, actors, stable=True)
                _need(source is actors[3] and not engine.pending_effects and engine._resolution_depth == 1)
                scope = Scope(len(engine.events), (), direct_turn_start=True)
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, _Reject):
                scope = None
            if scope is not None:
                setattr(engine, ATTRIBUTE, scope)
                owns = True
        if type(scope) is not Scope:
            yield from method(engine, source, trigger_sequence)
            return
        frame = Frame(trigger_sequence, len(engine.events), engine._resolution_depth,
                      engine._physical_resolution_depth, source_id=source.racer_id)
        scope.duels.append(frame)
        iterator = method(engine, source, trigger_sequence)
        returned = False
        try:
            while True:
                try:
                    child = next(iterator)
                except StopIteration:
                    returned = True
                    break
                frame.yielded.append(len(engine.events))
                yield child
        finally:
            iterator.close()
            _need(scope.duels and scope.duels[-1] is frame)
            scope.duels.pop()
            if returned:
                continuation_returned(engine, scope, trigger_sequence)
            if owns:
                try:
                    if returned:
                        completion(engine, scope)
                finally:
                    delattr(engine, ATTRIBUTE)
    return scoped
