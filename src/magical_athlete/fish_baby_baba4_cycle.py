"""Native F/S/Baby/Baba4 all-choice FIFO, including real silent Trip calls.

This proof is deliberately rooted in an actual Baba main3->4, with both
old qualified Trips served before a clean all-tripped cut. KEEP replaces S
by S_B,T_S; FOLLOW by S_F,S_B,T_F,T_S; an already-tripped T returns empty.
Thus positive S never decreases. Only fresh descendants may stop after the
entire captured original FIFO and its actual zero-event calls have returned.
"""
from collections import deque
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json

from .decisions import DecisionRecord
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import WILD_WILDS
from .fish_baby_star_cycle import _Reader as _FishReader, _Reject
from .fish_baby_arrow11_cycle import _typed_request, _Reject as _RequestReject
from .romantic_baby_shared12_cycle import _ENGINE_FIELDS, _EVENT_FIELDS, _RACER_FIELDS, _event, _known_pending, _ledger_valid

REASON = "native_fish_baby_colocated_baba4_fifo_cycle_proved"
_ROLES = (Ability.SUCKERFISH, Ability.BABA_YAGA, Ability.HUGE_BABY, Ability.SCOOCHER)


def _need(value):
    if not value: raise _Reject


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(events, choices):
    return hashlib.sha256(_canonical([[asdict(e) for e in events], [asdict(d) for d in choices]]).encode()).hexdigest()


def _actors(engine):
    _need(type(engine.racers) is dict and len(engine.racers) == 4)
    by = {actor.ability: actor for actor in engine.racers.values()}
    _need(set(by) == set(_ROLES))
    return tuple(by[role] for role in _ROLES)


def _control(engine, actors):
    f, y, b, s = actors
    _need(not set(vars(engine)) - _ENGINE_FIELDS and engine.track == WILD_WILDS
          and engine.termination_reason is None and type(engine.finish_order) is list and not engine.finish_order
          and type(engine.pending_effects) is deque and engine._preparing is False and engine._turn_in_progress is True
          and type(engine._copy_locks) is set and not engine._copy_locks and engine._refreshing_copies is False
          and getattr(engine, "_inert_duel_romantic", None) is None
          and type(getattr(engine, "_duel_offered_epochs", {})) is dict and not getattr(engine, "_duel_offered_epochs", {})
          and engine.next_turn_racer_id is None and engine._genius_prediction is None
          and type(engine.score_ledger) is ScoreLedger and type(engine.policies) is dict and type(engine.policy_metadata) is dict
          and type(engine.turn_index) is int and 0 <= engine.turn_index < 4
          and engine.turn_order[engine.turn_index] == y.racer_id)
    _need(all(type(getattr(engine, name)) is list for name in
              ("turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn"))
          and len(engine.turn_order) == 4 and set(engine.turn_order) == set(engine.racers)
          and all(type(rid) is int for rid in engine.turn_order)
          and type(engine.initial_remaining_abilities) is tuple and type(engine.previous_winners) is tuple)
    _need(engine.seed is None or type(engine.seed) is int)
    _need(type(engine._score_start) is int and engine._score_start >= 0
          and (not hasattr(engine, "max_action_events") or type(engine.max_action_events) is int and engine.max_action_events > 0))
    _need(all(type(getattr(engine, name, None)) is int and getattr(engine, name) > 0
              for name in ("_resolution_depth", "_physical_resolution_depth", "_arrival_counter", "race_index", "max_events"))
          and type(getattr(engine, "_movement_depth", 0)) is int and getattr(engine, "_movement_depth", 0) == 0)
    for actor in actors:
        _need(type(actor) is Racer and set(vars(actor)) == _RACER_FIELDS
              and type(actor.racer_id) is int and engine.racers.get(actor.racer_id) is actor
              and type(actor.ability) is Ability and type(actor.status) is RacerStatus and actor.status == RacerStatus.ACTIVE
              and actor.pre_race_resolved is True and type(actor.position) is int and type(actor.tripped) is bool
              and type(actor.turns_taken) is int and actor.turns_taken >= 0
              and actor.inherited_ability is None and actor.copied_target is None and actor.copied_ability is None
              and actor.prediction is None and actor.finish_place is None and type(actor.finish_places) is tuple and not actor.finish_places)
    _need(tuple(actor.position for actor in actors) == (4, 4, 5, 4) and f.tripped is s.tripped is True)
    initial = {row["racer_id"]: row for row in engine.initial_racers}
    _need(len(initial) == 4 and set(initial) == set(engine.racers)
          and all(initial[actor.racer_id]["ability"] == actor.ability.value and initial[actor.racer_id]["name"] == actor.name for actor in actors))
    _need(all(type(e) is RaceEvent and set(vars(e)) == _EVENT_FIELDS and type(e.sequence) is int and e.sequence == i + 1
              for i, e in enumerate(engine.events)) and all(type(d) is DecisionRecord for d in engine.decisions))
    _need([e.decision_id for e in engine.events if e.kind == "decision"] == list(range(1, len(engine.decisions) + 1)))
    _need(all(_known_pending(job) for job in engine.pending_effects))


@dataclass(slots=True)
class Snapshot:
    state: tuple
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


class _Reader(_FishReader):
    def __init__(self, engine, snapshot, actors, *, initial_trip=False):
        super().__init__(engine, snapshot, actors)
        self.trip = {self.f: not initial_trip, self.s: not initial_trip}
        self.stops, self.silent = [], 0

    def take(self, kind, actor, **values):
        _need(self.offset < len(self.events))
        event = self.events[self.offset]
        _need(_event(event, kind, actor, **values))
        self.offset += 1
        return event

    def star(self, actor, stop):
        self.stops.append(stop)  # p4 is not a Star; authenticate Baba below.

    def job(self, pending):
        _need(_known_pending(pending))
        if pending.kind == "trip":
            _need((pending.racer_id, pending.source_id, pending.steps, pending.ability) == (self.p, self.p, 1, "Baba Yaga")
                  and pending.target_id in self.trip)
            self.executed.append(pending.queued_sequence)
            if self.trip[pending.target_id]:
                self.silent += 1
                return ()
            effect = self.take("trip", pending.target_id, source_id=self.p, ability="Baba Yaga", parent_sequence=pending.queued_sequence)
            self.trip[pending.target_id] = True
            queued = self.take("effect_queued", self.s, source_id=self.p, ability="Scoocher", parent_sequence=effect.sequence)
            return (PendingEffect(self.s, self.p, effect.sequence, queued.sequence),)
        _need((pending.racer_id, pending.target_id, pending.kind, pending.steps, pending.ability) == (self.s, None, "move", 1, "Scoocher")
              and pending.source_id in {self.f, self.b, self.p} and all(self.trip.values()))
        try: _typed_request(self.choices[self.choice_index])
        except _RequestReject: raise _Reject
        self.stops = []
        children = list(super().job(pending))
        for stop in self.stops:
            queued = self.take("effect_queued", self.p, source_id=self.p, target_id=stop.racer_id,
                               ability="Baba Yaga", parent_sequence=stop.sequence)
            children.append(PendingEffect(self.p, self.p, stop.sequence, queued.sequence, "trip", 1, "Baba Yaga", stop.racer_id))
        return tuple(children)


def _root(engine, actors):
    """Derive the real main root, never recognize a pre-set queue alone."""
    f, y, b, s = (actor.racer_id for actor in actors)
    root = next(e for e in reversed(engine.events) if e.kind == "main_move_resolved")
    index = root.sequence - 3
    _need(root.racer_id == y and index >= 0)
    positions = {row["racer_id"]: row["position"] for row in engine.initial_racers}
    trips = {row["racer_id"]: row["tripped"] for row in engine.initial_racers}
    for event in engine.events[:index]:
        if event.kind in {"move", "warp", "stop"}: positions[event.racer_id] = event.after
        elif event.kind == "trip":
            _need(trips[event.racer_id] is False)
            trips[event.racer_id] = True
        elif event.kind == "recover":
            _need(trips[event.racer_id] is True)
            trips[event.racer_id] = False
        elif event.kind in {"inherit", "copy_changed", "finish", "eliminate"}: raise _Reject
    _need(positions == {f: 4, y: 3, b: 5, s: 4} and trips[f] is trips[s] is False
          and trips[y] == actors[1].tripped and trips[b] == actors[2].tripped and _ledger_valid(engine, index))
    def take(offset, kind, actor, **values):
        e = engine.events[index + offset]
        _need(_event(e, kind, actor, **values))
        return e
    roll = take(0, "roll", y, amount=1)
    locked = take(1, "roll_locked", y, amount=1, parent_sequence=roll.sequence)
    resolved = take(2, "main_move_resolved", y, before=1, after=1, parent_sequence=locked.sequence)
    batch = engine.events[index + 3].arrival_batch_id
    _need(type(batch) is int and batch > 0)
    moved = take(3, "move", y, before=3, after=4, amount=1, movement_kind="main", parent_sequence=resolved.sequence, arrival_batch_id=batch)
    stop = take(4, "stop", y, before=3, after=4, parent_sequence=moved.sequence, arrival_batch_id=batch)
    jobs = []
    for offset, target in enumerate((rid for rid in engine._clockwise_order() if rid in {f, s}), 5):
        event = take(offset, "effect_queued", y, source_id=y, target_id=target, ability="Baba Yaga", parent_sequence=stop.sequence)
        jobs.append(PendingEffect(y, y, stop.sequence, event.sequence, "trip", 1, "Baba Yaga", target))
    end = index + 7
    beginning = Snapshot((), end, sum(e.kind == "decision" for e in engine.events[:end]), tuple(jobs),
                         engine.score_ledger.snapshot(), batch, (), (), "")
    reader, queue = _Reader(engine, beginning, actors, initial_trip=True), deque(jobs)
    while reader.offset < len(reader.events):
        _need(queue)
        queue.extend(reader.job(queue.popleft()))
    # A genuine no-event Trip may have returned since the last visible event.
    # Only leading already-tripped T can be removed here, never an S body.
    while tuple(queue) != tuple(engine.pending_effects):
        _need(queue and queue[0].kind == "trip")
        _need(reader.job(queue.popleft()) == ())
    _need(reader.choice_index == len(reader.choices) and reader.batch == engine._arrival_counter)
    return index, end


def key(engine):
    try:
        actors = _actors(engine)
        _control(engine, actors)
        _need(engine.pending_effects and any(job.kind == "move" for job in engine.pending_effects))
        root = _root(engine, actors)
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor))) for rid, actor in sorted(engine.racers.items()))
        state = (roster, engine.score_ledger.snapshot(), tuple(engine.turns), engine.race_index, tuple(engine.turn_order), engine.turn_index,
                 tuple(engine.remaining_abilities), tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._resolution_depth,
                 engine._physical_resolution_depth, engine.max_events, getattr(engine, "max_action_events", None), engine.seed,
                 _canonical([engine.initial_racers, engine.initial_score_ledger, engine.policy_metadata, engine.initial_remaining_abilities]),
                 tuple(sorted(engine.policies)), tuple(sorted(vars(engine))), engine.initial_code_fingerprint, engine._score_start,
                 root, _digest(engine.events[:root[1]], engine.decisions[:sum(e.kind == "decision" for e in engine.events[:root[1]])]))
        hash(state)
        return state
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, StopIteration, _Reject): return None


def capture(engine, state=None):
    current = key(engine)
    if current is None or state is not None and current != state: return None
    return Snapshot(current, len(engine.events), len(engine.decisions), tuple(engine.pending_effects), engine.score_ledger.snapshot(),
                    engine._arrival_counter, tuple(engine.events), tuple(engine.decisions), _digest(engine.events, engine.decisions))


def job_started(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid: return
    if snapshot.in_flight is not None or not queue_before or queue_before[0] != pending:
        snapshot.invalid = True
        return
    snapshot.in_flight = pending, queue_before, event_start


def job_finished(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid: return
    if snapshot.in_flight != (pending, queue_before, event_start):
        snapshot.invalid = True
        return
    snapshot.jobs.append((pending, queue_before, event_start, len(engine.events)))
    snapshot.in_flight = None


def completed(engine, snapshot):
    try:
        _need(type(snapshot) is Snapshot and not snapshot.invalid and snapshot.in_flight is None and snapshot.jobs
              and key(engine) == snapshot.state
              and tuple(engine.events[:snapshot.event_start]) == snapshot.event_prefix
              and tuple(engine.decisions[:snapshot.decision_start]) == snapshot.decision_prefix
              and _digest(engine.events[:snapshot.event_start], engine.decisions[:snapshot.decision_start]) == snapshot.prefix_digest)
        reader = _Reader(engine, snapshot, _actors(engine))
        queue, served = deque(snapshot.original_pending), []
        for job, before, start, end in snapshot.jobs:
            _need(tuple(queue) == before and queue.popleft() == job and start == snapshot.event_start + reader.offset)
            queue.extend(reader.job(job))
            _need(end == snapshot.event_start + reader.offset)
            served.append(job.queued_sequence)
        originals = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in queue]
        _need([identity for identity in served if identity in originals] == originals
              and len(served) == len(set(served)) and not set(originals).intersection(fresh)
              and tuple(queue) == tuple(engine.pending_effects) and reader.offset == len(reader.events)
              and reader.choice_index == len(reader.choices) and reader.batch == engine._arrival_counter
              and any(job.kind == "move" for job in queue) and all(identity > snapshot.event_start for identity in fresh))
        return {"capture_event_count": snapshot.event_start, "original_pending_ids": originals, "executed_pending_ids": served,
                "new_pending_ids": fresh, "keep_jobs": len(reader.choice_ids) - reader.follow_count, "follow_jobs": reader.follow_count,
                "silent_trip_jobs_actually_finished": reader.silent, "suckerfish_decision_event_ids": reader.choice_ids,
                "mechanical_state_returned": True, "scores_unchanged": True, "all_original_jobs_executed": True,
                "only_fresh_pending_descendants_stopped": True, "current_main_roll_and_move_already_completed": True,
                "proof": "S KEEP->S_B,T_S; S FOLLOW->S_F,S_B,T_F,T_S; already-tripped T->empty; positive S never decreases"}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, StopIteration, _Reject): return False
