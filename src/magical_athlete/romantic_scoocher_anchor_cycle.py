"""A forced Romantic/Scoocher FIFO with original Dicemonger22 and Legs15.

R22->24->22 replaces each R by SR. S traverses12,13,14,15,16->12;
only S14 adds R at Legs15, and S12 earns a real Star. Some S jobs have
NO child. The number of R jobs never decreases, so a valid FIFO containing
R cannot drain. Before stopping only fresh descendants, authenticate every
original qualification and actual job, a full mechanical return, a complete
S phase lap, and every real Star entry. This module never clears a queue.
"""
from collections import Counter, deque
from dataclasses import dataclass, fields
import json

from .events import PendingEffect, RaceEvent
from .racer import Ability
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS


REASON = "romantic_scoocher_anchor_queue_cycle_proved"


def _event(event, kind, actor, **values):
    numeric = ("sequence", "racer_id", "source_id", "target_id", "before", "after", "amount",
               "parent_sequence", "decision_id", "arrival_batch_id")
    return (isinstance(event, RaceEvent)
            and all(getattr(event, name) is None or type(getattr(event, name)) is int for name in numeric)
            and event == RaceEvent(event.sequence, kind, actor, **values))


def _actors(engine):
    wanted = (Ability.SCOOCHER, Ability.DICEMONGER, Ability.ROMANTIC, Ability.LEGS)
    if len(engine.racers) != 4 or {actor.ability for actor in engine.racers.values()} != set(wanted):
        return None
    actors = tuple(next(actor for actor in engine.racers.values() if actor.ability == power) for power in wanted)
    s, d, r, l = actors
    if (type(s.position) is not int or s.position not in {12, 13, 14, 15}
            or (d.position, r.position, l.position) != (22, 22, 15)):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE or actor.pre_race_resolved is not True
                or actor.inherited_ability is not None or actor.copied_target is not None
                or actor.copied_ability is not None or actor.prediction is not None
                or actor.finish_place is not None or actor.finish_places
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or type(actor.tripped) is not bool or type(actor.turns_taken) is not int or actor.turns_taken < 0):
            return None
    return actors


def _queue_valid(engine, pending, actors, *, require_unconsumed=True):
    s, d, r, l = (actor.racer_id for actor in actors)
    ids = [job.queued_sequence for job in pending if isinstance(job, PendingEffect)]
    if (not pending or len(ids) != len(pending) or ids != sorted(set(ids))
            or {job.racer_id for job in pending} != {s, r}):
        return False
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    if (set(initial) != set(engine.racers)
            or any(initial[actor.racer_id]["ability"] != actor.ability
                   or initial[actor.racer_id]["inherited_ability"] is not None
                   or initial[actor.racer_id]["copied_ability"] is not None
                   or initial[actor.racer_id]["copied_target"] is not None for actor in actors)):
        return False
    positions = {rid: item["position"] for rid, item in initial.items()}
    history, batches, consumed = {}, Counter(), set()
    for index, event in enumerate(engine.events):
        if type(event.sequence) is not int or event.sequence != index + 1:
            return False
        if event.kind in {"inherit", "finish", "eliminate", "race_end"}:
            return False
        if event.kind in {"move", "warp", "stop"}:
            if event.racer_id not in positions or type(event.after) is not int:
                return False
            positions[event.racer_id] = event.after
        if event.kind == "stop":
            history[event.sequence] = dict(positions)
            batches[event.arrival_batch_id] += 1
        if event.kind in {"ability_effect", "effect_cancelled"}:
            consumed.add(event.parent_sequence)
    if positions != {rid: actor.position for rid, actor in engine.racers.items()}:
        return False

    def real_r_birth(queued):
        trigger = queued.parent_sequence
        if type(trigger) is not int or not 1 <= trigger < queued.sequence <= len(engine.events):
            return False
        stop = engine.events[trigger - 1]
        occupied = history.get(trigger, {})
        expected = {r, d} if (stop.racer_id, stop.after) == (r, 22) else {s, l}
        return (stop.kind == "stop" and (stop.racer_id, stop.after) in {(r, 22), (s, 15)}
                and type(stop.before) is int and type(stop.parent_sequence) is int
                and 1 <= stop.parent_sequence < stop.sequence
                and type(stop.arrival_batch_id) is int and stop.arrival_batch_id > 0
                and batches[stop.arrival_batch_id] == 1
                and {rid for rid, position in occupied.items() if position == stop.after} == expected
                and _event(stop, "stop", stop.racer_id, before=stop.before, after=stop.after,
                           parent_sequence=stop.parent_sequence, arrival_batch_id=stop.arrival_batch_id)
                and _event(queued, "effect_queued", r, source_id=r, target_id=r,
                           ability=Ability.ROMANTIC.value, parent_sequence=trigger))

    for job in pending:
        if (type(job.steps) is not int or type(job.trigger_sequence) is not int
                or type(job.queued_sequence) is not int
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)
                or require_unconsumed and job.queued_sequence in consumed):
            return False
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if job.racer_id == r:
            if (job != PendingEffect(r, r, job.trigger_sequence, job.queued_sequence, "move", 2,
                                     Ability.ROMANTIC.value, r) or not real_r_birth(queued)):
                return False
        elif job.racer_id == s:
            identity = trigger.parent_sequence
            if (job != PendingEffect(s, r, job.trigger_sequence, job.queued_sequence)
                    or type(identity) is not int or not 1 <= identity < trigger.sequence
                    or not _event(trigger, "ability_effect", r, source_id=r, ability=Ability.ROMANTIC.value,
                                  amount=2, parent_sequence=identity)
                    or not real_r_birth(engine.events[identity - 1])
                    or queued.sequence != trigger.sequence + 1
                    or not _event(queued, "effect_queued", s, source_id=r, ability=Ability.SCOOCHER.value,
                                  parent_sequence=trigger.sequence)):
                return False
        else:
            return False
    return True


def key(engine):
    if (engine.track != WILD_WILDS or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0
            or type(getattr(engine, "_resolution_depth", None)) is not int or engine._resolution_depth < 1
            or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    try:
        actors = _actors(engine)
        if actors is None or not _queue_valid(engine, tuple(engine.pending_effects), actors):
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        state = (engine.track, engine._state_key(include_scores=False, include_pending=False), roster,
                 tuple(engine.turns), engine.race_index, tuple(engine.turn_order),
                 tuple(engine.remaining_abilities), tuple(engine.egg_drawn), tuple(engine.previous_winners),
                 engine._preparing, engine._turn_in_progress, engine._resolution_depth,
                 tuple(sorted(getattr(engine, "_duel_offered_epochs", {}).items())),
                 json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                             "decisions": [record.event_sequence for record in engine.decisions]},
                            sort_keys=True, separators=(",", ":")),
                 engine._score_start, engine.initial_code_fingerprint, engine.max_events,
                 tuple(engine.initial_remaining_abilities))
        hash(state)
        return state
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    original_pending: tuple[PendingEffect, ...]
    score_before: ScoreSnapshot
    arrival_counter: int
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        raise ValueError("not a certified Romantic/Scoocher anchor FIFO cut")
    return Snapshot(actual, len(engine.events), tuple(engine.pending_effects), engine.score_ledger.snapshot(),
                    engine._arrival_counter, tuple(engine.events), tuple(engine.decisions))


class _Reject(Exception):
    pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.s, self.d, self.r, self.l = (actor.racer_id for actor in actors)
        self.phase = actors[0].position
        self.events, self.offset, self.batch = engine.events[snapshot.event_start:], 0, snapshot.arrival_counter
        self.balances = dict(snapshot.score_before.totals)
        self.stars, self.executed, self.phases = [], [], Counter()
        self.r_jobs = 0

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def move(self, actor, before, after, parent, *, ability=None, source=None, kind="ability"):
        self.batch += 1
        return self.take("move", actor, source_id=source, ability=ability, before=before, after=after,
                         amount=after - before, movement_kind=kind, parent_sequence=parent,
                         arrival_batch_id=self.batch)

    def stop(self, actor, before, after, parent):
        return self.take("stop", actor, before=before, after=after, parent_sequence=parent,
                         arrival_batch_id=self.batch)

    def r_child(self, stop):
        event = self.take("effect_queued", self.r, source_id=self.r, target_id=self.r,
                          ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
        return PendingEffect(self.r, self.r, stop.sequence, event.sequence, "move", 2, Ability.ROMANTIC.value, self.r)

    def job(self, pending):
        actor = pending.racer_id
        power = self.take("ability_effect", actor, source_id=actor,
                          ability=Ability.ROMANTIC.value if actor == self.r else Ability.SCOOCHER.value,
                          amount=2 if actor == self.r else 1, parent_sequence=pending.queued_sequence)
        children = []
        if actor == self.r:
            q = self.take("effect_queued", self.s, source_id=self.r, ability=Ability.SCOOCHER.value,
                          parent_sequence=power.sequence)
            children.append(PendingEffect(self.s, self.r, power.sequence, q.sequence))
            move = self.move(actor, 22, 24, power.sequence, ability=Ability.ROMANTIC.value, source=actor)
            stop = self.stop(actor, 22, 24, move.sequence)
            back = self.move(actor, 24, 22, stop.sequence, kind="track")
            children.append(self.r_child(self.stop(actor, 24, 22, back.sequence)))
            self.r_jobs += 1
        elif actor == self.s:
            phase = self.phase
            move = self.move(actor, phase, phase + 1, power.sequence, ability=Ability.SCOOCHER.value, source=actor)
            stop = self.stop(actor, phase, phase + 1, move.sequence)
            if phase == 12:
                total = self.balances[actor]
                self.stars.append(self.take("score", actor, before=total, after=total + 1, amount=1,
                                             reason="star", parent_sequence=stop.sequence))
                self.balances[actor] += 1
            elif phase == 14:
                children.append(self.r_child(stop))
            elif phase == 15:
                back = self.move(actor, 16, 12, stop.sequence, kind="track")
                self.take("passing", actor, target_id=self.l, parent_sequence=back.sequence,
                          arrival_batch_id=self.batch)
                self.stop(actor, 16, 12, back.sequence)
            self.phases[phase] += 1
            self.phase = 12 if phase == 15 else phase + 1
        else:
            raise _Reject
        self.executed.append(pending.queued_sequence)
        return children


def completed(engine, snapshot):
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions) != snapshot.decision_prefix):
        return False
    try:
        actors = _actors(engine)
        if not _queue_valid(engine, snapshot.original_pending, actors, require_unconsumed=False):
            return False
        old, current = snapshot.score_before, engine.score_ledger.snapshot()
        if current.player_ids != old.player_ids or current.entries[:len(old.entries)] != old.entries:
            return False
        reader, fifo = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
        if not reader.events or any(event.sequence != snapshot.event_start + i + 1
                                    for i, event in enumerate(reader.events)):
            return False
        while reader.offset < len(reader.events):
            if not fifo:
                return False
            fifo.extend(reader.job(fifo.popleft()))
        original = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        if (reader.batch != engine._arrival_counter or not reader.r_jobs
                or set(reader.phases) != {12, 13, 14, 15}
                or len(set(reader.phases.values())) != 1
                or [identity for identity in reader.executed if identity in original] != original
                or len(reader.executed) != len(set(reader.executed)) or not fifo
                or set(original).intersection(fresh) or any(identity <= snapshot.event_start for identity in fresh)
                or tuple(fifo) != tuple(engine.pending_effects)
                or not any(job.racer_id == actors[2].racer_id for job in fifo)
                or reader.balances != engine.score_ledger.totals()
                or current.totals != tuple((rid, reader.balances[rid]) for rid, _ in old.totals)):
            return False
        added = current.entries[len(old.entries):]
        if (len(added) != len(reader.stars)
                or any((entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source,
                        entry.reason, entry.event_sequence, entry.phase) != (
                            len(old.entries) + i + 1, event.racer_id, engine.race_index, 1,
                            "track", "star", event.sequence, "during_race")
                       for i, (entry, event) in enumerate(zip(added, reader.stars)))):
            return False
        races = {race: dict(totals) for race, totals in old.race_totals}
        for entry in added:
            races[entry.race_index][entry.player_id] += 1
        if current.race_totals != tuple((race, tuple((rid, races[race][rid]) for rid, _ in totals))
                                       for race, totals in old.race_totals):
            return False
        return {"original_pending_ids": original, "executed_pending_ids": reader.executed,
                "new_pending_ids": fresh, "queue_before": len(original), "queue_after": len(fresh),
                "mechanical_state_returned": True, "all_original_jobs_executed": True,
                "only_fresh_pending_descendants_stopped": True,
                "star_point_increments": {str(rid): reader.balances[rid] - total for rid, total in old.totals},
                "score_event_ids": [event.sequence for event in reader.stars],
                "scoocher_phase_jobs": {str(phase): reader.phases[phase] for phase in sorted(reader.phases)},
                "romantic_jobs": reader.r_jobs, "control_domain": "native-S12..15-D22-R22-L15",
                "proof": "All original FIFO jobs ran; complete mechanical phase and caller returned. R->SR preserves the positive R count; S14->R increases it and all other S have no child. Only fresh qualified descendants remain. Actual Star entries are retained."}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False
