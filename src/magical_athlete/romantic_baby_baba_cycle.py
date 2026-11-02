"""A no-input R/S/Baby/Baba FIFO with genuine silent Trip consumption.

Wild R12/14, S4, Baby5 and Baba4 form a closed typed FIFO. A real S job
returns4->5->Baby4 and replaces itself by S plus T and R (in actual CW
order); R replaces itself with S while toggling12/14; an already-tripped
S's Baba T really executes without events and has no child. Therefore S
count never decreases. T-only words are finite and never qualify here.
The original full cohort, including every zero-event T, must truly finish,
full mechanics/ledger must return, and only fresh descendants may stop.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
import json

from .decisions import DecisionWindow
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .static_inheritance import static_inheritance_evidence
from .track import WILD_WILDS

REASON = 'romantic_baby_baba_typed_fifo_cycle_proved'


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _actors(engine):
    actors = tuple(engine.racers.values())
    expected = (Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.BABA_YAGA)
    if (len(actors) != 4 or any(not isinstance(a.ability, Ability) for a in actors)
            or len({a.ability for a in actors}) != 4):
        return None
    # Keep the native path and its tuple/operation shape unchanged.
    if {a.ability for a in actors} == set(expected):
        by_power = {a.ability: a for a in actors}
        return tuple(by_power[power] for power in expected)
    principals = []
    for power in expected:
        matches = [actor for actor in actors if engine._ability(actor) == power]
        if len(matches) != 1 or static_inheritance_evidence(engine, matches[0], power) is None:
            return None
        principals.append(matches[0])
    return tuple(principals) if len({a.racer_id for a in principals}) == 4 else None


def _static_bindings(engine, actors):
    """Entire real preparation is bound, with no new decision in the suffix.

    Only original Egg/Twin carriers qualify. A borrowed Copy power never does.
    Every original/current qualification must follow the final actual grant;
    inheritance's own H10 notifications remain an executed finite prefix.
    """
    expected = (Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.BABA_YAGA)
    result = []
    for actor, power in zip(actors, expected):
        if actor.ability == power:
            if actor.inherited_ability is not None:
                return None
            continue
        evidence = static_inheritance_evidence(engine, actor, power)
        if not evidence:
            return None
        binding = json.loads(evidence[0])
        if (binding['choices'] != json.loads(json.dumps([asdict(record) for record in engine.decisions]))
                or any(record.request.window != DecisionWindow.BEFORE_RACE for record in engine.decisions)
                or binding['inherit_events'] != [asdict(event) for event in engine.events if event.kind == 'inherit']):
            return None
        result.append((actor.racer_id, actor.ability.value, power.value, evidence))
    return tuple(result)


def _grant_boundary(engine, bindings):
    return max((event.sequence for event in engine.events if event.kind == 'inherit'), default=0) if bindings else 0


def _queue_valid(engine, pending, actors, *, require_unconsumed=True):
    """Authenticate actual original qualification, including old first Trips.

    A real Baba first-Trip of R/S can notify S even though every T in the
    future closed suffix is silent. H03 keeps the stored target after it
    leaves the qualifying stop. This history is never replaced with flags.
    """
    r, s, b, y = (a.racer_id for a in actors)
    bindings = _static_bindings(engine, actors)
    if bindings is None: return False
    boundary = _grant_boundary(engine, bindings)
    if not pending or any(not isinstance(job, PendingEffect) for job in pending):
        return False
    ids = [job.queued_sequence for job in pending]
    if ids != sorted(set(ids)):
        return False
    initial = {a['racer_id']: a for a in engine.initial_racers}
    if set(initial) != set(engine.racers):
        return False
    positions = {rid: item['position'] for rid, item in initial.items()}
    tripped = {rid: item['tripped'] for rid, item in initial.items()}
    needed = {job.trigger_sequence for job in pending}
    past, batches, consumed = {}, Counter(), set()
    for event in engine.events:
        before_flags = dict(tripped)
        if event.kind in {'move', 'warp', 'stop'}:
            positions[event.racer_id] = event.after
        elif event.kind == 'trip':
            if tripped[event.racer_id] is not False:
                return False
            tripped[event.racer_id] = True
        elif event.kind == 'recover':
            tripped[event.racer_id] = False
        elif event.kind == 'inherit':
            if not bindings: return False
        elif event.kind in {'copy_changed', 'finish', 'eliminate', 'race_end'}:
            return False
        if event.kind in {'ability_effect', 'trip', 'effect_cancelled'}:
            consumed.add(event.parent_sequence)
        if event.kind == 'stop':
            batches[event.arrival_batch_id] += 1
        if event.sequence in needed:
            past[event.sequence] = dict(positions), before_flags
    if (positions != {rid: a.position for rid, a in engine.racers.items()}
            or tripped != {rid: a.tripped for rid, a in engine.racers.items()}):
        return False

    def stop_valid(event):
        return (event.kind == 'stop' and event.racer_id in engine.racers
                and type(event.before) is int and type(event.after) is int
                and boundary < event.sequence and 0 <= event.after < 30 and type(event.parent_sequence) is int
                and 1 <= event.parent_sequence < event.sequence
                and type(event.arrival_batch_id) is int and event.arrival_batch_id > 0
                and batches[event.arrival_batch_id] == 1
                and _event(event, 'stop', event.racer_id, before=event.before, after=event.after,
                           parent_sequence=event.parent_sequence, arrival_batch_id=event.arrival_batch_id))

    def historical_baba_job(queued, target):
        if (queued.kind != 'effect_queued' or queued.racer_id != y
                or not _event(queued, 'effect_queued', y, source_id=y, target_id=target,
                              ability=Ability.BABA_YAGA.value, parent_sequence=queued.parent_sequence)
                or type(queued.parent_sequence) is not int or not 1 <= queued.parent_sequence < queued.sequence):
            return False
        stop = engine.events[queued.parent_sequence - 1]
        old = {rid: item['position'] for rid, item in initial.items()}
        for event in engine.events[:stop.sequence]:
            if event.kind in {'move', 'warp', 'stop'}: old[event.racer_id] = event.after
        return (stop_valid(stop) and target in {r, s} and stop.after == old[y] == old[target]
                and (stop.racer_id == y or stop.racer_id == target))

    def historical_romantic_job(queued):
        if (not _event(queued, 'effect_queued', r, source_id=r, target_id=r,
                       ability=Ability.ROMANTIC.value, parent_sequence=queued.parent_sequence)
                or type(queued.parent_sequence) is not int or not 1 <= queued.parent_sequence < queued.sequence):
            return False
        stop = engine.events[queued.parent_sequence - 1]
        old = {rid: item['position'] for rid, item in initial.items()}
        for event in engine.events[:stop.sequence]:
            if event.kind in {'move', 'warp', 'stop'}: old[event.racer_id] = event.after
        return stop_valid(stop) and sum(pos == stop.after for pos in old.values()) == 2

    for job in pending:
        if (any(type(v) is not int for v in (job.racer_id, job.source_id, job.trigger_sequence,
                                             job.queued_sequence, job.steps))
                or job.target_id is not None and type(job.target_id) is not int
                or not boundary < job.trigger_sequence < job.queued_sequence <= len(engine.events)
                or require_unconsumed and job.queued_sequence in consumed):
            return False
        trigger, queued = (engine.events[i - 1] for i in (job.trigger_sequence, job.queued_sequence))
        history, old_flags = past[job.trigger_sequence]
        if job.kind == 'trip':
            if ((job.racer_id, job.source_id, job.target_id, job.steps, job.ability) != (y, y, s, 1, Ability.BABA_YAGA.value)
                    or not historical_baba_job(queued, s) or queued.parent_sequence != trigger.sequence):
                return False
        elif job.racer_id == r:
            if ((job.source_id, job.target_id, job.kind, job.steps, job.ability) != (r, r, 'move', 2, Ability.ROMANTIC.value)
                    or not stop_valid(trigger) or sum(p == trigger.after for p in history.values()) != 2
                    or not _event(queued, 'effect_queued', r, source_id=r, target_id=r,
                                  ability=Ability.ROMANTIC.value, parent_sequence=trigger.sequence)):
                return False
        elif job.racer_id == s:
            if ((job.source_id not in {r, b, y}) or (job.target_id, job.kind, job.steps, job.ability) !=
                    (None, 'move', 1, Ability.SCOOCHER.value)
                    or not _event(queued, 'effect_queued', s, source_id=job.source_id,
                                  ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)):
                return False
            if job.source_id == r:
                if not _event(trigger, 'ability_effect', r, source_id=r, ability=Ability.ROMANTIC.value,
                              amount=2, parent_sequence=trigger.parent_sequence):
                    return False
                if (type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                        or not historical_romantic_job(engine.events[trigger.parent_sequence - 1])):
                    return False
            elif job.source_id == b:
                if (type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1 or history[b] == 0
                        or not any(rid != b and pos == history[b] for rid, pos in history.items())
                        or not _event(trigger, 'ability_effect', b, source_id=b, ability=Ability.HUGE_BABY.value,
                                      parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)):
                    return False
            else:
                target = trigger.racer_id
                if (target not in {r, s} or old_flags[target] is not False
                        or type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                        or not _event(trigger, 'trip', target, source_id=y, ability=Ability.BABA_YAGA.value,
                                      parent_sequence=trigger.parent_sequence)
                        or not historical_baba_job(engine.events[trigger.parent_sequence - 1], target)):
                    return False
            if (type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence):
                return False
        else:
            return False
    return True


def key(engine):
    if (engine.track != WILD_WILDS or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies
            or hasattr(engine, '_arrow_visits') or hasattr(engine, '_arrow_queue_visits')
            or getattr(engine, '_movement_depth', 0) != 0 or getattr(engine, '_duel_offered_epochs', {})
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    actors = _actors(engine)
    if actors is None: return None
    r, s, b, y = actors
    bindings = _static_bindings(engine, actors)
    if bindings is None or not bindings and engine.decisions: return None
    if ((r.position not in {12, 14}) or (s.position, b.position, y.position) != (4, 5, 4) or s.tripped is not True):
        return None
    try:
        initial = {item['racer_id']: item for item in engine.initial_racers}
        for actor in actors:
            if (actor.status != RacerStatus.ACTIVE or type(actor.racer_id) is not int or type(actor.position) is not int
                    or type(actor.tripped) is not bool or actor.pre_race_resolved is not True
                    or (actor.inherited_ability is not None and not bindings) or actor.copied_target is not None or actor.copied_ability is not None
                    or actor.prediction is not None or actor.finish_place is not None or actor.finish_places
                    or type(actor.turns_taken) is not int or actor.turns_taken < 0
                    or (initial[actor.racer_id]['ability'], initial[actor.racer_id]['name']) != (actor.ability, actor.name)):
                return None
        depth, physical = engine._resolution_depth, engine._physical_resolution_depth
        if type(depth) is not int or depth < 1 or type(physical) is not int or physical < 1: return None
        pending = tuple(engine.pending_effects)
        if not any(job.racer_id == s.racer_id and job.kind == 'move' for job in pending): return None
        if not _queue_valid(engine, pending, actors): return None
        roster = tuple((rid, tuple((item.name, getattr(actor, item.name)) for item in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        state = (engine.track, engine._state_key(include_pending=False), roster, engine.score_ledger.snapshot(),
                 tuple(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), depth, physical,
                 json.dumps({'racers': engine.initial_racers, 'scores': engine.initial_score_ledger,
                             'deck': engine.initial_remaining_abilities}, sort_keys=True),
                 engine._score_start, engine.initial_code_fingerprint, engine.max_events)
        if bindings: state += ('static-principals-v1', bindings, tuple(engine.decisions))
        hash(state); return state
    except (KeyError, ValueError, TypeError, IndexError, AttributeError):
        return None


@dataclass(slots=True)
class Snapshot:
    state: tuple
    event_start: int
    original_pending: tuple
    arrival_counter: int
    event_prefix: tuple
    jobs: list = field(default_factory=list)
    in_flight: object = None
    invalid: bool = False


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual: return None
    return Snapshot(actual, len(engine.events), tuple(engine.pending_effects), engine._arrival_counter, tuple(engine.events))


def job_started(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid: return
    if snapshot.in_flight is not None or not queue_before or queue_before[0] != pending:
        snapshot.invalid = True; return
    snapshot.in_flight = pending, queue_before, event_start


def job_finished(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid: return
    if snapshot.in_flight != (pending, queue_before, event_start):
        snapshot.invalid = True; return
    snapshot.jobs.append((pending, queue_before, event_start, len(engine.events)))
    snapshot.in_flight = None


class _Reject(Exception): pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot, self.actors = engine, snapshot, actors
        self.r, self.s, self.b, self.y = (a.racer_id for a in actors)
        self.positions = {a.racer_id: a.position for a in actors}
        self.events, self.offset, self.batch = engine.events[snapshot.event_start:], 0, snapshot.arrival_counter

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events): raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values): raise _Reject
        self.offset += 1; return event

    def job(self, pending):
        r, s, b, y = self.r, self.s, self.b, self.y
        if pending.kind == 'trip':
            if (pending.racer_id, pending.source_id, pending.target_id, pending.steps, pending.ability) != (
                    y, y, s, 1, Ability.BABA_YAGA.value): raise _Reject
            return ()  # The callback must still certify the actual call returned.
        actor = pending.racer_id
        power, amount = (Ability.ROMANTIC, 2) if actor == r else (Ability.SCOOCHER, 1)
        if actor not in {r, s} or (pending.kind, pending.steps, pending.ability) != ('move', amount, power.value): raise _Reject
        effect = self.take('ability_effect', actor, source_id=actor, ability=power.value,
                           amount=amount, parent_sequence=pending.queued_sequence)
        if actor == s:
            self.batch += 1
            self.take('move', s, source_id=s, ability=power.value, before=4, after=5, amount=1,
                      movement_kind='ability', parent_sequence=effect.sequence, arrival_batch_id=self.batch)
            baby = self.take('ability_effect', b, source_id=b, ability=Ability.HUGE_BABY.value,
                             parent_sequence=effect.sequence, arrival_batch_id=self.batch)
            notification = self.take('effect_queued', s, source_id=b, ability=Ability.SCOOCHER.value,
                                     parent_sequence=baby.sequence)
            self.batch += 1
            stop = self.take('stop', s, before=5, after=4, parent_sequence=baby.sequence, arrival_batch_id=self.batch)
            children = [PendingEffect(s, b, baby.sequence, notification.sequence)]
            for rid in self.engine._clockwise_order():
                if rid == y:
                    queued = self.take('effect_queued', y, source_id=y, target_id=s, ability=Ability.BABA_YAGA.value,
                                       parent_sequence=stop.sequence)
                    children.append(PendingEffect(y, y, stop.sequence, queued.sequence, 'trip', 1, Ability.BABA_YAGA.value, s))
                elif rid == r:
                    queued = self.take('effect_queued', r, source_id=r, target_id=r, ability=Ability.ROMANTIC.value,
                                       parent_sequence=stop.sequence)
                    children.append(PendingEffect(r, r, stop.sequence, queued.sequence, 'move', 2, Ability.ROMANTIC.value, r))
            return tuple(children)
        notification = self.take('effect_queued', s, source_id=r, ability=Ability.SCOOCHER.value, parent_sequence=effect.sequence)
        before = self.positions[r]; after = before + 2; self.batch += 1
        move = self.take('move', r, source_id=r, ability=Ability.ROMANTIC.value, before=before, after=after, amount=2,
                         movement_kind='ability', parent_sequence=effect.sequence, arrival_batch_id=self.batch)
        stop = self.take('stop', r, before=before, after=after, parent_sequence=move.sequence, arrival_batch_id=self.batch)
        if after == 16:
            self.batch += 1
            move = self.take('move', r, before=16, after=12, amount=-4, movement_kind='track',
                             parent_sequence=stop.sequence, arrival_batch_id=self.batch)
            self.take('stop', r, before=16, after=12, parent_sequence=move.sequence, arrival_batch_id=self.batch)
            after = 12
        self.positions[r] = after
        return (PendingEffect(s, r, effect.sequence, notification.sequence),)


def completed(engine, snapshot):
    if (snapshot.invalid or snapshot.in_flight is not None or not snapshot.jobs
            or key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix):
        return False
    actors = _actors(engine)
    try:
        if not _queue_valid(engine, snapshot.original_pending, actors, require_unconsumed=False): return False
        reader, queue = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
        labels = {actors[0].racer_id: 'R', actors[1].racer_id: 'S', actors[3].racer_id: 'T'}
        executed, counts = [], Counter({'S': 0, 'R': 0, 'T': 0})
        persistent = lambda q: sum(j.racer_id == reader.s and j.kind == 'move' for j in q)
        for pending, actual_before, start, end in snapshot.jobs:
            old_s = persistent(queue)
            if (not queue or old_s < 1 or actual_before != tuple(queue) or queue.popleft() != pending
                    or start != snapshot.event_start + reader.offset): return False
            children = reader.job(pending); queue.extend(children)
            if (end != snapshot.event_start + reader.offset or persistent(queue) != old_s + int(pending.racer_id == reader.r)):
                return False
            executed.append(pending.queued_sequence); counts[labels[pending.racer_id]] += 1
        original = [j.queued_sequence for j in snapshot.original_pending]; fresh = [j.queued_sequence for j in queue]
        if (reader.offset != len(reader.events) or reader.batch != engine._arrival_counter or tuple(queue) != tuple(engine.pending_effects)
                or reader.positions != {rid: a.position for rid, a in engine.racers.items()}
                or [qid for qid in executed if qid in original] != original or len(executed) != len(set(executed))
                or set(original).intersection(fresh) or any(qid <= snapshot.event_start for qid in fresh)
                or counts['S'] < 1 or counts['R'] < 2 or counts['R'] % 2 or persistent(queue) < 1):
            return False
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, _Reject): return False
    result = {'original_pending_ids': original, 'executed_pending_ids': executed, 'new_pending_ids': fresh,
            'queue_before': len(original), 'queue_after': len(fresh), 'job_counts': dict(counts),
            'silent_trip_jobs_actually_finished': counts['T'], 'S_before': persistent(snapshot.original_pending),
            'S_after': persistent(queue), 'S_count_never_decreased': True,
            'mechanical_state_returned': True, 'scores_unchanged': True, 'all_original_jobs_executed': True,
            'only_fresh_pending_descendants_stopped': True,
            'replacement_rules': {'S': 'S plus real clockwise T,R', 'R': 'S', 'T': 'epsilon (already tripped)'},
            'proof': 'nondecreasing positive S count; full actual old FIFO including every silent T completed; full M and ledger returned'}

    bindings = _static_bindings(engine, actors)
    if bindings:
        result['static_principals'] = [
            {'racer_id': rid, 'original_ability': original, 'effective_ability': effective,
             'inherit_event_sequences': [event.sequence for event in engine.events
                                         if event.kind == 'inherit' and event.racer_id == rid]}
            for rid, original, effective, _ in bindings]
    return result
