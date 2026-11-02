"""Native R/D22 and S12..15, with original Baba already first at Finish.

R -> S,D,R; D KEEP -> empty; every real Duel/dice branch -> S,D,R.
S advances its four real phases and scores the printed Star at13. Thus an
existing R token cannot disappear. A D-only queue is not this proof domain.
Capture follows the earliest real KEEP at the clean initial phase. All old
jobs and the first safe complete S phase return must execute; a real KEEP allows
the existing suspended Duel generators to return without changing an epoch.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from functools import wraps
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import WILD_WILDS
from .romantic_baby_shared12_cycle import _ENGINE_FIELDS, _EVENT_FIELDS, _RACER_FIELDS, _event, _known_pending, _ledger_valid
from .scoocher_baby_duel22_cycle import Frame, _epoch, _past_digest, _ACTUAL
from .fish_baby_arrow11_cycle import _typed_request, _Reject as _BadRequest

REASON = "native_duel_romantic_scoocher22_star_fifo_cycle_proved"
ATTRIBUTE = "_duel_romantic_scoocher22_scope"
_ROLES = (Ability.ROMANTIC, Ability.DUELIST, Ability.SCOOCHER, Ability.BABA_YAGA)


class _Reject(Exception):
    pass


def _need(condition):
    if not condition:
        raise _Reject


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _actors(engine):
    _need(type(engine.racers) is dict and len(engine.racers) == 4)
    by_power = {r.ability: r for r in engine.racers.values()}
    _need(set(by_power) == set(_ROLES))
    return tuple(by_power[power] for power in _ROLES)


def _base(engine, actors):
    r, d, s, b = actors
    _need(not (set(vars(engine)) - (_ENGINE_FIELDS | {ATTRIBUTE})))
    _need(engine.track == WILD_WILDS and engine.termination_reason is None and engine.finish_order == [b.racer_id]
          and type(engine.finish_order) is list and type(engine.pending_effects) is deque
          and engine._preparing is False and engine._turn_in_progress is True
          and type(engine._copy_locks) is set and not engine._copy_locks and engine._refreshing_copies is False
          and getattr(engine, "_inert_duel_romantic", None) is None and engine.next_turn_racer_id is None
          and engine._genius_prediction is None and type(engine.score_ledger) is ScoreLedger)
    _need(type(engine.policies) is dict and type(engine.policy_metadata) is dict
          and set(engine.policies) == set(engine.racers))
    _need(all(type(getattr(engine, name)) is list for name in
              ("turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn")))
    _need(type(engine.initial_remaining_abilities) is tuple and type(engine.previous_winners) is tuple)
    _need(type(engine.turn_index) is int and 0 <= engine.turn_index < 4
          and len(engine.turn_order) == 4 and set(engine.turn_order) == set(engine.racers)
          and all(type(rid) is int for rid in engine.turn_order)
          and engine.turn_order[engine.turn_index] in {r.racer_id, d.racer_id})
    _need(engine.seed is None or type(engine.seed) is int)
    _need(type(engine.race_index) is int and 1 <= engine.race_index <= 4
          and type(engine.max_events) is int and engine.max_events > 0
          and type(engine._score_start) is int and engine._score_start >= 0)
    _need(not hasattr(engine, "max_action_events") or type(engine.max_action_events) is int and engine.max_action_events > 0)
    _need(all(type(getattr(engine, name, None)) is int and getattr(engine, name) > 0
              for name in ("_resolution_depth", "_physical_resolution_depth"))
          and type(getattr(engine, "_arrival_counter", None)) is int and engine._arrival_counter > 0
          and type(getattr(engine, "_movement_depth", 0)) is int and getattr(engine, "_movement_depth", 0) == 0)
    offered = getattr(engine, "_duel_offered_epochs", {})
    _need(type(offered) is dict and set(offered) <= {d.racer_id}
          and all(type(key) is int and type(value) is tuple and len(value) == 2
                  and all(type(n) is int and n >= 0 for n in value) for key, value in offered.items()))
    for actor in actors:
        _need(type(actor) is Racer and set(vars(actor)) == _RACER_FIELDS
              and type(actor.racer_id) is int and engine.racers.get(actor.racer_id) is actor
              and type(actor.ability) is Ability and type(actor.status) is RacerStatus
              and actor.pre_race_resolved is True and type(actor.position) is int and type(actor.tripped) is bool
              and type(actor.turns_taken) is int and actor.turns_taken >= 0
              and actor.inherited_ability is None and actor.copied_target is None and actor.copied_ability is None
              and actor.prediction is None and type(actor.finish_places) is tuple)
    _need(all(actor.status == RacerStatus.ACTIVE and actor.finish_place is None and not actor.finish_places for actor in (r, d, s))
          and (r.position, d.position) == (22, 22) and 12 <= s.position <= 15
          and b.status == RacerStatus.FINISHED and b.position == 30 and b.finish_place == 1 and b.finish_places == (1,))
    _need(all(type(event) is RaceEvent and type(event.sequence) is int and event.sequence == i + 1
              and set(vars(event)) == _EVENT_FIELDS for i, event in enumerate(engine.events)))
    _need([event.decision_id for event in engine.events if event.kind == "decision"] == list(range(1, len(engine.decisions) + 1))
          and all(_known_pending(job) for job in engine.pending_effects))


def _mechanics(engine):
    # All score fields are checked separately by a real append-only ledger
    # reconstruction. Only the authenticated printed-Star increments vary.
    return _json({"roster": [{**asdict(r), "status": r.status.value} for _, r in sorted(engine.racers.items())],
        "turns": [asdict(t) for t in engine.turns], "order": engine.turn_order, "index": engine.turn_index,
        "race": engine.race_index, "remaining": engine.remaining_abilities, "initial_remaining": engine.initial_remaining_abilities,
        "egg": engine.egg_drawn, "winners": engine.previous_winners, "finish": engine.finish_order,
        "initial_racers": engine.initial_racers, "initial_scores": engine.initial_score_ledger,
        "metadata": engine.policy_metadata, "policy_seats": sorted(engine.policies), "score_start": engine._score_start,
        "fingerprint": engine.initial_code_fingerprint, "seed": engine.seed, "budget": engine.max_events,
        "action_budget": getattr(engine, "max_action_events", None), "physical_depth": engine._physical_resolution_depth,
        "preparing": engine._preparing, "turn_in_progress": engine._turn_in_progress,
        "next_turn": engine.next_turn_racer_id, "prediction": engine._genius_prediction})


@dataclass(frozen=True, slots=True)
class Snapshot:
    event_start: int
    decision_start: int
    original_pending: tuple
    prefix_sha256: str
    mechanics: str
    scores_json: str
    phase: int
    arrival: int
    raw_epoch: tuple
    owner_ids: tuple
    owner_depth: int


@dataclass(slots=True)
class Scope:
    root_end: int
    root_pending: tuple
    root_arrival: int
    root_positions: dict
    root_scores_json: str
    root_prefix_sha256: str
    root_decisions: int
    drains: list = field(default_factory=list)
    duels: list = field(default_factory=list)
    snapshot: Snapshot | None = None
    closed: bool = False
    certificate_event: int | None = None
    expected_returns: list = field(default_factory=list)
    actual_returns: list = field(default_factory=list)


def _root(engine, actors, end):
    """Real native main arrival22, or main24 followed by Arrow24->22."""
    r, d, _s, _b = (actor.racer_id for actor in actors)
    _need(type(end) is int and 7 <= end <= len(engine.events))
    last_stop = engine.events[end - 3]
    _need(last_stop.kind == "stop" and type(last_stop.parent_sequence) is int
          and 1 <= last_stop.parent_sequence <= len(engine.events))
    returned = engine.events[last_stop.parent_sequence - 1].movement_kind == "track"
    length = 9 if returned else 7
    _need(end >= length)
    word = engine.events[end - length:end]
    roll, lock, resolved, move, stop = word[:5]
    q1, q2 = word[-2:]
    actor, n = roll.racer_id, roll.amount
    destination = 24 if returned else 22
    _need(actor == engine.turn_order[engine.turn_index] and actor in {r, d}
          and type(n) is int and 1 <= n <= 6
          and _event(roll, "roll", actor, amount=n)
          and _event(lock, "roll_locked", actor, amount=n, parent_sequence=roll.sequence)
          and _event(resolved, "main_move_resolved", actor, before=n, after=n, parent_sequence=lock.sequence)
          and type(move.arrival_batch_id) is int and move.arrival_batch_id > 0
          and _event(move, "move", actor, before=destination - n, after=destination, amount=n, movement_kind="main",
                     parent_sequence=resolved.sequence, arrival_batch_id=move.arrival_batch_id)
          and _event(stop, "stop", actor, before=destination - n, after=destination,
                     parent_sequence=move.sequence, arrival_batch_id=move.arrival_batch_id))
    if returned:
        arrow, back = word[5:7]
        _need(_event(arrow, "move", actor, before=24, after=22, amount=-2, movement_kind="track",
                     parent_sequence=stop.sequence, arrival_batch_id=move.arrival_batch_id + 1)
              and _event(back, "stop", actor, before=24, after=22, parent_sequence=arrow.sequence,
                         arrival_batch_id=move.arrival_batch_id + 1))
        stop = back
    pending = []
    for event, target in zip((q1, q2), (rid for rid in engine._clockwise_order() if rid in {r, d})):
        ability = "Romantic" if target == r else "Duelist"
        _need(_event(event, "effect_queued", target, source_id=target, target_id=target,
                     ability=ability, parent_sequence=stop.sequence))
        pending.append(PendingEffect(target, target, stop.sequence, event.sequence,
            "move" if target == r else "duel", 2 if target == r else 0, ability, target))
    return tuple(pending), stop.arrival_batch_id


class _Reader:
    def __init__(self, engine, actors, start, batch, positions, scores_json):
        self.engine, self.actors, self.offset, self.batch = engine, actors, start, batch
        self.r, self.d, self.s, self.b = (actor.racer_id for actor in actors)
        self.positions = dict(positions)
        self.ledger = ScoreLedger.from_trace(json.loads(scores_json))
        self.executed, self.counts, self.keeps = [], Counter(), []
        self.duel_starts, self.duel_yields, self.star_events = {}, {}, []

    def take(self, kind, actor, **values):
        _need(self.offset < len(self.engine.events))
        event = self.engine.events[self.offset]
        _need(event.sequence == self.offset + 1 and _event(event, kind, actor, **values))
        self.offset += 1
        if kind in {"move", "stop"}:
            self.positions[actor] = event.after
        return event

    def move(self, actor, before, after, amount, parent, *, ability=None, source=None, kind="ability"):
        _need(self.positions[actor] == before)
        self.batch += 1
        moved = self.take("move", actor, before=before, after=after, amount=amount,
            movement_kind=kind, source_id=source, ability=ability, parent_sequence=parent, arrival_batch_id=self.batch)
        return self.take("stop", actor, before=before, after=after, parent_sequence=moved.sequence, arrival_batch_id=self.batch)

    def children(self, stop):
        result = []
        for target in (rid for rid in self.engine._clockwise_order() if rid in {self.r, self.d}):
            ability = "Romantic" if target == self.r else "Duelist"
            queued = self.take("effect_queued", target, source_id=target, target_id=target,
                               ability=ability, parent_sequence=stop.sequence)
            result.append(PendingEffect(target, target, stop.sequence, queued.sequence,
                "move" if target == self.r else "duel", 2 if target == self.r else 0, ability, target))
        return tuple(result)

    def scooch(self, source, effect):
        queued = self.take("effect_queued", self.s, source_id=source, ability="Scoocher", parent_sequence=effect.sequence)
        return PendingEffect(self.s, source, effect.sequence, queued.sequence)

    def returned(self, actor, source, ability, effect):
        stop = self.move(actor, 22, 24, 2, effect.sequence, ability=ability, source=source)
        stop = self.move(actor, 24, 22, -2, stop.sequence, kind="track")
        return self.children(stop)

    def choice(self, parent):
        event = self.engine.events[self.offset]
        _need(type(event.decision_id) is int and 1 <= event.decision_id <= len(self.engine.decisions))
        record = self.engine.decisions[event.decision_id - 1]
        try:
            _typed_request(record)
        except _BadRequest:
            raise _Reject
        public = tuple(PublicRacer(actor.racer_id, actor.name, self.positions[actor.racer_id], actor.status.value,
            actor.ability.value, actor.tripped, actor.finish_place, None)
            for actor in (self.engine.racers[rid] for rid in self.engine.turn_order))
        observation = Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            self.engine.track.name, 30, 15, tuple(sorted(self.engine.track.trip_positions)), public,
            tuple(self.engine.finish_order), None, None, score_totals=tuple(self.ledger.totals().items()))
        legal = (Action(ActionKind.KEEP), Action(ActionKind.DUEL, self.d, self.r))
        _need(record.request == DecisionRequest(event.decision_id, self.d, DecisionWindow.DUEL, observation, legal)
              and record.action in legal and record.event_sequence == event.sequence)
        action = record.action
        self.take("decision", self.d, source_id=action.source_id, target_id=action.target_id,
            action_kind=action.kind.value, decision_id=event.decision_id, parent_sequence=parent)
        if action.kind == ActionKind.KEEP:
            self.keeps.append(event.sequence)
        return action

    def job(self, pending):
        _need(_known_pending(pending))
        self.executed.append(pending.queued_sequence)
        if pending.racer_id == self.s:
            _need((pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id)
                  in {(source, "move", 1, "Scoocher", None) for source in (self.r, self.d)})
            effect = self.take("ability_effect", self.s, source_id=self.s, ability="Scoocher", amount=1,
                               parent_sequence=pending.queued_sequence)
            before = self.positions[self.s]
            _need(12 <= before <= 15)
            stop = self.move(self.s, before, before + 1, 1, effect.sequence, ability="Scoocher", source=self.s)
            if before == 12:
                total = self.ledger.total(self.s)
                score = self.take("score", self.s, before=total, after=total + 1, amount=1,
                                  parent_sequence=stop.sequence, reason="star")
                self.ledger.record(self.s, self.engine.race_index, 1, source="track", reason="star",
                                   event_sequence=score.sequence, phase="during_race")
                self.star_events.append(score.sequence)
            elif before == 15:
                self.move(self.s, 16, 12, -4, stop.sequence, kind="track")
            self.counts["S"] += 1
            return ()
        if pending.racer_id == self.r:
            _need((pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id)
                  == (self.r, "move", 2, "Romantic", self.r))
            effect = self.take("ability_effect", self.r, source_id=self.r, ability="Romantic", amount=2,
                               parent_sequence=pending.queued_sequence)
            child = self.scooch(self.r, effect)
            result = self.returned(self.r, self.r, "Romantic", effect)
            self.counts["R"] += 1
            return (child, *result)
        _need((pending.racer_id, pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id)
              == (self.d, self.d, "duel", 0, "Duelist", self.d))
        identity = pending.queued_sequence
        self.duel_starts[identity] = self.offset
        action = self.choice(identity)
        if action.kind == ActionKind.KEEP:
            self.counts["D_KEEP"] += 1
            return ()
        effect = self.take("ability_effect", self.d, source_id=self.d, target_id=self.r, ability="Duelist",
                           parent_sequence=self.engine.events[self.duel_starts[identity]].sequence)
        child = self.scooch(self.d, effect)
        dice = []
        for actor in (self.d, self.r):
            event = self.engine.events[self.offset]
            _need(type(event.amount) is int and 1 <= event.amount <= 6)
            dice.append(event.amount)
            self.take("duel_roll", actor, source_id=self.d, ability="Duelist", amount=event.amount,
                      parent_sequence=effect.sequence)
        winner = self.d if dice[0] >= dice[1] else self.r
        result = self.returned(winner, self.d, "Duelist", effect)
        self.duel_yields[identity] = self.offset
        self.counts["D_DUEL"] += 1
        return (child, *result)


def _birth(engine, actors):
    _base(engine, actors)
    _need(engine._resolution_depth == 1 and len(engine.pending_effects) == 2)
    pending, arrival = _root(engine, actors, len(engine.events))
    _need(pending == tuple(engine.pending_effects) and arrival == engine._arrival_counter
          and _ledger_valid(engine, len(engine.events)))
    return Scope(len(engine.events), pending, arrival, {actor.racer_id: actor.position for actor in actors},
        _json(engine.score_ledger.trace()), _past_digest(engine.events, engine.decisions), len(engine.decisions))


def _authenticate(engine, scope):
    actors = _actors(engine)
    _base(engine, actors)
    _need(type(scope) is Scope and type(scope.closed) is bool and not scope.closed
          and type(scope.root_end) is int and scope.root_end >= 7
          and type(scope.root_decisions) is int and scope.root_decisions >= 0
          and type(scope.root_arrival) is int and scope.root_arrival > 0
          and type(scope.root_positions) is dict and set(scope.root_positions) == set(engine.racers)
          and all(type(key) is int and type(value) is int for key, value in scope.root_positions.items())
          and type(scope.root_scores_json) is str and type(scope.root_prefix_sha256) is str
          and type(scope.root_pending) is tuple and all(_known_pending(job) for job in scope.root_pending)
          and _past_digest(engine.events[:scope.root_end], engine.decisions[:scope.root_decisions]) == scope.root_prefix_sha256)
    pending, arrival = _root(engine, actors, scope.root_end)
    _need(pending == scope.root_pending and arrival == scope.root_arrival)
    reader = _Reader(engine, actors, scope.root_end, scope.root_arrival, scope.root_positions, scope.root_scores_json)
    queue = deque(scope.root_pending)
    cuts = {reader.offset: tuple(queue)}
    while reader.offset < len(engine.events):
        _need(queue)
        queue.extend(reader.job(queue.popleft()))
        cuts[reader.offset] = tuple(queue)
    _need(tuple(queue) == tuple(engine.pending_effects) and reader.offset == len(engine.events)
          and reader.positions == {actor.racer_id: actor.position for actor in actors}
          and reader.batch == engine._arrival_counter and _json(reader.ledger.trace()) == _json(engine.score_ledger.trace()))
    _need(scope.drains and len(scope.drains) == len(scope.duels) + 1
          and scope.drains[-1].depth == engine._resolution_depth)
    for frame in (*scope.drains, *scope.duels):
        _need(type(frame) is Frame and type(frame.depth) is int and type(frame.physical) is int
              and frame.physical == engine._physical_resolution_depth and type(frame.start) is int
              and type(frame.yielded) is list and all(type(value) is int for value in frame.yielded))
    for index, owner in enumerate(scope.duels):
        child = scope.drains[index + 1]
        _need(owner.source_id == actors[1].racer_id and type(owner.trigger) is int
              and owner.start == reader.duel_starts.get(owner.trigger)
              and owner.depth == scope.drains[index].depth + 1 and child.depth == owner.depth + 1
              and owner.yielded == [reader.duel_yields.get(owner.trigger)]
              and child.trigger == owner.trigger and child.start == owner.yielded[0])
    return actors, reader, cuts


def capture(engine, scope):
    try:
        actors, reader, _ = _authenticate(engine, scope)
        r, d, s, _b = actors
        _need(reader.keeps and reader.keeps[-1] == len(engine.events)
              and engine._duel_offered_epochs.get(d.racer_id) == _epoch(engine)
              and any(job.racer_id == r.racer_id for job in engine.pending_effects))
        return Snapshot(len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
            _past_digest(engine.events, engine.decisions), _mechanics(engine), _json(engine.score_ledger.trace()),
            s.position, engine._arrival_counter, tuple(sorted(engine._duel_offered_epochs.items())),
            tuple(owner.trigger for owner in scope.duels), engine._resolution_depth)
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
        return None


def completed(engine, snapshot):
    try:
        scope = getattr(engine, ATTRIBUTE)
        actors, reader, cuts = _authenticate(engine, scope)
        r, d, s, _b = actors
        _need(type(snapshot) is Snapshot and type(snapshot.original_pending) is tuple
              and all(type(value) is int for value in (snapshot.event_start, snapshot.decision_start,
                                                       snapshot.arrival, snapshot.phase, snapshot.owner_depth))
              and snapshot.event_start >= scope.root_end and snapshot.decision_start >= scope.root_decisions
              and snapshot.arrival >= scope.root_arrival and 12 <= snapshot.phase <= 15 and snapshot.owner_depth > 0
              and type(snapshot.owner_ids) is tuple and all(type(value) is int and value > 0 for value in snapshot.owner_ids)
              and type(snapshot.raw_epoch) is tuple and len(snapshot.raw_epoch) == 1
              and type(snapshot.raw_epoch[0]) is tuple and len(snapshot.raw_epoch[0]) == 2
              and type(snapshot.raw_epoch[0][0]) is int and snapshot.raw_epoch[0][0] == d.racer_id
              and type(snapshot.raw_epoch[0][1]) is tuple and len(snapshot.raw_epoch[0][1]) == 2
              and all(type(value) is int for value in snapshot.raw_epoch[0][1])
              and all(_known_pending(job) for job in snapshot.original_pending)
              and len(engine.events) > snapshot.event_start and cuts.get(snapshot.event_start) == snapshot.original_pending
              and _past_digest(engine.events[:snapshot.event_start], engine.decisions[:snapshot.decision_start]) == snapshot.prefix_sha256
              and _mechanics(engine) == snapshot.mechanics and s.position == snapshot.phase
              and reader.keeps and reader.keeps[-1] == len(engine.events)
              and engine._duel_offered_epochs.get(d.racer_id) == _epoch(engine))
        capture_epoch = (len(engine.turns), next(event.sequence for event in reversed(engine.events[:snapshot.event_start])
                                                   if event.kind in _ACTUAL))
        _need(snapshot.raw_epoch == ((d.racer_id, capture_epoch),))
        parsed = _Reader(engine, actors, snapshot.event_start, snapshot.arrival,
            {actor.racer_id: actor.position for actor in actors}, snapshot.scores_json)
        queue = deque(snapshot.original_pending)
        originals = [job.queued_sequence for job in snapshot.original_pending]
        first_eligible = None
        while parsed.offset < len(engine.events):
            _need(queue)
            queue.extend(parsed.job(queue.popleft()))
            remaining_ids = {job.queued_sequence for job in queue}
            if (first_eligible is None and parsed.counts["S"] >= 4 and parsed.counts["S"] % 4 == 0
                    and parsed.positions[s.racer_id] == snapshot.phase
                    and [identity for identity in parsed.executed if identity in originals] == originals
                    and not set(originals).intersection(remaining_ids)
                    and parsed.keeps and parsed.keeps[-1] == parsed.offset
                    and any(job.racer_id == r.racer_id for job in queue)):
                first_eligible = (parsed.offset, parsed.counts["S"] // 4)
        fresh = [job.queued_sequence for job in queue]
        _need(parsed.counts["S"] >= 4 and parsed.counts["S"] % 4 == 0
              and len(parsed.star_events) == parsed.counts["S"] // 4
              and first_eligible == (len(engine.events), parsed.counts["S"] // 4)
              and [identity for identity in parsed.executed if identity in originals] == originals
              and not set(originals).intersection(fresh) and fresh and min(fresh) > snapshot.event_start
              and any(job.racer_id == r.racer_id for job in queue)
              and tuple(queue) == tuple(engine.pending_effects) and parsed.batch == engine._arrival_counter
              and _json(parsed.ledger.trace()) == _json(engine.score_ledger.trace()))
        owners = tuple(owner.trigger for owner in scope.duels)
        extras = owners[len(snapshot.owner_ids):]
        _need(owners[:len(snapshot.owner_ids)] == snapshot.owner_ids
              and engine._resolution_depth == snapshot.owner_depth + 2 * len(extras)
              and all(identity in parsed.executed and identity in parsed.duel_yields for identity in extras))
        return {"capture_event_count": snapshot.event_start, "capture_phase": snapshot.phase,
            "original_pending_ids": originals, "executed_pending_ids": parsed.executed, "new_pending_ids": fresh,
            "root_pending_ids": [job.queued_sequence for job in scope.root_pending],
            "root_executed_pending_ids": reader.executed, "typed_job_counts": dict(parsed.counts),
            "scoocher_complete_orbits": parsed.counts["S"] // 4, "retained_actual_Star_events": parsed.star_events,
            "required_original_S_jobs": sum(job.racer_id == s.racer_id for job in snapshot.original_pending),
            "mandatory_original_cohort_size": len(originals),
            "required_first_safe_complete_orbits_given_actual_choices": first_eligible[1],
            "actual_complete_orbits": parsed.counts["S"] // 4,
            "first_safe_cut_event": first_eligible[0],
            "score_increment": {str(actor.racer_id): parsed.ledger.total(actor.racer_id) - ScoreLedger.from_trace(json.loads(snapshot.scores_json)).total(actor.racer_id) for actor in actors},
            "first_complete_four_S_phase_return": parsed.counts["S"] == 4,
            "first_eligible_all_originals_and_phase_and_KEEP_return": True, "no_canonical_phase_delay": True,
            "Duel_continuations_still_to_return": list(reversed(owners)),
            "suspended_Duel_body_evidence": [{"pending_id": identity,
                "complete_real_body_end": reader.duel_yields[identity]} for identity in owners],
            "raw_epoch_at_capture": dict(snapshot.raw_epoch), "raw_epoch_at_return": dict(engine._duel_offered_epochs),
            "epoch_mutated_by_certificate": False, "actual_KEEP_at_return": len(engine.events),
            "owner_resolution_depth": snapshot.owner_depth, "actual_resolution_depth": engine._resolution_depth,
            "physical_resolution_depth": engine._physical_resolution_depth,
            "arrival_counter_at_capture": snapshot.arrival, "arrival_counter_at_return": engine._arrival_counter,
            "all_choice_invariant": "R->S,D,R; D KEEP->empty; every36-dice DUEL->S,D,R; S->empty and printed Star13; existing R never disappears",
            "all_original_jobs_really_served": True, "current_main_already_executed": True,
            "current_turn_completion_still_owed": True}
    except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
        return None


def probe(engine):
    scope = getattr(engine, ATTRIBUTE, None)
    if type(scope) is not Scope or scope.closed:
        return
    if scope.snapshot is None:
        scope.snapshot = capture(engine, scope)
        return
    certificate = completed(engine, scope.snapshot)
    if certificate is None:
        return
    scope.expected_returns = list(certificate["Duel_continuations_still_to_return"])
    engine._emit("loop_stopped", engine.turn_order[engine.turn_index], reason=REASON,
        parent_sequence=scope.snapshot.event_start + 1, amount=len(engine.pending_effects), operation=_json(certificate))
    scope.certificate_event = len(engine.events)
    scope.closed = True
    engine.pending_effects.clear()


def continuation_returned(engine, scope, trigger):
    if scope.closed:
        _need(scope.expected_returns and scope.expected_returns.pop(0) == trigger
              and len(engine.events) == scope.certificate_event and not engine.pending_effects
              and engine._duel_offered_epochs.get(_actors(engine)[1].racer_id) == _epoch(engine))
        scope.actual_returns.append(trigger)


def completion(engine, scope):
    if scope.closed:
        _need(not scope.expected_returns and not scope.duels and not scope.drains
              and len(engine.events) == scope.certificate_event and not engine.pending_effects)


def drain_scope(method):
    @wraps(method)
    def scoped(engine, *args, **kwargs):
        scope = getattr(engine, ATTRIBUTE, None)
        owns = False
        if scope is None:
            try:
                scope = _birth(engine, _actors(engine))
            except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration, _Reject):
                yield from method(engine, *args, **kwargs)
                return
            setattr(engine, ATTRIBUTE, scope)
            owns = True
        if type(scope) is not Scope:
            yield from method(engine, *args, **kwargs)
            return
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
            if scope.drains and scope.drains[-1] is frame:
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
        if type(scope) is not Scope:
            yield from method(engine, source, trigger_sequence)
            return
        frame = Frame(trigger_sequence, len(engine.events), engine._resolution_depth, engine._physical_resolution_depth)
        frame.source_id = source.racer_id
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
            if scope.duels and scope.duels[-1] is frame:
                scope.duels.pop()
            if returned:
                continuation_returned(engine, scope, trigger_sequence)
    return scoped
