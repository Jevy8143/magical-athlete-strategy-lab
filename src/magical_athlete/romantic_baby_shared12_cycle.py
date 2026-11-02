"""Authenticate a forced, zero-score Romantic/Scoocher/Baby FIFO on Wild12.

Native S12/B13 always replace an S job by another S. Native R alternates
12/14 through Arrow16; each R job adds S, sometimes also another R. Native
Dicemonger10 has no roll window inside these bodies. Every replacement is
nonempty, independently of queue length. The real original FIFO and any
descendants needed for a full R12 return must execute before the existing
caller may stop only authenticated fresh descendants. This module neither
executes nor removes a task.
"""

from collections import Counter, deque
from dataclasses import dataclass, fields
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .scoring import ScoreLedger, ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS


REASON = "romantic_baby_shared12_fifo_cycle_proved"
_ROLES = (Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.DICEMONGER)
_ENGINE_FIELDS = frozenset({
    "track", "max_events", "_turn_in_progress", "_preparing", "initial_remaining_abilities",
    "remaining_abilities", "previous_winners", "egg_drawn", "_copy_locks", "_refreshing_copies",
    "race_index", "score_ledger", "initial_score_ledger", "_score_start", "racers", "turn_order",
    "dice", "seed", "policies", "policy_metadata", "decisions", "finish_order", "turns", "events",
    "pending_effects", "turn_index", "next_turn_racer_id", "termination_reason", "initial_racers",
    "initial_code_fingerprint", "_genius_prediction", "_resolution_depth", "_physical_resolution_depth",
    "_arrival_counter", "_movement_depth", "_inert_duel_romantic", "_duel_offered_epochs", "max_action_events",
})
_EVENT_FIELDS = frozenset(field.name for field in fields(RaceEvent))
_RACER_FIELDS = frozenset(field.name for field in fields(Racer))
_PENDING_FIELDS = frozenset(field.name for field in fields(PendingEffect))


class _Reject(Exception):
    pass


def _event(event, kind, actor, **values):
    numeric = ("sequence", "racer_id", "source_id", "target_id", "before", "after", "amount",
               "parent_sequence", "decision_id", "arrival_batch_id")
    return (type(event) is RaceEvent and set(vars(event)) == _EVENT_FIELDS
            and all(getattr(event, name) is None or type(getattr(event, name)) is int for name in numeric)
            and event == RaceEvent(event.sequence, kind, actor, **values))


def _actors(engine):
    if len(engine.racers) != 4:
        raise _Reject
    by_power = {actor.ability: actor for actor in engine.racers.values()}
    if set(by_power) != set(_ROLES):
        raise _Reject
    return tuple(by_power[power] for power in _ROLES)


def _known_pending(pending):
    return (type(pending) is PendingEffect and set(vars(pending)) == _PENDING_FIELDS
            and all(type(value) is int for value in (pending.racer_id, pending.source_id,
                pending.trigger_sequence, pending.queued_sequence, pending.steps))
            and (pending.target_id is None or type(pending.target_id) is int))


def _ledger_valid(engine, root_start):
    """Authenticate the initial account plus every actual pre-root Star entry."""
    old, current = ScoreLedger.from_trace(engine.initial_score_ledger).snapshot(), engine.score_ledger.snapshot()
    if (engine._score_start != len(old.entries) or current.player_ids != old.player_ids
            or current.entries[:len(old.entries)] != old.entries):
        return False
    stars = [event for event in engine.events if event.kind == "score"]
    added = current.entries[len(old.entries):]
    if len(stars) != len(added) or any(event.sequence > root_start for event in stars):
        return False
    balances, races = dict(old.totals), {index: dict(totals) for index, totals in old.race_totals}
    for i, (event, entry) in enumerate(zip(stars, added), len(old.entries) + 1):
        if (event.racer_id not in balances or type(event.parent_sequence) is not int
                or not 1 <= event.parent_sequence < event.sequence
                or not _event(event, "score", event.racer_id, before=balances[event.racer_id],
                              after=balances[event.racer_id] + 1, amount=1,
                              parent_sequence=event.parent_sequence, reason="star")
                or (entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source,
                    entry.reason, entry.event_sequence, entry.phase) != (
                        i, event.racer_id, engine.race_index, 1, "track", "star", event.sequence, "during_race")):
            return False
        stop = engine.events[event.parent_sequence - 1]
        if stop.kind != "stop" or stop.racer_id != event.racer_id or stop.after not in engine.track.star_positions:
            return False
        balances[event.racer_id] += 1
        races[engine.race_index][event.racer_id] += 1
    return (current.totals == tuple((rid, balances[rid]) for rid, _ in old.totals)
            and current.race_totals == tuple((index, tuple((rid, races[index][rid]) for rid, _ in totals))
                                             for index, totals in old.race_totals))


class _Reader:
    def __init__(self, engine, actors, start, arrival_counter, phase):
        self.engine, self.actors = engine, actors
        self.r, self.s, self.b, self.d = (actor.racer_id for actor in actors)
        self.offset, self.batch, self.phase = start, arrival_counter, phase
        self.executed, self.counts = [], Counter()

    def take(self, kind, actor, **values):
        if self.offset >= len(self.engine.events):
            raise _Reject
        event = self.engine.events[self.offset]
        if event.sequence != self.offset + 1 or not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def queued_s(self, source, trigger):
        event = self.take("effect_queued", self.s, source_id=source, ability=Ability.SCOOCHER.value,
                          parent_sequence=trigger.sequence)
        return PendingEffect(self.s, source, trigger.sequence, event.sequence)

    def queued_r(self, stop):
        event = self.take("effect_queued", self.r, source_id=self.r, target_id=self.r,
                          ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
        return PendingEffect(self.r, self.r, stop.sequence, event.sequence, "move", 2, Ability.ROMANTIC.value, self.r)

    def job(self, pending):
        if not _known_pending(pending):
            raise _Reject
        actor = pending.racer_id
        if actor == self.s:
            if (pending.source_id not in {self.r, self.b}
                    or (pending.kind, pending.steps, pending.ability, pending.target_id) != ("move", 1, "Scoocher", None)):
                raise _Reject
            power = self.take("ability_effect", self.s, source_id=self.s, ability="Scoocher", amount=1,
                              parent_sequence=pending.queued_sequence)
            self.batch += 1
            self.take("move", self.s, source_id=self.s, ability="Scoocher", before=12, after=13, amount=1,
                      movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=self.batch)
            baby = self.take("ability_effect", self.b, source_id=self.b, ability="Huge Baby",
                             parent_sequence=power.sequence, arrival_batch_id=self.batch)
            children = [self.queued_s(self.b, baby)]
            self.batch += 1
            stop = self.take("stop", self.s, before=13, after=12, parent_sequence=baby.sequence, arrival_batch_id=self.batch)
            if self.phase == 12:
                children.append(self.queued_r(stop))
            self.counts[f"S_at_R{self.phase}"] += 1
        elif actor == self.r:
            if (pending.source_id, pending.kind, pending.steps, pending.ability, pending.target_id) != (
                    self.r, "move", 2, "Romantic", self.r):
                raise _Reject
            power = self.take("ability_effect", self.r, source_id=self.r, ability="Romantic", amount=2,
                              parent_sequence=pending.queued_sequence)
            children = [self.queued_s(self.r, power)]
            old = self.phase
            self.batch += 1
            move = self.take("move", self.r, source_id=self.r, ability="Romantic", before=old, after=old + 2, amount=2,
                             movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=self.batch)
            if old == 12:
                self.take("passing", self.r, target_id=self.b, parent_sequence=move.sequence, arrival_batch_id=self.batch)
            stop = self.take("stop", self.r, before=old, after=old + 2, parent_sequence=move.sequence, arrival_batch_id=self.batch)
            if old == 14:
                self.batch += 1
                back = self.take("move", self.r, before=16, after=12, amount=-4, movement_kind="track",
                                 parent_sequence=stop.sequence, arrival_batch_id=self.batch)
                self.take("passing", self.r, target_id=self.b, parent_sequence=back.sequence, arrival_batch_id=self.batch)
                returned = self.take("stop", self.r, before=16, after=12, parent_sequence=back.sequence, arrival_batch_id=self.batch)
                children.append(self.queued_r(returned))
            self.phase = 26 - old
            self.counts[f"R_at{old}"] += 1
        else:
            raise _Reject
        self.executed.append(pending.queued_sequence)
        return children


def _root(engine, actors, *, end_event_count=None):
    r, s, b, d = actors
    main = next(event for event in reversed(engine.events) if event.kind == "main_move_resolved")
    if main.sequence < 4 or main.racer_id != r.racer_id or main.before != 5 or main.after != 5:
        raise _Reject
    start = main.sequence - 4
    positions = {row["racer_id"]: row["position"] for row in engine.initial_racers}
    for event in engine.events[:start]:
        if event.kind in {"move", "warp", "stop"}:
            positions[event.racer_id] = event.after
    if positions != {r.racer_id: 5, s.racer_id: 12, b.racer_id: 13, d.racer_id: 10}:
        raise _Reject
    direct = engine.events[start + 4]
    if type(direct.arrival_batch_id) is not int or direct.arrival_batch_id < 1:
        raise _Reject
    reader = _Reader(engine, actors, start, direct.arrival_batch_id - 1, 10)
    roll = reader.take("roll", r.racer_id, amount=5)
    event = engine.events[reader.offset]
    if type(event.decision_id) is not int or event.decision_id < 1:
        raise _Reject
    record = engine.decisions[event.decision_id - 1]
    public = tuple(PublicRacer(actor.racer_id, actor.name, positions[actor.racer_id], "active", actor.ability.value,
                              actor.tripped, None, None) for actor in (engine.racers[rid] for rid in engine.turn_order))
    observation = Observation(r.racer_id, len(engine.turns) + 1, engine.track.name, engine.track.finish_position,
        engine.track.second_corner_position, tuple(sorted(engine.track.trip_positions)), public, (), 5, None,
        dicemonger_rerolls_remaining=1, score_totals=tuple(engine.score_ledger.totals().items()))
    legal = (Action(ActionKind.KEEP), Action(ActionKind.DICEMONGER_REROLL, d.racer_id, r.racer_id))
    obs = record.request.observation
    if (type(record.request.decision_id) is not int or type(record.request.player_id) is not int
            or type(record.event_sequence) is not int
            or any(type(value) is not int for value in (obs.current_player_id, obs.turn_number,
                obs.magician_rerolls_remaining, obs.dicemonger_rerolls_remaining))
            or any(type(rid) is not int or type(total) is not int for rid, total in obs.score_totals)
            or any(type(actor.racer_id) is not int or type(actor.position) is not int
                   or type(actor.tripped) is not bool for actor in obs.racers)
            or any(value is not None and type(value) is not int
                   for action in record.request.legal_actions for value in (action.source_id, action.target_id))
            or record.request != DecisionRequest(event.decision_id, r.racer_id, DecisionWindow.REROLL, observation, legal)
            or record.action != legal[0] or record.event_sequence != event.sequence
            or len(engine.decisions) != event.decision_id):
        raise _Reject
    reader.take("decision", r.racer_id, parent_sequence=roll.sequence, action_kind="keep", decision_id=event.decision_id)
    locked = reader.take("roll_locked", r.racer_id, amount=5, parent_sequence=roll.sequence)
    resolved = reader.take("main_move_resolved", r.racer_id, before=5, after=5, parent_sequence=locked.sequence)
    reader.batch += 1
    move = reader.take("move", r.racer_id, before=5, after=10, amount=5, movement_kind="main",
                       parent_sequence=resolved.sequence, arrival_batch_id=reader.batch)
    stop = reader.take("stop", r.racer_id, before=5, after=10, parent_sequence=move.sequence, arrival_batch_id=reader.batch)
    original_r = reader.queued_r(stop)
    power = reader.take("ability_effect", r.racer_id, source_id=r.racer_id, ability="Romantic", amount=2,
                        parent_sequence=original_r.queued_sequence)
    original_s = reader.queued_s(r.racer_id, power)
    reader.batch += 1
    moved = reader.take("move", r.racer_id, source_id=r.racer_id, ability="Romantic", before=10, after=12, amount=2,
                        movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=reader.batch)
    returned = reader.take("stop", r.racer_id, before=10, after=12, parent_sequence=moved.sequence, arrival_batch_id=reader.batch)
    fifo = deque((original_s, reader.queued_r(returned)))
    root_ids, root_end = tuple(range(start + 1, reader.offset + 1)), reader.offset
    limit = len(engine.events) if end_event_count is None else end_event_count
    if type(limit) is not int or not root_end <= limit <= len(engine.events):
        raise _Reject
    reader.phase = 12
    while reader.offset < limit:
        if not fifo:
            raise _Reject
        fifo.extend(reader.job(fifo.popleft()))
    if reader.offset != limit or not fifo or not _ledger_valid(engine, start):
        raise _Reject
    if end_event_count is None and (reader.phase != r.position or reader.batch != engine._arrival_counter
                                    or tuple(fifo) != tuple(engine.pending_effects)):
        raise _Reject
    return root_ids, root_end, tuple(fifo), reader.batch, reader.phase


def key(engine):
    try:
        if engine.track != WILD_WILDS:
            return None
        # Reject unrelated rosters before scanning their potentially long
        # histories. The complete typed gates still run for every admitted cut.
        actors = _actors(engine)
        if tuple(actor.position for actor in actors) != (12, 12, 13, 10):
            return None
        if (set(vars(engine)) - _ENGINE_FIELDS
                or type(engine.pending_effects) is not deque or type(engine.finish_order) is not list
                or any(type(getattr(engine, name)) is not dict for name in ("racers", "policies", "policy_metadata"))
                or engine.termination_reason is not None or engine.finish_order
                or engine._preparing is not False or engine._turn_in_progress is not True
                or type(engine._copy_locks) is not set or engine._copy_locks or engine._refreshing_copies is not False
                or type(getattr(engine, "_movement_depth", 0)) is not int or getattr(engine, "_movement_depth", 0) != 0
                or getattr(engine, "_inert_duel_romantic", None) is not None
                or type(getattr(engine, "_duel_offered_epochs", {})) is not dict or getattr(engine, "_duel_offered_epochs", {})
                or engine.next_turn_racer_id is not None or engine._genius_prediction is not None
                or engine.seed is not None and type(engine.seed) is not int
                or type(engine.max_events) is not int or engine.max_events < 1
                or type(engine.race_index) is not int or engine.race_index not in (1, 2, 3, 4)
                or type(engine._score_start) is not int or engine._score_start < 0
                or type(engine.score_ledger) is not ScoreLedger
                or any(type(getattr(engine, name)) is not list for name in (
                    "turn_order", "initial_racers", "events", "decisions", "turns", "remaining_abilities", "egg_drawn"))
                or any(type(getattr(engine, name)) is not tuple for name in ("initial_remaining_abilities", "previous_winners"))
                or hasattr(engine, "max_action_events") and (
                    type(engine.max_action_events) is not int or engine.max_action_events < 1)
                or any(type(getattr(engine, name, None)) is not int or getattr(engine, name) < 1
                       for name in ("_resolution_depth", "_physical_resolution_depth", "_arrival_counter"))
                or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
                or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
                or any(type(rid) is not int for rid in engine.turn_order)
                or any(type(event) is not RaceEvent or set(vars(event)) != _EVENT_FIELDS
                       or type(event.sequence) is not int or event.sequence != i + 1 for i, event in enumerate(engine.events))):
            return None
        if engine.turn_order[engine.turn_index] != actors[0].racer_id or actors[0].tripped is not False:
            return None
        for actor in actors:
            if (type(actor) is not Racer or set(vars(actor)) != _RACER_FIELDS
                    or type(actor.racer_id) is not int or engine.racers.get(actor.racer_id) is not actor
                    or type(actor.ability) is not Ability or type(actor.status) is not RacerStatus
                    or actor.status != RacerStatus.ACTIVE or actor.pre_race_resolved is not True
                    or type(actor.position) is not int or type(actor.tripped) is not bool
                    or type(actor.turns_taken) is not int or actor.turns_taken < 0
                    or actor.inherited_ability is not None or actor.copied_target is not None or actor.copied_ability is not None
                    or actor.prediction is not None or actor.finish_place is not None or actor.finish_places):
                return None
        initial = {row["racer_id"]: row for row in engine.initial_racers}
        if (len(initial) != 4 or set(initial) != set(engine.racers)
                or any(type(row["racer_id"]) is not int for row in engine.initial_racers)
                or any(initial[actor.racer_id]["ability"] != actor.ability.value for actor in actors)
                or any(not _known_pending(job) for job in engine.pending_effects)
                or {job.racer_id for job in engine.pending_effects} != {actors[0].racer_id, actors[1].racer_id}):
            return None
        root_ids, root_end, _, _, _ = _root(engine, actors)
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        immutable = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                                "policies": engine.policy_metadata}, sort_keys=True, separators=(",", ":"))
        state = (engine.track, roster, engine.score_ledger.snapshot(), tuple(engine.turns), tuple(engine.turn_order),
            engine.turn_index, engine.race_index, tuple(engine.remaining_abilities), tuple(engine.initial_remaining_abilities),
            tuple(engine.previous_winners), tuple(engine.egg_drawn), engine._resolution_depth, engine._physical_resolution_depth,
            immutable, engine._score_start, engine.initial_code_fingerprint, engine.max_events,
            getattr(engine, "max_action_events", None), engine.seed, tuple(sorted(engine.policies)), tuple(sorted(vars(engine))),
            root_ids, root_end)
        hash(state)
        return state
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, StopIteration, _Reject):
        return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple[PendingEffect, ...]
    arrival_counter: int
    score_before: ScoreSnapshot
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple
    root_event_ids: tuple[int, ...]


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual:
        raise ValueError("not a certified original Romantic/Scoocher/Baby shared12 cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine._arrival_counter, engine.score_ledger.snapshot(), tuple(engine.events), tuple(engine.decisions), actual[-2])


def completed(engine, snapshot):
    try:
        if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
                or type(snapshot.event_start) is not int or snapshot.event_start != len(snapshot.event_prefix)
                or type(snapshot.decision_start) is not int or snapshot.decision_start != len(snapshot.decision_prefix)
                or tuple(engine.decisions) != snapshot.decision_prefix
                or engine.score_ledger.snapshot() != snapshot.score_before or snapshot.root_event_ids != snapshot.state[-2]
                or any(not _known_pending(job) for job in snapshot.original_pending)):
            return False
        actors = _actors(engine)
        _, _, actual_original, actual_arrivals, phase = _root(engine, actors, end_event_count=snapshot.event_start)
        if (actual_original != snapshot.original_pending or actual_arrivals != snapshot.arrival_counter or phase != 12):
            return False
        reader = _Reader(engine, actors, snapshot.event_start, snapshot.arrival_counter, 12)
        fifo = deque(snapshot.original_pending)
        while reader.offset < len(engine.events):
            if not fifo:
                return False
            fifo.extend(reader.job(fifo.popleft()))
        old = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        if (not reader.executed or old != sorted(set(old)) or not old or not fifo
                or [identity for identity in reader.executed if identity in old] != old
                or len(reader.executed) != len(set(reader.executed)) or set(old).intersection(fresh)
                or any(identity <= snapshot.event_start for identity in fresh)
                or tuple(fifo) != tuple(engine.pending_effects)
                or reader.phase != 12 or reader.batch != engine._arrival_counter):
            return False
        return {"capture_event_count": snapshot.event_start, "root_event_ids": list(snapshot.root_event_ids),
                "original_pending_ids": old, "executed_pending_ids": reader.executed, "new_pending_ids": fresh,
                "queue_before": len(old), "queue_after": len(fresh), "typed_job_counts": dict(sorted(reader.counts.items())),
                "full_R_orbits": reader.counts["R_at14"], "all_original_jobs_executed": True,
                "mechanical_state_returned": True, "score_ledger_unchanged": True,
                "only_fresh_pending_descendants_stopped": True,
                "proof": "native Wild shared12: every S preserves one S and every R adds S; exact actual FIFO bodies, full mechanics and ledger return"}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False
