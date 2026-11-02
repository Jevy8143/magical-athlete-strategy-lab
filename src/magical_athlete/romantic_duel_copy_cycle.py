"""A whole-cohort R/C FIFO cycle after a real native Duelist loses to Copy.

D stays12, R/C return together to14. Copy's forced D/R choice at12
converges to R before its next old Romantic job. Native D jobs reach their
turn only after both movers leave12, so they really execute without a window.
No pre-Duel choice or die is removed; only fresh descendants stop.
"""

from collections import deque
from dataclasses import dataclass, field, fields
from functools import lru_cache
import hashlib
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .track import WILD_WILDS
from .romantic_anchor_cycle import _queue_valid as _romantic_queue_valid


REASON = "romantic_duel_copy_fifo_cycle_proved"


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _successors(state):
    r, c, target, word = state
    head, tail = word[0], word[1:]
    if head == "D":
        if r != 14 or c != 14:
            raise ValueError("Duelist would have a genuine shared target")
        return (("silent", (r, c, target, tail)),)
    actor = 1 if head == "R" else 2
    positions = {0: 12, 1: r, 2: c, 3: 5}
    children = ""
    destination = positions[actor] + 2
    for stop in ((destination, 12) if destination == 16 else (destination,)):
        positions[actor] = stop
        count = sum(value == stop for value in positions.values())
        if count == 2:
            children += "R" + ("C" if target == "R" else "")
        if target == "D" and positions[2] == stop and count > 1:
            raise ValueError("a copied Duelist opportunity would be born")
        if stop == 12 and count > 1:
            children += "D"
    leaders = [rid for rid in (1, 2, 3, 0) if positions[rid] == max(positions.values()) and rid != 2]
    if len(leaders) > 1:
        if target is not None:
            raise ValueError("unexpected optional Copy window")
        choices = (("copy_r", "R"), ("copy_d", "D"))
    else:
        following = "R" if leaders == [1] else "D" if leaders == [0] else None
        choices = (("forced", following),)
    return tuple((action, (positions[1], positions[2], following, tail + children))
                 for action, following in choices)


@lru_cache(maxsize=1)
def _finite_proof():
    first = (14, 14, "R", "RC")
    states, todo, edges = {first}, deque([first]), []
    while todo:
        state = todo.popleft()
        for action, target in _successors(state):
            if not target[3]:
                raise ValueError("D/R/Copy family became drainable")
            edges.append((state, action, target))
            if target not in states:
                states.add(target)
                todo.append(target)
    if len(states) != 7 or len(edges) != 8:
        raise ValueError("D/R/Copy finite FIFO family changed")
    table = {"states": sorted(states, key=repr), "edges": sorted(edges, key=repr)}
    digest = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return frozenset(states), tuple(edges), digest


def _actors(engine):
    actors = tuple(engine.racers.values())
    if len(actors) != 4 or any(not isinstance(actor.ability, Ability) for actor in actors) or {actor.ability for actor in actors} != {
            Ability.DUELIST, Ability.ROMANTIC, Ability.COPY_CAT, Ability.GENIUS}:
        return None
    by_power = {actor.ability: actor for actor in actors}
    return tuple(by_power[power] for power in (Ability.DUELIST, Ability.ROMANTIC, Ability.COPY_CAT, Ability.GENIUS))


def _observation(engine, actors, positions, target, totals):
    d, r, c, _ = actors
    public = tuple(PublicRacer(
        actor.racer_id, actor.name, positions[actor.racer_id], "active",
        (Ability.ROMANTIC.value if target == r.racer_id else Ability.DUELIST.value if target == d.racer_id else None)
        if actor is c else actor.ability.value,
        actor.tripped, None, target if actor is c else None)
        for actor in (engine.racers[rid] for rid in engine.turn_order))
    return Observation(engine.turn_order[engine.turn_index], len(engine.turns) + 1,
                       WILD_WILDS.name, 30, WILD_WILDS.second_corner_position,
                       tuple(sorted(WILD_WILDS.trip_positions)), public, (), None, None, score_totals=totals)


def key(engine):
    if (engine.track != WILD_WILDS or engine._preparing is not False or engine._turn_in_progress is not True
            or engine.termination_reason is not None or engine.finish_order
            or engine._copy_locks or engine._refreshing_copies
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
            or getattr(engine, "_movement_depth", 0) != 0 or hasattr(engine, "_arrow_visits")
            or hasattr(engine, "_arrow_queue_visits") or getattr(engine, "_inert_duel_romantic", None) is not None):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    d, r, c, g = actors
    if ((d.position, r.position, c.position, g.position) != (12, 14, 14, 5)
            or engine.turn_order[engine.turn_index] != r.racer_id
            or c.copied_target != r.racer_id or c.copied_ability != Ability.ROMANTIC
            or [rid for rid in engine._clockwise_order() if rid != g.racer_id] != [r.racer_id, c.racer_id, d.racer_id]):
        return None
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    if set(initial) != set(engine.racers):
        return None
    for actor in actors:
        if (type(actor.racer_id) is not int or type(actor.position) is not int
                or actor.status is not RacerStatus.ACTIVE or type(actor.tripped) is not bool
                or actor.inherited_ability is not None or actor.pre_race_resolved is not True
                or type(actor.turns_taken) is not int or actor.turns_taken < 0
                or actor.finish_place is not None or actor.finish_places or actor.prediction is not None
                or initial[actor.racer_id]["ability"] != actor.ability
                or initial[actor.racer_id]["name"] != actor.name
                or actor is not c and (actor.copied_target is not None or actor.copied_ability is not None)):
            return None
    try:
        depth, physical = engine._resolution_depth, engine._physical_resolution_depth
        if type(depth) is not int or depth < 1 or type(physical) is not int or physical < 1:
            return None
        pending = tuple(engine.pending_effects)
        if (len(pending) != 2 or [job.racer_id for job in pending] != [r.racer_id, c.racer_id]
                or not _romantic_queue_valid(engine, pending, (r, c, d, g))):
            return None
        offered = getattr(engine, "_duel_offered_epochs", {})
        if set(offered) - {d.racer_id, c.racer_id}:
            return None
        latest = engine._duel_actual_epoch()
        relations = tuple((rid, offered.get(rid) == latest) for rid in (d.racer_id, c.racer_id))
        roster = tuple((rid, tuple((item.name, getattr(actor, item.name)) for item in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        state = (engine.track, engine._state_key(include_pending=False, include_decisions=False), roster,
                 engine.score_ledger.snapshot(), tuple(engine.turns), engine.race_index, tuple(engine.turn_order),
                 tuple(engine.remaining_abilities), tuple(engine.egg_drawn), tuple(engine.previous_winners),
                 depth, physical, tuple(sorted(offered.items())), relations,
                 json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger}, sort_keys=True),
                 engine._score_start, engine.initial_code_fingerprint, engine.max_events)
        hash(state)
        return state
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def _seed(engine, actors, pending):
    """Capture only AFTER the actual two-die C win and both qualifications."""
    d, r, c, g = actors
    first, second = pending
    if first.trigger_sequence != second.trigger_sequence:
        return None
    stop = engine.events[first.trigger_sequence - 1]
    if (stop.racer_id != c.racer_id or stop.kind != "stop" or stop.before != 12 or stop.after != 14
            or [first.queued_sequence, second.queued_sequence] != [stop.sequence + 1, stop.sequence + 2]
            or second.queued_sequence != len(engine.events)):
        return None
    try:
        decision, effect, a, b, move = engine.events[stop.sequence - 6:stop.sequence - 1]
        record = engine.decisions[-1]
        legal = (Action(ActionKind.KEEP), Action(ActionKind.DUEL, d.racer_id, c.racer_id))
        positions = {d.racer_id: 12, r.racer_id: 14, c.racer_id: 12, g.racer_id: 5}
        request = DecisionRequest(record.request.decision_id, d.racer_id, DecisionWindow.DUEL,
                                  _observation(engine, actors, positions, r.racer_id, engine.score_ledger.snapshot().totals), legal)
        if (record.request != request or record.action != legal[1] or record.event_sequence != decision.sequence
                or not _event(decision, "decision", d.racer_id, source_id=d.racer_id, target_id=c.racer_id,
                              action_kind=ActionKind.DUEL.value, decision_id=request.decision_id)
                or not _event(effect, "ability_effect", d.racer_id, source_id=d.racer_id, target_id=c.racer_id,
                              ability=Ability.DUELIST.value, parent_sequence=decision.sequence)
                or any(type(roll.amount) is not int or not 1 <= roll.amount <= 6 for roll in (a, b))
                or not a.amount < b.amount
                or not _event(a, "duel_roll", d.racer_id, source_id=d.racer_id, ability=Ability.DUELIST.value,
                              amount=a.amount, parent_sequence=effect.sequence)
                or not _event(b, "duel_roll", c.racer_id, source_id=d.racer_id, ability=Ability.DUELIST.value,
                              amount=b.amount, parent_sequence=effect.sequence)
                or not _event(move, "move", c.racer_id, source_id=d.racer_id, ability=Ability.DUELIST.value,
                              before=12, after=14, amount=2, movement_kind="ability", parent_sequence=effect.sequence,
                              arrival_batch_id=stop.arrival_batch_id)
                or not _event(stop, "stop", c.racer_id, before=12, after=14,
                              parent_sequence=move.sequence, arrival_batch_id=move.arrival_batch_id)):
            return None
        return (decision.sequence, a.sequence, b.sequence)
    except (ValueError, IndexError, TypeError, AttributeError, KeyError):
        return None


@dataclass(slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple
    arrival_counter: int
    event_prefix: tuple
    decision_prefix: tuple
    seed: tuple
    jobs: list = field(default_factory=list)
    in_flight: object = None
    invalid: bool = False


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        return None
    seed = _seed(engine, _actors(engine), tuple(engine.pending_effects))
    if seed is None:
        return None
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine._arrival_counter, tuple(engine.events), tuple(engine.decisions), seed)


def job_started(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid:
        return
    if snapshot.in_flight is not None or not queue_before or queue_before[0] != pending:
        snapshot.invalid = True
        return
    snapshot.in_flight = (pending, queue_before, event_start)


def job_finished(engine, snapshot, pending, queue_before, event_start):
    if snapshot.invalid:
        return
    if snapshot.in_flight != (pending, queue_before, event_start):
        snapshot.invalid = True
        return
    snapshot.jobs.append((pending, queue_before, event_start, len(engine.events)))
    snapshot.in_flight = None


class _Reject(Exception):
    pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot, self.actors = engine, snapshot, actors
        self.d, self.r, self.c, self.g = (a.racer_id for a in actors)
        self.positions = {a.racer_id: a.position for a in actors}
        self.target = self.r
        self.events = engine.events[snapshot.event_start:]
        self.offset, self.decision, self.batch = 0, snapshot.decision_start, snapshot.arrival_counter
        self.children, self.choices = [], []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def control(self, fifo):
        labels = {self.r: "R", self.c: "C", self.d: "D"}
        return (self.positions[self.r], self.positions[self.c],
                "R" if self.target == self.r else "D" if self.target == self.d else None,
                "".join(labels[job.racer_id] for job in fifo))

    def arrival(self, actor, destination, amount, cause, *, arrow=False):
        before = self.positions[actor]
        self.batch += 1
        values = dict(before=before, after=destination, amount=amount, parent_sequence=cause.sequence,
                      movement_kind="track" if arrow else "ability", arrival_batch_id=self.batch)
        if not arrow:
            values.update(source_id=actor, ability=Ability.ROMANTIC.value)
        move = self.take("move", actor, **values)
        for rid in self.engine._clockwise_order():
            if rid != actor and (before < self.positions[rid] < destination or destination < self.positions[rid] < before):
                self.take("passing", actor, target_id=rid, parent_sequence=move.sequence, arrival_batch_id=self.batch)
        self.positions[actor] = destination
        stop = self.take("stop", actor, before=before, after=destination,
                         parent_sequence=move.sequence, arrival_batch_id=self.batch)
        occupants = sum(p == destination for p in self.positions.values())
        for rid in self.engine._clockwise_order():
            romantic = rid == self.r or rid == self.c and self.target == self.r
            duel = rid == self.d or rid == self.c and self.target == self.d
            if romantic and occupants == 2:
                ability, kind, steps = Ability.ROMANTIC.value, "move", 2
            elif duel and self.positions[rid] == destination and occupants > 1:
                ability, kind, steps = Ability.DUELIST.value, "duel", 0
            else:
                continue
            queued = self.take("effect_queued", rid, source_id=rid, target_id=rid,
                               ability=ability, parent_sequence=stop.sequence)
            self.children.append(PendingEffect(rid, rid, stop.sequence, queued.sequence, kind, steps, ability, rid))
        if destination == 16:
            self.arrival(actor, 12, -4, stop, arrow=True)

    def refresh(self):
        maximum = max(self.positions.values())
        targets = tuple(rid for rid in self.engine._clockwise_order()
                        if rid != self.c and self.positions[rid] == maximum)
        old, action = self.target, "forced"
        if len(targets) > 1:
            if old is not None or targets != (self.r, self.d):
                raise _Reject
            legal = tuple(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets)
            record = self.engine.decisions[self.decision]
            request = DecisionRequest(self.decision + 1, self.c, DecisionWindow.COPY_CHOICE,
                _observation(self.engine, self.actors, self.positions, old, self.snapshot.state[3].totals), legal)
            if record.request != request or record.action not in legal:
                raise _Reject
            event = self.take("decision", self.c, source_id=self.c, target_id=record.action.target_id,
                              action_kind=ActionKind.COPY_TARGET.value, decision_id=request.decision_id)
            if event.sequence != record.event_sequence:
                raise _Reject
            self.target = record.action.target_id
            self.choices.append(event.sequence)
            self.decision += 1
            action = "copy_r" if self.target == self.r else "copy_d"
        else:
            self.target = targets[0] if targets else None
        if old != self.target:
            ability = Ability.ROMANTIC.value if self.target == self.r else Ability.DUELIST.value if self.target == self.d else None
            self.take("copy_changed", self.c, source_id=self.c, target_id=self.target, ability=ability)
        return action

    def job(self, pending):
        self.children = []
        if pending.racer_id == self.d:
            if (pending.kind, pending.steps, pending.ability, pending.source_id, pending.target_id) != (
                    "duel", 0, Ability.DUELIST.value, self.d, self.d) or any(
                        self.positions[rid] == 12 for rid in (self.r, self.c, self.g)):
                raise _Reject
            return (), "silent"
        actor = pending.racer_id
        if actor not in (self.r, self.c) or (pending.kind, pending.steps, pending.ability,
                pending.source_id, pending.target_id) != ("move", 2, Ability.ROMANTIC.value, actor, actor):
            raise _Reject
        effect = self.take("ability_effect", actor, source_id=actor, ability=Ability.ROMANTIC.value,
                           amount=2, parent_sequence=pending.queued_sequence)
        self.arrival(actor, self.positions[actor] + 2, 2, effect)
        action = self.refresh()
        return tuple(self.children), action


def completed(engine, snapshot):
    if (snapshot.invalid or snapshot.in_flight is not None or not snapshot.jobs
            or key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    try:
        if not _romantic_queue_valid(engine, snapshot.original_pending, (actors[1], actors[2], actors[0], actors[3]),
                                      require_unconsumed=False):
            return False
        reader, fifo = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
        states, edges, digest = _finite_proof()
        executed, counts = [], {"R": 0, "C": 0, "D": 0}
        labels = {reader.r: "R", reader.c: "C", reader.d: "D"}
        for pending, actual_before, start, end in snapshot.jobs:
            before = reader.control(fifo)
            if (before not in states or actual_before != tuple(fifo) or not fifo or fifo.popleft() != pending
                    or start != snapshot.event_start + reader.offset):
                return False
            children, action = reader.job(pending)
            fifo.extend(children)
            if ((action, reader.control(fifo)) not in _successors(before)
                    or end != snapshot.event_start + reader.offset):
                return False
            executed.append(pending.queued_sequence)
            counts[labels[pending.racer_id]] += 1
        originals = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        if (reader.offset != len(reader.events) or reader.decision != len(engine.decisions)
                or reader.batch != engine._arrival_counter or tuple(fifo) != tuple(engine.pending_effects)
                or reader.positions != {rid: actor.position for rid, actor in engine.racers.items()}
                or reader.target != actors[2].copied_target
                or [value for value in executed if value in originals] != originals
                or len(executed) != len(set(executed)) or set(originals) & set(fresh)
                or any(value <= snapshot.event_start for value in fresh)
                or any(counts[label] < 2 for label in ("R", "C", "D"))):
            return False
    except (KeyError, ValueError, TypeError, IndexError, AttributeError, _Reject):
        return False
    return {"original_pending_ids": originals, "executed_pending_ids": executed, "new_pending_ids": fresh,
            "queue_before": len(originals), "queue_after": len(fresh), "job_counts": counts,
            "copy_decision_event_ids": reader.choices, "retained_duel_decision_event_id": snapshot.seed[0],
            "retained_duel_roll_event_ids": list(snapshot.seed[1:]),
            "finite_configurations": len(states), "all_legal_action_edges": len(edges), "finite_table_sha256": digest,
            "clockwise_principals": [reader.r, reader.c, reader.d], "duelist_id": reader.d,
            "romantic_id": reader.r, "copy_id": reader.c, "genius_id": reader.g,
            "scores_unchanged": True, "mechanical_state_returned": True,
            "all_original_jobs_executed": True, "silent_duel_jobs_actually_finished": counts["D"],
            "all_actual_copy_choices_retained": True, "new_duel_windows": 0,
            "only_fresh_pending_descendants_stopped": True,
            "proof": "7-state8-edge exactFIFO closure after actual C-winning Duel; all Copy choices converge before old C job; D genuinely has no sharing target"}
