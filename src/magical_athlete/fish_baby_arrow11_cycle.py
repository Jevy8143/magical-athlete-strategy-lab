"""Recognize a narrow all-choice Arrow11/Baby/Fish physical cycle.

This module reads real prefixes only. It never moves a racer, selects an
action, executes a qualification, or clears a queue. KEEP repeats the single
arrow state. FOLLOW enters a forced simultaneous two-racer arrow state;
the existing forced-arrow recognizer handles that actual branch separately.
All original and newly registered Romantic qualifications remain owed.
"""
from dataclasses import asdict, dataclass, fields
from collections import deque
import json

from .copy_fish_baby_cycle import _ENGINE_FIELDS
from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .fish_baby_star_cycle import _event
from .racer import Ability, Racer
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .track import WILD_WILDS

REASON = "fish_baby_arrow11_choice_cycle_proved"
CACHE_TAG = "fish-baby-arrow11-single-choice-v1"
_ROSTER_R = frozenset({Ability.HUGE_BABY, Ability.SUCKERFISH, Ability.LOVEABLE_LOSER, Ability.ROMANTIC})
_ROSTER_G = frozenset({Ability.HUGE_BABY, Ability.SUCKERFISH, Ability.ROCKET_SCIENTIST, Ability.GUNK})
_RACER_FIELDS = frozenset(field.name for field in fields(Racer))
_EVENT_FIELDS = frozenset(field.name for field in fields(RaceEvent))
_PENDING_FIELDS = frozenset(field.name for field in fields(PendingEffect))
_ALLOWED_ENGINE = _ENGINE_FIELDS | {"_arrow_visits", "_arrow_queue_visits"}


class _Reject(Exception):
    pass


def _plain(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _actors(engine):
    if type(engine.racers) is not dict or len(engine.racers) != 4:
        return None
    roles = {actor.ability: actor for actor in engine.racers.values()}
    if set(roles) == _ROSTER_R:
        return ("Romantic", roles[Ability.LOVEABLE_LOSER], roles[Ability.SUCKERFISH],
                roles[Ability.HUGE_BABY], roles[Ability.ROMANTIC])
    if set(roles) == _ROSTER_G:
        return ("Gunk", roles[Ability.ROCKET_SCIENTIST], roles[Ability.SUCKERFISH],
                roles[Ability.HUGE_BABY], roles[Ability.GUNK])
    return None


def _typed_request(record):
    request, action = record.request, record.action
    public = request.observation
    if (type(request) is not DecisionRequest or type(public) is not Observation
            or type(request.player_id) is not int or type(request.decision_id) is not int
            or type(record.event_sequence) is not int
            or any(type(getattr(public, name)) is not int for name in (
                "current_player_id", "turn_number", "finish_position", "second_corner_position",
                "magician_rerolls_remaining", "dicemonger_rerolls_remaining"))
            or public.latest_roll is not None and type(public.latest_roll) is not int
            or public.current_move_value is not None and type(public.current_move_value) is not int
            or any(type(value) is not int for pair in public.score_totals for value in pair)
            or any(type(value) is not int for value in public.trip_positions)
            or any(type(actor) is not PublicRacer or type(actor.racer_id) is not int
                   or type(actor.position) is not int or type(actor.tripped) is not bool
                   or actor.copied_target is not None and type(actor.copied_target) is not int
                   for actor in public.racers)
            or any(type(choice) is not Action or not isinstance(choice.kind, ActionKind)
                   or any(value is not None and type(value) is not int for value in (
                       choice.source_id, choice.target_id, choice.value))
                   for choice in (*request.legal_actions, action))):
        raise _Reject


class _Reader:
    def __init__(self, engine, actors, start, decision, batch, stop_sequence, *, root=False):
        self.engine, self.actors = engine, actors
        self.domain, mover, fish, baby, observer = actors
        self.m, self.f, self.b, self.o = (actor.racer_id for actor in actors[1:])
        self.positions = {self.m: 13 if root else 11, self.f: 11, self.b: 12,
                          self.o: 4 if self.domain == "Romantic" else 22}
        self.offset, self.decision, self.batch = start, decision, batch
        self.stop_sequence = stop_sequence
        self.children, self.choice_ids = [], []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.engine.events):
            raise _Reject
        event = self.engine.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def observation(self, *, roll=None, move=1):
        actors = {actor.racer_id: actor for actor in self.actors[1:]}
        return Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            WILD_WILDS.name, WILD_WILDS.finish_position, WILD_WILDS.second_corner_position,
            tuple(sorted(WILD_WILDS.trip_positions)),
            tuple(PublicRacer(rid, actors[rid].name, self.positions[rid], "active", actors[rid].ability.value,
                              False, None, None) for rid in self.engine.turn_order), (), roll, move,
            score_totals=tuple(self.engine.score_ledger.totals().items()))

    def romantic(self, stop):
        if self.domain == "Romantic":
            queued = self.take("effect_queued", self.o, source_id=self.o, target_id=self.o,
                               ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
            self.children.append(PendingEffect(self.o, self.o, stop.sequence, queued.sequence,
                                               kind="move", steps=2, ability=Ability.ROMANTIC.value,
                                               target_id=self.o))

    def root(self, parent):
        self.batch += 1
        main = self.take("move", self.m, before=13, after=16, amount=3, movement_kind="main",
                         parent_sequence=parent.sequence, arrival_batch_id=self.batch)
        self.positions[self.m] = 16
        outer = self.take("stop", self.m, before=13, after=16,
                          parent_sequence=main.sequence, arrival_batch_id=self.batch)
        self.batch += 1
        self.take("move", self.m, before=16, after=12, amount=-4, movement_kind="track",
                  parent_sequence=outer.sequence, arrival_batch_id=self.batch)
        baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                         parent_sequence=outer.sequence, arrival_batch_id=self.batch)
        self.batch += 1
        returned = self.take("stop", self.m, before=12, after=11, parent_sequence=baby.sequence,
                             arrival_batch_id=self.batch)
        self.positions[self.m] = 11
        self.stop_sequence = returned.sequence
        self.romantic(returned)

    def keep(self):
        if self.decision >= len(self.engine.decisions):
            raise _Reject
        record = self.engine.decisions[self.decision]
        _typed_request(record)
        legal = (Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, self.f, self.m))
        request = DecisionRequest(self.decision + 1, self.f, DecisionWindow.MOVE_START,
                                  self.observation(), legal)
        if record.request != request or record.action != legal[0]:
            raise _Reject
        decision = self.take("decision", self.f, action_kind=ActionKind.KEEP.value,
                             decision_id=request.decision_id, parent_sequence=self.stop_sequence)
        if record.event_sequence != decision.sequence:
            raise _Reject
        self.decision += 1
        self.choice_ids.append(decision.sequence)
        self.batch += 1
        self.take("move", self.m, before=11, after=12, amount=1, movement_kind="track",
                  parent_sequence=self.stop_sequence, arrival_batch_id=self.batch)
        baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                         parent_sequence=self.stop_sequence, arrival_batch_id=self.batch)
        self.batch += 1
        returned = self.take("stop", self.m, before=12, after=11,
                             parent_sequence=baby.sequence, arrival_batch_id=self.batch)
        self.stop_sequence = returned.sequence
        self.romantic(returned)


def _history_positions(engine, boundary):
    actors = {row["racer_id"]: row for row in engine.initial_racers}
    if len(actors) != 4 or set(actors) != set(engine.racers):
        raise _Reject
    positions = {rid: row["position"] for rid, row in actors.items()}
    trips = {rid: row["tripped"] for rid, row in actors.items()}
    for rid, actor in engine.racers.items():
        if actors[rid]["ability"] != actor.ability.value or actors[rid]["inherited_ability"] is not None:
            raise _Reject
    for event in engine.events[:boundary]:
        if event.kind in {"move", "warp"}:
            if event.before != positions[event.racer_id] or type(event.after) is not int or not 0 <= event.after < 30:
                raise _Reject
            positions[event.racer_id] = event.after
        elif event.kind == "stop":
            if positions[event.racer_id] not in {event.before, event.after} or type(event.after) is not int:
                raise _Reject
            positions[event.racer_id] = event.after
        elif event.kind == "trip":
            trips[event.racer_id] = True
        elif event.kind == "recover":
            trips[event.racer_id] = False
        elif event.kind in {"finish", "eliminate", "inherit", "copy_changed"}:
            raise _Reject
    if any(value is not False for value in trips.values()):
        raise _Reject
    return positions


def _birth(engine, actors, stop):
    domain, mover, fish, baby, observer = actors
    m, f, b, o = (actor.racer_id for actor in actors[1:])
    parents = [event for event in engine.events if event.kind == "main_move_resolved"]
    if not parents:
        raise _Reject
    parent = parents[-1]
    locked = engine.events[parent.parent_sequence - 1] if type(parent.parent_sequence) is int else None
    roll = engine.events[locked.parent_sequence - 1] if locked is not None and type(locked.parent_sequence) is int else None
    amount = 3 if domain == "Romantic" else 4
    if (parent.racer_id != m or not _event(parent, "main_move_resolved", m, before=3, after=3,
                                          parent_sequence=parent.parent_sequence)
            or locked is None or roll is None
            or not 1 <= roll.sequence < locked.sequence < parent.sequence
            or not _event(roll, "roll", m, amount=amount)
            or not _event(locked, "roll_locked", m, amount=amount, parent_sequence=roll.sequence)):
        raise _Reject
    positions = _history_positions(engine, parent.sequence)
    if positions != {m: 13, f: 11, b: 12, o: 4 if domain == "Romantic" else 22}:
        raise _Reject
    if domain == "Romantic":
        if parent.sequence != locked.sequence + 1 or locked.sequence != roll.sequence + 1:
            raise _Reject
    else:
        if (locked.sequence != roll.sequence + 1 or parent.sequence != locked.sequence + 3):
            raise _Reject
        decision, modifier = engine.events[locked.sequence:locked.sequence + 2]
        if (not _event(decision, "decision", m, action_kind=ActionKind.KEEP.value,
                       decision_id=decision.decision_id, parent_sequence=locked.sequence)
                or not _event(modifier, "modifier", m, source_id=o, ability=Ability.GUNK.value,
                              before=4, after=3, amount=-1, operation="add", parent_sequence=locked.sequence)):
            raise _Reject
        record = engine.decisions[decision.decision_id - 1]
        _typed_request(record)
        prototype = _Reader(engine, actors, 0, decision.decision_id - 1, 0, 0, root=True)
        legal = (Action(ActionKind.KEEP), Action(ActionKind.ROCKET_DOUBLE, m, m))
        expected = DecisionRequest(decision.decision_id, m, DecisionWindow.MAIN_MOVE_MODIFIER,
                                   prototype.observation(roll=4, move=4), legal)
        if (record.event_sequence != decision.sequence or record.request != expected or record.action != legal[0]):
            raise _Reject
    direct = engine.events[parent.sequence]
    if direct.kind != "move" or type(direct.arrival_batch_id) is not int or direct.arrival_batch_id < 1:
        raise _Reject
    decision_count = sum(record.event_sequence <= parent.sequence for record in engine.decisions)
    reader = _Reader(engine, actors, parent.sequence, decision_count, direct.arrival_batch_id - 1, 0, root=True)
    reader.root(parent)
    first_cut = reader.offset
    while reader.offset < len(engine.events):
        reader.keep()
    if (reader.offset != len(engine.events) or reader.decision != len(engine.decisions)
            or reader.batch != engine._arrival_counter or reader.stop_sequence != stop.sequence
            or reader.positions != {actor.racer_id: actor.position for actor in actors[1:]}
            or tuple(reader.children) != tuple(engine.pending_effects)):
        raise _Reject
    return {"root_main_sequence": parent.sequence, "root_roll_sequence": roll.sequence,
            "root_locked_roll": amount, "root_final_move": 3, "first_clean_cut_event_count": first_cut,
            "domain": domain}


def _pending(engine, actors):
    domain, mover, fish, baby, observer = actors
    if type(engine.pending_effects) is not deque:
        return False
    if domain == "Gunk":
        return not engine.pending_effects
    consumed = {event.parent_sequence for event in engine.events if event.kind in {"ability_effect", "effect_cancelled"}}
    ids = []
    for job in engine.pending_effects:
        if (type(job) is not PendingEffect or set(vars(job)) != _PENDING_FIELDS
                or any(type(value) is not int for value in (job.racer_id, job.source_id, job.steps,
                    job.target_id, job.trigger_sequence, job.queued_sequence))
                or (job.racer_id, job.source_id, job.kind, job.steps, job.ability, job.target_id)
                    != (observer.racer_id, observer.racer_id, "move", 2, Ability.ROMANTIC.value, observer.racer_id)
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)
                or job.queued_sequence in consumed):
            return False
        queued, stop = engine.events[job.queued_sequence - 1], engine.events[job.trigger_sequence - 1]
        if (not _event(queued, "effect_queued", observer.racer_id, source_id=observer.racer_id,
                       target_id=observer.racer_id, ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
                or stop.kind != "stop" or stop.racer_id != mover.racer_id or stop.before != 12 or stop.after != 11):
            return False
        ids.append(job.queued_sequence)
    return bool(ids) and ids == sorted(set(ids))


def key(engine, group, position, stop):
    try:
        if (set(vars(engine)) - _ALLOWED_ENGINE or engine.track != WILD_WILDS
                or type(engine._arrow_visits) is not set or type(engine._arrow_queue_visits) is not dict
                or type(engine._copy_locks) is not set or engine._copy_locks
                or type(engine.events) is not list or type(engine.decisions) is not list
                or type(engine.turns) is not list or type(engine.turn_order) is not list
                or type(engine.finish_order) is not list or engine.finish_order
                or type(engine.policies) is not dict or type(engine.policy_metadata) is not dict
                or engine._preparing is not False or engine._turn_in_progress is not True
                or engine._refreshing_copies is not False or engine.termination_reason is not None
                or engine.next_turn_racer_id is not None or engine._genius_prediction is not None
                or getattr(engine, "_inert_duel_romantic", None) is not None or getattr(engine, "_duel_offered_epochs", {})
                or any(type(getattr(engine, name, None)) is not int or getattr(engine, name) < 1 for name in (
                    "_resolution_depth", "_physical_resolution_depth", "_movement_depth", "_arrival_counter"))
                or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
                or type(engine.race_index) is not int or not 1 <= engine.race_index <= 4
                or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
                or any(type(rid) is not int for rid in engine.turn_order)):
            return None
        actors = _actors(engine)
        if actors is None:
            return None
        domain, mover, fish, baby, observer = actors
        if (position != 11 or group != [(mover.racer_id, 1)]
                or engine.turn_order[engine.turn_index] != mover.racer_id
                or (mover.position, fish.position, baby.position, observer.position)
                    != (11, 11, 12, 4 if domain == "Romantic" else 22)):
            return None
        for actor in actors[1:]:
            if (type(actor) is not Racer or set(vars(actor)) != _RACER_FIELDS
                    or type(actor.racer_id) is not int or type(actor.position) is not int
                    or actor.status != RacerStatus.ACTIVE or actor.tripped is not False
                    or actor.pre_race_resolved is not True or actor.inherited_ability is not None
                    or actor.copied_target is not None or actor.copied_ability is not None
                    or actor.prediction is not None or actor.finish_place is not None or actor.finish_places
                    or type(actor.turns_taken) is not int or actor.turns_taken < 0):
                return None
        if (any(type(event) is not RaceEvent or set(vars(event)) != _EVENT_FIELDS
                or type(event.sequence) is not int or event.sequence != index + 1
                for index, event in enumerate(engine.events))
                or type(stop) is not RaceEvent or not 1 <= stop.sequence <= len(engine.events)
                or engine.events[stop.sequence - 1] != stop or stop.kind != "stop"
                or stop.racer_id != mover.racer_id or stop.before != 12 or stop.after != 11
                or not _pending(engine, actors)):
            return None
        root = _birth(engine, actors, stop)
        fixed = {"domain": domain, "roster": [{**asdict(actor), "status": actor.status.value}
                                              for _, actor in sorted(engine.racers.items())],
            "turns": [asdict(turn) for turn in engine.turns], "turn_order": engine.turn_order,
            "turn_index": engine.turn_index, "race_index": engine.race_index,
            "remaining": engine.remaining_abilities, "initial_remaining": engine.initial_remaining_abilities,
            "egg_drawn": engine.egg_drawn, "previous_winners": engine.previous_winners,
            "scores": asdict(engine.score_ledger.snapshot()), "initial_scores": engine.initial_score_ledger,
            "initial_racers": engine.initial_racers, "policy_metadata": engine.policy_metadata,
            "policy_seats": sorted(engine.policies), "score_start": engine._score_start,
            "fingerprint": engine.initial_code_fingerprint, "seed": engine.seed,
            "max_events": engine.max_events, "max_action_events": getattr(engine, "max_action_events", None),
            "known_fields": sorted(vars(engine)), "root": root}
        return json.dumps(fixed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, _Reject):
        return None


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: str
    event_start: int
    decision_start: int
    arrival_counter: int
    stop_sequence: int
    original_pending: tuple[PendingEffect, ...]
    score_before: ScoreSnapshot
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple
    depths: tuple[int, int, int]


def capture(engine, state, stop):
    actors = _actors(engine)
    if actors is None or key(engine, [(actors[1].racer_id, 1)], 11, stop) != state:
        raise ValueError("not a certified single Fish/Baby Arrow11 cut")
    return Snapshot(state, len(engine.events), len(engine.decisions), engine._arrival_counter, stop.sequence,
        tuple(engine.pending_effects), engine.score_ledger.snapshot(), tuple(engine.events), tuple(engine.decisions),
        (engine._resolution_depth, engine._physical_resolution_depth, engine._movement_depth))


def completed(engine, snapshot, stop):
    try:
        actors = _actors(engine)
        if (actors is None or key(engine, [(actors[1].racer_id, 1)], 11, stop) != snapshot.state
                or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
                or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix
                or engine.score_ledger.snapshot() != snapshot.score_before
                or (engine._resolution_depth, engine._physical_resolution_depth, engine._movement_depth)
                    != (snapshot.depths[0] + 3, snapshot.depths[1] + 3, snapshot.depths[2] + 1)):
            return False
        reader = _Reader(engine, actors, snapshot.event_start, snapshot.decision_start,
                         snapshot.arrival_counter, snapshot.stop_sequence)
        reader.keep()
        if (reader.offset != len(engine.events) or reader.decision != len(engine.decisions)
                or reader.batch != engine._arrival_counter or reader.stop_sequence != stop.sequence
                or tuple(engine.pending_effects) != snapshot.original_pending + tuple(reader.children)):
            return False
        return {"capture_event_count": snapshot.event_start, "old_arrow_stop_sequence": snapshot.stop_sequence,
            "current_arrow_stop_sequence": stop.sequence, "fish_decision_event_ids": reader.choice_ids,
            "actual_choice": "keep", "domain": actors[0],
            "original_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
            "new_deferred_pending_ids": [job.queued_sequence for job in reader.children],
            "all_owed_pending_ids": [job.queued_sequence for job in engine.pending_effects],
            "original_and_new_qualifications_preserved": True, "pending_jobs_executed_by_certificate": 0,
            "pending_jobs_cancelled_by_certificate": 0, "scores_unchanged": True,
            "mechanical_single_arrow_state_returned": True, "one_complete_actual_arrow_period": True,
            "only_repeated_arrow_step_stopped": True,
            "all_choice_proof": "KEEP returns the single state; FOLLOW reaches the closed forced two-arrow state. Simultaneous coarrival produces no Romantic qualification. No policy RNG or future action is inspected."}
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, _Reject):
        return False
