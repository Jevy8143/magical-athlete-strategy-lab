"""Authenticate the all-choice Fish/Scoocher/Baby FIFO on printed Wild Stars.

At F=S=p in {1,13}, B=p+1, KEEP really returns S and adds one Star;
FOLLOW really returns both actors and adds two Stars. The respective FIFO
substitutions are S->S and S->SS. Both are nonempty for every legal choice,
including choices based on the changing public scores. This proof therefore
projects scores and the decision counter only after authenticating every
actual choice, observation, complete original job, and actual Star entry.
It neither executes nor removes tasks. Its caller stops only fresh children.
"""

from collections import deque
from dataclasses import dataclass, fields
import json

from .decisions import Action, ActionKind, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .fish_baby_cycle import (PASSIVE_SPECTATOR_ABILITIES, _actors, _queue_valid,
                              _new_static_principals, _principal_bindings, _principal_packet)
from .racer import Ability
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .static_inheritance import static_inheritance_evidence
from .track import WILD_WILDS


REASON = "fish_baby_star_choice_queue_cycle_proved"
RETURN_POSITIONS = frozenset({1, 13})
# These original powers cannot respond to somebody else's arrival at their
# square. Mouth/Centaur listen only to their own arrival/passing. All other
# admitted powers require their own turn, roll, main move or finish. Keep the
# three unproved same-square responders outside this family explicitly.
COLOCATED_PASSIVE_ABILITIES = PASSIVE_SPECTATOR_ABILITIES - {
    Ability.BABA_YAGA, Ability.BANANA, Ability.DUELIST,
}


def _event(event, kind, actor, **values):
    numeric = ("sequence", "racer_id", "source_id", "target_id", "before", "after", "amount",
               "parent_sequence", "decision_id", "arrival_batch_id")
    return (isinstance(event, RaceEvent)
            and all(getattr(event, name) is None or type(getattr(event, name)) is int for name in numeric)
            and event == RaceEvent(event.sequence, kind, actor, **values))


def _qualified(engine, pending, fish, baby, scoocher, *, require_unconsumed=True):
    """Use the existing real source gate, also rejecting consumed native jobs."""
    if not _queue_valid(engine, pending, fish, baby, scoocher,
                        require_unconsumed=require_unconsumed):
        return False
    if any(type(job.steps) is not int for job in pending):
        return False
    ids = [job.queued_sequence for job in pending]
    if ids != sorted(set(ids)):
        return False
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}}
    if require_unconsumed and consumed.intersection(ids):
        return False
    for job in pending:
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if not _event(queued, "effect_queued", scoocher.racer_id, source_id=job.source_id,
                      ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence):
            return False
        if job.source_id == fish.racer_id:
            if not _event(trigger, "ability_effect", fish.racer_id, source_id=fish.racer_id,
                          target_id=scoocher.racer_id, ability=Ability.SUCKERFISH.value, amount=1,
                          parent_sequence=trigger.parent_sequence):
                return False
        elif (type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1
              or not _event(trigger, "ability_effect", baby.racer_id, source_id=baby.racer_id,
                            ability=Ability.HUGE_BABY.value, parent_sequence=trigger.parent_sequence,
                            arrival_batch_id=trigger.arrival_batch_id)):
            return False
    return True


def key(engine):
    """Complete mechanical/control state; FIFO, actual choices and ledger separate."""
    if (engine.track != WILD_WILDS or len(engine.racers) != 4
            or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0
            or type(getattr(engine, "_resolution_depth", None)) is not int or engine._resolution_depth < 1
            or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    fish, spectator, baby, scoocher = actors
    position = scoocher.position
    if (type(position) is not int or position not in RETURN_POSITIONS
            or fish.position != position or baby.position != position + 1
            or spectator.position == position + 1
            or spectator.position == position and spectator.ability not in COLOCATED_PASSIVE_ABILITIES):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE or actor.pre_race_resolved is not True
                or actor is spectator and actor.inherited_ability is not None
                or actor.copied_target is not None or actor.copied_ability is not None
                or actor.finish_place is not None or actor.finish_places
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or not 0 <= actor.position < engine.track.finish_position
                or type(actor.tripped) is not bool or type(actor.turns_taken) is not int or actor.turns_taken < 0
                or actor.prediction is not None and (actor.ability != Ability.MASTERMIND
                    or type(actor.prediction) is not int or actor.prediction not in engine.racers)):
            return None
    next_player, prediction = engine.next_turn_racer_id, getattr(engine, "_genius_prediction", None)
    if (next_player is not None and (type(next_player) is not int or next_player not in engine.racers)
            or prediction is not None and (type(prediction) is not int or not 1 <= prediction <= 6)):
        return None
    try:
        pending = tuple(engine.pending_effects)
        if not _qualified(engine, pending, fish, baby, scoocher):
            return None
        inheritance = static_inheritance_evidence(engine, baby, Ability.HUGE_BABY)
        if inheritance is None:
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        actual = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                  "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
        epoch = (len(engine.turns), next((event.sequence for event in reversed(engine.events)
                                        if event.kind in actual), 0))
        offered = getattr(engine, "_duel_offered_epochs", {})
        control = (tuple(sorted(offered.items())),
                   tuple((rid, offered.get(rid) == epoch) for rid in sorted(engine.racers)))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger},
                             sort_keys=True, separators=(",", ":"))
        state = (engine.track, engine._state_key(include_scores=False, include_pending=False,
                                               include_decisions=False), roster, tuple(engine.turns),
                 engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._preparing,
                 engine._turn_in_progress, engine._resolution_depth, initial, engine._score_start,
                 engine.initial_code_fingerprint, engine.max_events, tuple(engine.initial_remaining_abilities),
                 inheritance, control)
        if _new_static_principals(fish, scoocher):
            bindings = _principal_bindings(engine, fish, baby, scoocher)
            if bindings is None:
                return None
            state = (*state[:-1], bindings, state[-1])
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
        raise ValueError("not a certified Fish/Baby Star FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine.score_ledger.snapshot(), engine._arrival_counter,
                    tuple(engine.events), tuple(engine.decisions))


class _Reject(Exception):
    pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot = engine, snapshot
        self.f, self.p, self.b, self.s = (actor.racer_id for actor in actors)
        self.position = actors[3].position
        self.events = engine.events[snapshot.event_start:]
        self.choices = engine.decisions[snapshot.decision_start:]
        self.offset, self.choice_index, self.batch = 0, 0, snapshot.arrival_counter
        self.balances = dict(snapshot.score_before.totals)
        self.stars, self.executed, self.choice_ids = [], [], []
        self.follow_count = 0

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def observe(self):
        return Observation(
            self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            self.engine.track.name, self.engine.track.finish_position, self.engine.track.second_corner_position,
            tuple(sorted(self.engine.track.trip_positions)),
            tuple(PublicRacer(actor.racer_id, actor.name, actor.position, actor.status.value,
                              self.engine._ability(actor).value, actor.tripped, actor.finish_place,
                              actor.copied_target)
                  for actor in (self.engine.racers[rid] for rid in self.engine.turn_order)),
            tuple(self.engine.finish_order), None, 1,
            score_totals=tuple((rid, self.balances[rid]) for rid, _ in self.snapshot.score_before.totals))

    def star(self, actor, stop):
        balance = self.balances[actor]
        event = self.take("score", actor, before=balance, after=balance + 1, amount=1,
                          parent_sequence=stop.sequence, reason="star")
        self.balances[actor] += 1
        self.stars.append(event)

    def job(self, pending):
        if self.choice_index >= len(self.choices):
            raise _Reject
        record = self.choices[self.choice_index]
        self.choice_index += 1
        request, action = record.request, record.action
        legal = (Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, self.f, self.s))
        observation = request.observation
        if (type(request.decision_id) is not int
                or any(type(value) is not int for value in (observation.current_player_id, observation.turn_number,
                    observation.current_move_value, observation.magician_rerolls_remaining,
                    observation.dicemonger_rerolls_remaining))
                or any(type(rid) is not int or type(total) is not int for rid, total in observation.score_totals)
                or any(type(actor.racer_id) is not int or type(actor.position) is not int
                       or type(actor.tripped) is not bool for actor in observation.racers)
                or request.decision_id != self.snapshot.decision_start + self.choice_index
                or request.player_id != self.f or request.window != DecisionWindow.MOVE_START
                or request.legal_actions != legal or action not in legal
                or request.observation != self.observe()):
            raise _Reject
        decision = self.take("decision", self.f, source_id=action.source_id, target_id=action.target_id,
                             parent_sequence=pending.queued_sequence, action_kind=action.kind.value,
                             decision_id=request.decision_id)
        if record.event_sequence != decision.sequence:
            raise _Reject
        self.choice_ids.append(decision.sequence)
        power = self.take("ability_effect", self.s, source_id=self.s, ability=Ability.SCOOCHER.value,
                          amount=1, parent_sequence=pending.queued_sequence)
        children = []
        follow = action.kind == ActionKind.SUCKERFISH_FOLLOW
        if follow:
            fish = self.take("ability_effect", self.f, source_id=self.f, target_id=self.s,
                             ability=Ability.SUCKERFISH.value, amount=1, parent_sequence=power.sequence)
            queued = self.take("effect_queued", self.s, source_id=self.f,
                               ability=Ability.SCOOCHER.value, parent_sequence=fish.sequence)
            children.append(PendingEffect(self.s, self.f, fish.sequence, queued.sequence))
            self.follow_count += 1
        self.batch += 1
        self.take("move", self.s, source_id=self.s, ability=Ability.SCOOCHER.value,
                  before=self.position, after=self.position + 1, amount=1, movement_kind="ability",
                  parent_sequence=power.sequence, arrival_batch_id=self.batch)
        if follow:
            self.take("move", self.f, source_id=self.f, ability=Ability.SUCKERFISH.value,
                      before=self.position, after=self.position + 1, amount=1, movement_kind="ability",
                      parent_sequence=fish.sequence, arrival_batch_id=self.batch)
        baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                         parent_sequence=power.sequence, arrival_batch_id=self.batch)
        queued = self.take("effect_queued", self.s, source_id=self.b,
                           ability=Ability.SCOOCHER.value, parent_sequence=baby.sequence)
        children.append(PendingEffect(self.s, self.b, baby.sequence, queued.sequence))
        self.batch += 1
        order = [rid for rid in self.engine._clockwise_order() if rid in {self.s, self.f}] if follow else [self.s]
        stops = [self.take("stop", rid, before=self.position + 1, after=self.position,
                           parent_sequence=baby.sequence, arrival_batch_id=self.batch) for rid in order]
        # The whole batch stops before _resolve_stops awards its clockwise Stars.
        for rid, stop in zip(order, stops):
            self.star(rid, stop)
        self.executed.append(pending.queued_sequence)
        return tuple(children)


def completed(engine, snapshot):
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    try:
        if not _qualified(engine, snapshot.original_pending, actors[0], actors[2], actors[3],
                          require_unconsumed=False):
            return False
        old, current = snapshot.score_before, engine.score_ledger.snapshot()
        if (current.player_ids != old.player_ids or current.entries[:len(old.entries)] != old.entries
                or any(type(rid) is not int or type(total) is not int for rid, total in old.totals)):
            return False
        events = engine.events[snapshot.event_start:]
        if not events or any(event.sequence != snapshot.event_start + i + 1 for i, event in enumerate(events)):
            return False
        reader, fifo = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
        while reader.offset < len(events):
            if not fifo:
                return False
            fifo.extend(reader.job(fifo.popleft()))
        original = [job.queued_sequence for job in snapshot.original_pending]
        fresh = [job.queued_sequence for job in fifo]
        if (reader.choice_index != len(reader.choices) or reader.batch != engine._arrival_counter
                or [identity for identity in reader.executed if identity in original] != original
                or len(reader.executed) != len(set(reader.executed))
                or not fifo or set(original).intersection(fresh)
                or any(identity <= snapshot.event_start for identity in fresh)
                or tuple(fifo) != tuple(engine.pending_effects)
                or reader.balances != engine.score_ledger.totals()
                or current.totals != tuple((rid, reader.balances[rid]) for rid, _ in old.totals)):
            return False
        added = current.entries[len(old.entries):]
        if (len(added) != len(reader.stars)
                or any((entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source,
                        entry.reason, entry.event_sequence, entry.phase) != (
                            len(old.entries) + i + 1, event.racer_id, engine.race_index, 1, "track",
                            "star", event.sequence, "during_race")
                       for i, (entry, event) in enumerate(zip(added, reader.stars)))):
            return False
        races = {race: dict(totals) for race, totals in old.race_totals}
        for entry in added:
            races[entry.race_index][entry.player_id] += 1
        if current.race_totals != tuple((race, tuple((rid, races[race][rid]) for rid, _ in totals))
                                       for race, totals in old.race_totals):
            return False
        result = {"original_pending_ids": original, "executed_pending_ids": reader.executed,
                  "new_pending_ids": fresh, "queue_before": len(original), "queue_after": len(fresh),
                  "suckerfish_decision_event_ids": reader.choice_ids,
                  "keep_jobs": len(reader.executed) - reader.follow_count, "follow_jobs": reader.follow_count,
                  "return_position": actors[3].position,
                  "star_point_increments": {str(rid): reader.balances[rid] - total for rid, total in old.totals},
                  "score_event_ids": [event.sequence for event in reader.stars],
                  "all_original_jobs_executed": True, "mechanical_state_returned": True,
                  "only_fresh_pending_descendants_stopped": True, "all_real_Star_entries_retained": True,
                  "proof": "all legal Fish choices preserve nonempty FIFO: KEEP S->S; FOLLOW S->SS; exact real observations, Stars and full mechanical return"}
        if actors[2].ability != Ability.HUGE_BABY:
            result["static_baby"] = {"racer_id": actors[2].racer_id, "original_ability": actors[2].ability.value,
                                     "inherited_ability": actors[2].inherited_ability.value,
                                     "acquisition": static_inheritance_evidence(engine, actors[2], Ability.HUGE_BABY)}
        if _new_static_principals(actors[0], actors[3]):
            result["static_principals"] = _principal_packet(engine, actors[0], actors[2], actors[3])
        return result
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False
