"""A forced Romantic/Scoocher/Baby FIFO with real Star drift on Wild.

P and effective Scoocher S share p, Baby B stays p+1, and Romantic R
alternates12/14 through16's Arrow. The explicit p domain is0..10 except7;
at Trip5 S must already be tripped. A real S job returns p->p+1->Baby p,
scores only on Star1 and appends S,R. An R job appends S and moves12->14
or14->16->12. Every replacement is nonempty; no choices/rolls are opened.
R/S/B can be original or genuinely resolved Egg/Twin static principals;
shared authentication preserves their original entities and complete grants.
The fourth passive entity may also be an original Twin with one real,
direct BEFORE_RACE acquisition of Stickler. One separate exact Mild branch
admits direct Egg->Stickler at R29/S=P/B=p+1 for1..27, with the same FIFO proof.

This module authenticates real history and actual FIFO bodies. It never
executes or removes a task. All captured original jobs must really finish,
the complete mechanical state must return, and every actual Star must
match the full old ledger plus its real entries. Only verified fresh
descendants in the same closed family may then stop under existing O12.
"""

from collections import Counter, deque
from dataclasses import dataclass, fields
import json

from .events import PendingEffect, RaceEvent
from .racer import Ability
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .static_inheritance import authenticated_static_power
from .track import MILD_MILE, WILD_WILDS


REASON = "growing_romantic_baby_scoocher_star_cycle_proved"
RETURN_POSITIONS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10})
BLOCKED_PROFILE = "stickler-blocked-R29-S4-B5-P4-v1"
MILD_EGG_BLOCKED_PROFILE = "stickler-blocked-R29-S-P-Egg-Bplus1-Mild-v1"
MILD_EGG_RETURN_POSITIONS = frozenset(range(1, 28))
ANCHOR22_PROFILE = "native-R22-S-D12-B13-Wild-v1"

# Explicitly reviewed for P STANDING at1, rather than copied powers or a
# different movement path. MOUTH/CENTAUR only listen to their own arrivals/
# passing; P is not this arriver. Leaptoad is not moving, Stickler sees no
# finish overrun. All remaining powers need a turn/roll/main-move window.
# Inheritance, Copy and all other arrival responders stay outside this list.
PASSIVE_ABILITIES = frozenset({
    Ability.SKIPPER, Ability.HECKLER, Ability.INCHWORM, Ability.LACKEY,
    Ability.ALCHEMIST, Ability.BLIMP, Ability.CENTAUR, Ability.CHEERLEADER,
    Ability.COACH, Ability.DICEMONGER, Ability.FLIP_FLOP, Ability.GENIUS,
    Ability.GUNK, Ability.HARE, Ability.HYPNOTIST, Ability.LEAPTOAD,
    Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN, Ability.MASTERMIND,
    Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.STICKLER, Ability.THIRD_WHEEL,
})


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _passive_twin_stickler_grant(engine, actor):
    """Return the sole authenticated direct grant; never infer it from a flag.

    Stickler cannot act on these moves: the largest forward destination is16,
    well before the finish. Its physical Twin identity, full real preparation
    and grant still belong to M, and every captured qualification must be born
    after that grant. Egg or chained acquisitions remain outside this family.
    """
    if (actor.ability != Ability.TWIN or actor.inherited_ability != Ability.STICKLER
            or not authenticated_static_power(engine, actor, Ability.STICKLER)):
        return None
    grants = [event for event in engine.events
              if event.kind == "inherit" and event.racer_id == actor.racer_id]
    if len(grants) != 1 or grants[0].ability != Ability.STICKLER.value:
        return None
    return grants[0].sequence


def _passive_egg_stickler_grant(engine, actor):
    """One actual direct Egg draw/choice/grant, not an inherited flag.

    Shared authentication reconstructs the exact three-card public menu and
    complete preparation. No Twin chain or dynamic copied power is added.
    The existing blocked FIFO reader is reused only at the exact Mild cut.
    """
    if (actor.ability != Ability.EGG or actor.inherited_ability != Ability.STICKLER
            or not authenticated_static_power(engine, actor, Ability.STICKLER)):
        return None
    grants = [event for event in engine.events
              if event.kind == "inherit" and event.racer_id == actor.racer_id]
    if len(grants) != 1 or grants[0].ability != Ability.STICKLER.value:
        return None
    return grants[0].sequence


def _passive_stickler_grant(engine, actor):
    return (_passive_egg_stickler_grant(engine, actor) if actor.ability == Ability.EGG
            else _passive_twin_stickler_grant(engine, actor))


def _mild_egg_blocked_profile(actors):
    r, s, b, p = actors
    return (p.ability == Ability.EGG and p.inherited_ability == Ability.STICKLER
            and (r.ability, s.ability, b.ability) == (
                Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY)
            and s.position in MILD_EGG_RETURN_POSITIONS
            and (r.position, b.position, p.position) == (29, s.position + 1, s.position))


def _actors(engine):
    actors = tuple(engine.racers.values())
    if (len(actors) != 4 or any(not isinstance(actor.ability, Ability) for actor in actors)
            or len({actor.ability for actor in actors}) != 4):
        return None
    principals = []
    for power in (Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY):
        matches = [actor for actor in actors if engine._ability(actor) == power]
        if len(matches) != 1 or not authenticated_static_power(engine, matches[0], power):
            return None
        principals.append(matches[0])
    if len({actor.racer_id for actor in principals}) != 3:
        return None
    p = next(actor for actor in actors if actor not in principals)
    native_passive = p.ability in PASSIVE_ABILITIES and p.inherited_ability is None
    if not native_passive and _passive_stickler_grant(engine, p) is None:
        return None
    result = (*principals, p)
    if p.ability == Ability.EGG and (engine.track != MILD_MILE or not _mild_egg_blocked_profile(result)):
        return None
    return result


def _inheritance_bindings(engine, actors):
    """Bind authenticated grants into full M without relabelling entities."""
    records = {record.event_sequence: record for record in engine.decisions}
    result = []
    principals = list(zip(actors[:3], (Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY)))
    if actors[3].ability in {Ability.TWIN, Ability.EGG}:
        principals.append((actors[3], Ability.STICKLER))
    for actor, power in principals:
        if actor.ability != power:
            chain = tuple((records[event.parent_sequence], event) for event in engine.events
                          if event.kind == "inherit" and event.racer_id == actor.racer_id)
            result.append((actor.racer_id, actor.ability.value, power.value, chain))
    return tuple(result)


def _blocked_stickler_profile(actors):
    """A separate proved branch: a blocked R job is only Stickler's use.

    _move checks _blocked before emitting the moving Romantic's power.
    Thus real R29+2 yields move_blocked->S, not a Romantic notification.
    Only the observed original R/S/B family at4/5 is included here.
    """
    if _mild_egg_blocked_profile(actors):
        return True
    r, s, b, p = actors
    stickler = (p.ability == Ability.STICKLER and p.inherited_ability is None
                or p.ability == Ability.TWIN and p.inherited_ability == Ability.STICKLER)
    return (stickler and (r.ability, s.ability, b.ability) == (
                Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY)
            and (r.position, s.position, b.position, p.position) == (29, 4, 5, 4))


def _anchor22_dicemonger_profile(engine, actors):
    """A separate single-phase R22 return, with S/D12 and native Baby13.

    Dicemonger only offers an actual roll window, never one inside this
    forced FIFO. S12->13->Baby12 replaces S by S,R; R22->24->Arrow22
    replaces R by S. No job changes a Trip flag or earns a Star here.
    Qualification history and all old jobs use the same existing reader.
    In particular a just-created Dicemonger S must execute before this key
    can accept the family, and no pending main-roll debt is marked complete.
    """
    r, s, b, p = actors
    return (engine.track == WILD_WILDS
            and tuple(actor.ability for actor in actors) == (
                Ability.ROMANTIC, Ability.SCOOCHER, Ability.HUGE_BABY, Ability.DICEMONGER)
            and (r.position, s.position, b.position, p.position) == (22, 12, 13, 12)
            and engine.turn_order[engine.turn_index] == s.racer_id)


def _queue_valid(engine, pending, actors, *, require_unconsumed=True):
    r, s, b, p = (actor.racer_id for actor in actors)
    passive_grant = _passive_stickler_grant(engine, actors[3]) if actors[3].ability in {Ability.TWIN, Ability.EGG} else None
    if actors[3].ability in {Ability.TWIN, Ability.EGG} and passive_grant is None:
        return False
    blocked = _blocked_stickler_profile(actors)
    if not pending or any(not isinstance(job, PendingEffect) for job in pending):
        return False
    for job in pending:
        if any(type(value) is not int for value in (job.racer_id, job.source_id, job.steps,
                                                   job.trigger_sequence, job.queued_sequence)):
            return False
        if (job.racer_id not in {r, s} or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)
                or job.racer_id == r and (type(job.target_id) is not int
                    or (job.source_id, job.kind, job.steps, job.ability, job.target_id) != (
                        r, "move", 2, Ability.ROMANTIC.value, r))
                or job.racer_id == s and (job.source_id not in ({r, b, p} if blocked else {r, b})
                    or (job.kind, job.steps, job.ability, job.target_id) != ("move", 1, Ability.SCOOCHER.value, None))):
            return False
    ids = [job.queued_sequence for job in pending]
    if ids != sorted(set(ids)):
        return False
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    if set(initial) != set(engine.racers):
        return False
    positions = {rid: item["position"] for rid, item in initial.items()}
    grants = {actor.racer_id: max((event.sequence for event in engine.events
              if event.kind == "inherit" and event.racer_id == actor.racer_id), default=0) for actor in actors[:3]}
    if blocked:
        grants[p] = passive_grant or 0
    inherited_ids = {actor.racer_id for actor in actors[:3] if actor.ability in {Ability.EGG, Ability.TWIN}}
    if passive_grant is not None:
        inherited_ids.add(p)
    needed = {job.trigger_sequence for job in pending}
    if blocked:
        needed.update(event.parent_sequence for event in engine.events
                      if event.kind == "effect_queued" and event.racer_id == r
                      and type(event.parent_sequence) is int)
    historical, batches, consumed = {}, Counter(), set()
    numeric = ("racer_id", "source_id", "target_id", "before", "after", "amount", "parent_sequence",
               "decision_id", "arrival_batch_id")
    for index, event in enumerate(engine.events):
        if (type(event.sequence) is not int or event.sequence != index + 1
                or any(getattr(event, name) is not None and type(getattr(event, name)) is not int for name in numeric)):
            return False
        if event.kind in {"move", "warp", "stop"}:
            positions[event.racer_id] = event.after
        if (event.kind in {"finish", "eliminate", "race_end"}
                or event.kind == "inherit" and event.racer_id not in inherited_ids):
            return False
        if event.kind in {"ability_effect", "effect_cancelled"}:
            consumed.add(event.parent_sequence)
        if (blocked and event.kind == "move_blocked" and event.racer_id == r
                and event.movement_kind == "ability"):
            if (type(event.parent_sequence) is not int
                    or not 1 <= event.parent_sequence < event.sequence
                    or not _event(event, "move_blocked", r, source_id=p, ability=Ability.STICKLER.value,
                                  before=29, after=29, amount=2, movement_kind="ability",
                                  parent_sequence=event.parent_sequence)):
                return False
            qualification = engine.events[event.parent_sequence - 1]
            if not _event(qualification, "effect_queued", r, source_id=r, target_id=r,
                          ability=Ability.ROMANTIC.value, parent_sequence=qualification.parent_sequence):
                return False
            consumed.add(event.parent_sequence)
        if event.kind == "stop":
            batches[event.arrival_batch_id] += 1
        if event.sequence in needed:
            historical[event.sequence] = dict(positions)
    if positions != {rid: actor.position for rid, actor in engine.racers.items()}:
        return False
    for job in pending:
        if (require_unconsumed and job.queued_sequence in consumed
                or passive_grant is not None and job.trigger_sequence <= passive_grant
                or job.trigger_sequence <= grants[job.racer_id]
                or job.racer_id == s and job.trigger_sequence <= grants[job.source_id]):
            return False
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if job.racer_id == r:
            previous = historical[job.trigger_sequence]
            if (trigger.kind != "stop" or trigger.racer_id not in engine.racers
                    or type(trigger.before) is not int or type(trigger.after) is not int
                    or not 0 <= trigger.after < engine.track.finish_position
                    or type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                    or type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1
                    or not _event(trigger, "stop", trigger.racer_id, before=trigger.before, after=trigger.after,
                                  parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)
                    or batches[trigger.arrival_batch_id] != 1
                    or previous[trigger.racer_id] != trigger.after
                    or sum(position == trigger.after for position in previous.values()) != 2
                    or not _event(queued, "effect_queued", r, source_id=r, target_id=r,
                                  ability=Ability.ROMANTIC.value, parent_sequence=trigger.sequence)):
                return False
        else:
            if (type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence
                    or not _event(queued, "effect_queued", s, source_id=job.source_id,
                                  ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)):
                return False
            if job.source_id == r:
                if (not _event(trigger, "ability_effect", r, source_id=r, ability=Ability.ROMANTIC.value,
                               amount=2, parent_sequence=trigger.parent_sequence)
                        or engine.events[trigger.parent_sequence - 1].kind != "effect_queued"
                        or engine.events[trigger.parent_sequence - 1].racer_id != r
                        or engine.events[trigger.parent_sequence - 1].ability != Ability.ROMANTIC.value):
                    return False
            elif job.source_id == p and blocked:
                qualification = engine.events[trigger.parent_sequence - 1]
                if (not _event(trigger, "move_blocked", r, source_id=p, ability=Ability.STICKLER.value,
                               before=29, after=29, amount=2, movement_kind="ability",
                               parent_sequence=qualification.sequence)
                        or not _event(qualification, "effect_queued", r, source_id=r, target_id=r,
                                      ability=Ability.ROMANTIC.value, parent_sequence=qualification.parent_sequence)
                        or type(qualification.parent_sequence) is not int
                        or not 1 <= qualification.parent_sequence < qualification.sequence
                        or qualification.parent_sequence <= max(grants[r], grants[p])
                        or queued.sequence != trigger.sequence + 1):
                    return False
                arrival = engine.events[qualification.parent_sequence - 1]
                previous = historical.get(arrival.sequence, {})
                if (arrival.kind != "stop" or arrival.racer_id not in {s, p} or arrival.after != actors[1].position
                        or type(arrival.parent_sequence) is not int
                        or not 1 <= arrival.parent_sequence < arrival.sequence
                        or type(arrival.arrival_batch_id) is not int
                        or not _event(arrival, "stop", arrival.racer_id, before=arrival.before, after=actors[1].position,
                                      parent_sequence=arrival.parent_sequence, arrival_batch_id=arrival.arrival_batch_id)
                        or batches[arrival.arrival_batch_id] != 1
                        or {rid for rid, position in previous.items() if position == actors[1].position} != {s, p}):
                    return False
            elif (type(trigger.arrival_batch_id) is not int or trigger.arrival_batch_id < 1
                    or not _event(trigger, "ability_effect", b, source_id=b, ability=Ability.HUGE_BABY.value,
                                  parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)
                    or historical[trigger.sequence][b] == 0
                    or not any(rid != b and position == historical[trigger.sequence][b]
                               for rid, position in historical[trigger.sequence].items())):
                return False
    return True


def key(engine):
    if (engine.track not in {WILD_WILDS, MILD_MILE} or engine.termination_reason is not None or engine.finish_order
            or engine._preparing is not False or engine._turn_in_progress is not True
            or engine._copy_locks or engine._refreshing_copies
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0
            or type(getattr(engine, "_resolution_depth", 0)) is not int or engine._resolution_depth < 1
            or type(getattr(engine, "_arrival_counter", None)) is not int or engine._arrival_counter < 1
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    r, s, b, p = actors
    if engine.track == MILD_MILE and not _mild_egg_blocked_profile(actors):
        return None
    position = s.position
    blocked = _blocked_stickler_profile(actors)
    positions = MILD_EGG_RETURN_POSITIONS if engine.track == MILD_MILE else RETURN_POSITIONS
    anchor22 = _anchor22_dicemonger_profile(engine, actors)
    if ((r.position not in (12, 14) and not blocked and not anchor22)
            or position not in positions and not anchor22
            or (b.position, p.position) != (position + 1, position)
            or position in engine.track.trip_positions and s.tripped is not True):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE
                or actor.copied_target is not None or actor.copied_ability is not None
                or actor.pre_race_resolved is not True or actor.finish_place is not None or actor.finish_places
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or type(actor.tripped) is not bool or type(actor.turns_taken) is not int or actor.turns_taken < 0
                or actor.prediction is not None and (actor.ability != Ability.MASTERMIND
                    or type(actor.prediction) is not int or actor.prediction not in engine.racers)):
            return None
    next_player, prediction = engine.next_turn_racer_id, getattr(engine, "_genius_prediction", None)
    if (next_player is not None and (type(next_player) is not int or next_player not in engine.racers)
            or prediction is not None and (type(prediction) is not int or not 1 <= prediction <= 6)):
        return None
    try:
        if not _queue_valid(engine, tuple(engine.pending_effects), actors):
            return None
        roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                       for rid, actor in sorted(engine.racers.items()))
        initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger}, sort_keys=True)
        state = (engine.track, engine._state_key(include_scores=False, include_pending=False), roster,
                 tuple(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
                 tuple(engine.egg_drawn), tuple(engine.previous_winners), engine._preparing, engine._turn_in_progress,
                 engine._resolution_depth, tuple(sorted(getattr(engine, "_duel_offered_epochs", {}).items())),
                 initial, engine._score_start, engine.initial_code_fingerprint, engine.max_events,
                 tuple(engine.initial_remaining_abilities), _inheritance_bindings(engine, actors))
        if blocked:
            state = (*state, MILD_EGG_BLOCKED_PROFILE if engine.track == MILD_MILE else BLOCKED_PROFILE)
        if anchor22:
            state = (*state, ANCHOR22_PROFILE)
        hash(state)
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return state


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
    if actual is None or state is not None and state != actual:
        raise ValueError("not a certified Romantic/Scoocher/Baby Star FIFO cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    engine.score_ledger.snapshot(), engine._arrival_counter, tuple(engine.events), tuple(engine.decisions))


class _Reject(Exception):
    pass


class _Reader:
    def __init__(self, engine, snapshot, actors):
        self.engine, self.snapshot = engine, snapshot
        self.r, self.s, self.b, self.p = (actor.racer_id for actor in actors)
        # completed first requires full M equality, so this is also the exact
        # starting geometry. Scores are authenticated separately below.
        self.positions = {actor.racer_id: actor.position for actor in actors}
        self.return_position = self.positions[self.s]
        self.events = engine.events[snapshot.event_start:]
        self.offset, self.batch = 0, snapshot.arrival_counter
        self.balances = dict(snapshot.score_before.totals)
        self.children, self.stars = [], []

    def take(self, kind, actor, **values):
        if self.offset >= len(self.events):
            raise _Reject
        event = self.events[self.offset]
        if not _event(event, kind, actor, **values):
            raise _Reject
        self.offset += 1
        return event

    def notify(self, source, cause):
        event = self.take("effect_queued", self.s, source_id=source,
                          ability=Ability.SCOOCHER.value, parent_sequence=cause.sequence)
        self.children.append(PendingEffect(self.s, source, cause.sequence, event.sequence))

    def scoocher(self, pending):
        position = self.return_position
        power = self.take("ability_effect", self.s, source_id=self.s, ability=Ability.SCOOCHER.value,
                          amount=1, parent_sequence=pending.queued_sequence)
        self.batch += 1
        self.take("move", self.s, source_id=self.s, ability=Ability.SCOOCHER.value,
                  before=position, after=position + 1, amount=1,
                  movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=self.batch)
        baby = self.take("ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                         parent_sequence=power.sequence, arrival_batch_id=self.batch)
        self.notify(self.b, baby)
        self.batch += 1
        stop = self.take("stop", self.s, before=position + 1, after=position,
                         parent_sequence=baby.sequence, arrival_batch_id=self.batch)
        queued = self.take("effect_queued", self.r, source_id=self.r, target_id=self.r,
                           ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
        self.children.append(PendingEffect(self.r, self.r, stop.sequence, queued.sequence,
                                           "move", 2, Ability.ROMANTIC.value, self.r))
        if position in self.engine.track.star_positions:
            balance = self.balances[self.s]
            score = self.take("score", self.s, before=balance, after=balance + 1, amount=1,
                              parent_sequence=stop.sequence, reason="star")
            self.balances[self.s] += 1
            self.stars.append(score)

    def romantic(self, pending):
        before = self.positions[self.r]
        if before == 29:
            blocked = self.take("move_blocked", self.r, source_id=self.p, ability=Ability.STICKLER.value,
                                before=29, after=29, amount=2, movement_kind="ability",
                                parent_sequence=pending.queued_sequence)
            self.notify(self.p, blocked)
            return
        power = self.take("ability_effect", self.r, source_id=self.r, ability=Ability.ROMANTIC.value,
                          amount=2, parent_sequence=pending.queued_sequence)
        self.notify(self.r, power)
        self.batch += 1
        move = self.take("move", self.r, source_id=self.r, ability=Ability.ROMANTIC.value,
                         before=before, after=before + 2, amount=2, movement_kind="ability",
                         parent_sequence=power.sequence, arrival_batch_id=self.batch)
        stop = self.take("stop", self.r, before=before, after=before + 2,
                         parent_sequence=move.sequence, arrival_batch_id=self.batch)
        if before == 14:
            self.batch += 1
            arrow = self.take("move", self.r, before=16, after=12, amount=-4, movement_kind="track",
                              parent_sequence=stop.sequence, arrival_batch_id=self.batch)
            self.take("stop", self.r, before=16, after=12,
                      parent_sequence=arrow.sequence, arrival_batch_id=self.batch)
        elif before == 22:
            self.batch += 1
            arrow = self.take("move", self.r, before=24, after=22, amount=-2, movement_kind="track",
                              parent_sequence=stop.sequence, arrival_batch_id=self.batch)
            self.take("stop", self.r, before=24, after=22,
                      parent_sequence=arrow.sequence, arrival_batch_id=self.batch)
        self.positions[self.r] = 22 if before == 22 else 26 - before

    def job(self, pending):
        self.children = []
        if pending.racer_id == self.s:
            self.scoocher(pending)
        elif pending.racer_id == self.r:
            self.romantic(pending)
        else:
            raise _Reject
        return tuple(self.children)


def completed(engine, snapshot):
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    try:
        if not _queue_valid(engine, snapshot.original_pending, actors, require_unconsumed=False):
            return False
    except (KeyError, ValueError, TypeError, IndexError):
        return False
    current, old = engine.score_ledger.snapshot(), snapshot.score_before
    if current.player_ids != old.player_ids or current.entries[:len(old.entries)] != old.entries:
        return False
    events = engine.events[snapshot.event_start:]
    if not events or any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(events)):
        return False
    reader, fifo = _Reader(engine, snapshot, actors), deque(snapshot.original_pending)
    labels = {actors[0].racer_id: "R", actors[1].racer_id: "S"}
    executed, counts = [], {"R": 0, "S": 0}
    try:
        while reader.offset < len(events):
            if not fifo:
                return False
            pending = fifo.popleft()
            children = reader.job(pending)
            expected = "SR" if pending.racer_id == reader.s else "S"
            if "".join(labels[job.racer_id] for job in children) != expected:
                return False
            executed.append(pending.queued_sequence)
            counts[labels[pending.racer_id]] += 1
            fifo.extend(children)
    except (KeyError, ValueError, IndexError, _Reject):
        return False
    original = [job.queued_sequence for job in snapshot.original_pending]
    fresh = [job.queued_sequence for job in fifo]
    if (reader.batch != engine._arrival_counter
            or counts["R"] % 2 and not _anchor22_dicemonger_profile(engine, actors)
            or [identity for identity in executed if identity in original] != original
            or len(executed) != len(set(executed)) or set(original).intersection(fresh)
            or any(identity <= snapshot.event_start for identity in fresh)
            or not fifo or tuple(fifo) != tuple(engine.pending_effects)
            or reader.positions != {actor.racer_id: actor.position for actor in actors}
            or reader.balances != engine.score_ledger.totals()
            or current.totals != tuple((rid, reader.balances[rid]) for rid, _ in old.totals)):
        return False
    added = current.entries[len(old.entries):]
    if (len(added) != len(reader.stars)
            or any((entry.sequence, entry.player_id, entry.race_index, entry.delta, entry.source, entry.reason,
                    entry.event_sequence, entry.phase) != (
                        len(old.entries) + index + 1, event.racer_id, engine.race_index, 1, "track", "star",
                        event.sequence, "during_race")
                   for index, (entry, event) in enumerate(zip(added, reader.stars)))):
        return False
    races = {race: dict(totals) for race, totals in old.race_totals}
    for entry in added:
        races[entry.race_index][entry.player_id] += entry.delta
    expected_races = tuple((race, tuple((rid, races[race][rid]) for rid, _ in totals)) for race, totals in old.race_totals)
    if current.race_totals != expected_races:
        return False
    result = {"original_pending_ids": original, "executed_pending_ids": executed, "new_pending_ids": fresh,
            "queue_before": len(original), "queue_after": len(fresh), "job_counts": counts,
            "original_romantic_id": reader.r, "scoocher_id": reader.s, "baby_id": reader.b, "passive_id": reader.p,
            "passive_ability": actors[3].ability.value,
            "star_point_increments": {str(rid): reader.balances[rid] - total for rid, total in old.totals},
            "score_event_ids": [event.sequence for event in reader.stars], "mechanical_state_returned": True,
            "all_original_jobs_executed": True, "only_fresh_pending_descendants_stopped": True,
            "all_real_Star_entries_retained": True, "new_input_events": 0,
            "proof": "nonempty FIFO substitutions S->SR and R->S; complete actual cohort, full mechanical return and exact Star ledger projection"}
    # Preserve every original p1/native certificate byte: historical TRAIN
    # sources may already contain these exact JSON operations. New bindings
    # appear only for an actually expanded position or static principal.
    if reader.return_position != 1:
        result["return_position"] = reader.return_position
    inherited = _inheritance_bindings(engine, actors)
    if inherited:
        result["static_principals"] = [
            {"racer_id": rid, "original_ability": original, "effective_ability": effective,
             "inherit_event_sequences": [event.sequence for _, event in chain]}
            for rid, original, effective, chain in inherited]
    if _blocked_stickler_profile(actors):
        result["profile"] = MILD_EGG_BLOCKED_PROFILE if engine.track == MILD_MILE else BLOCKED_PROFILE
        result["romantic_position"] = 29
        result["stickler_id"] = reader.p
        result["romantic_job_transition"] = "move_blocked before Romantic power notification; one real Stickler S child"
    if _anchor22_dicemonger_profile(engine, actors):
        result["profile"] = ANCHOR22_PROFILE
        result["romantic_position"] = 22
        result["romantic_job_transition"] = "one genuine S child; R22->24->official Arrow22 returns in one job"
    return result
