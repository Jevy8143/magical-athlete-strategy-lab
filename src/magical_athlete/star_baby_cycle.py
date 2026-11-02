"""Prove one narrow forced Star cycle with a growing, heterogeneous FIFO.

On official Wild, original Scoocher S follows 12->13->14->15->16->12,
Copy C returns to0<=p<11 except Arrow7, original Baby B stays at p+1 (never
Start), and an original passive spectator stays at q with0<=q<=p. At Trip5
Copy must already be tripped, so the proved period has no new Trip. A winning
Genius prediction or Hypnotist turn-start can queue the prefix before its
main move; unconsumed main moves and next-turn controls are preserved.
S is the unique leader, so C continuously borrows Scoocher without a choice.
The actual transitions are S-job -> [C-job], and C-job -> [S-job] followed
by Baby's two clockwise [S,C] or [C,S] notifications. Every replacement is
nonempty. S gains a printed Star per four S-jobs; C gains one per C-job only
at p=1. Returning to Start0 does not award a Star or reactivate Baby there.
Scores never branch any of these four powers in this prefix.

The generic Star detector deliberately requires an unchanged whole FIFO.
This independent certificate instead verifies every actual job body and its
exact fresh descendants. It stops only after every original job executed,
the full S phase returned, and the complete remaining queue contains solely
proved new descendants. This module neither executes nor removes any task.
"""

from collections import deque
from dataclasses import asdict, dataclass, fields
import json

from .decisions import Action, ActionKind, DecisionRequest, DecisionWindow, Observation, PublicRacer
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .static_inheritance import authenticated_static_power, static_inheritance_evidence
from .track import WILD_WILDS


REASON = "growing_scoocher_copy_baby_star_cycle_proved"
RETURN_POSITIONS = frozenset({0, 1, 2, 3, 4, 5, 6, 8, 9, 10})


# Original spectators only. At the returning Copy's own square, Baba would
# trip it and Suckerfish would offer a real follow choice, so those two are
# admitted only strictly below p. Sisyphus is not admitted: its before-race
# +4 is outside the real normal-turn entry matrix verified for this family.
# Banana is not passed at the starting square
# and Baby's put is not passing. Other listed powers have no window here.
PASSIVE_SPECTATOR_ABILITIES = frozenset({
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
    Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER,
    Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE, Ability.HECKLER,
    Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY, Ability.LEAPTOAD,
    Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN, Ability.MASTERMIND,
    Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL,
})


def _copy_source(engine, actor):
    """Only a sole original Copy or an authenticated, completed static Copy."""
    if actor.ability == Ability.COPY_CAT:
        return ()
    if actor.ability not in {Ability.EGG, Ability.TWIN} or actor.inherited_ability != Ability.COPY_CAT:
        return None
    evidence = static_inheritance_evidence(engine, actor, Ability.COPY_CAT)
    if not evidence:
        return None
    binding = json.loads(evidence[0])
    grants = [item["sequence"] for item in binding["inherit_events"]
              if item["racer_id"] == actor.racer_id and item["ability"] == Ability.COPY_CAT.value]
    if len(grants) != 1:
        return None
    grant = grants[0]
    if any(event.racer_id == actor.racer_id and event.sequence <= grant
           and event.kind in {"ability_effect", "copy_changed"} for event in engine.events):
        return None
    return grant, evidence


def _copy_metadata(actor, source):
    return {"racer_id": actor.racer_id, "original_ability": actor.ability.value,
            "inherited_ability": Ability.COPY_CAT.value, "copy_grant_sequence": source[0],
            "acquisition": source[1]}


def _actors(engine):
    roles = {actor.ability: actor for actor in engine.racers.values()}
    core = (Ability.SCOOCHER,)
    if (len(roles) != 4 or any(not isinstance(actor.ability, Ability) for actor in engine.racers.values())
            or any(power not in roles for power in core)):
        return None
    copies = [actor for actor in engine.racers.values() if engine._is_copy(actor)]
    if len(copies) != 1:
        return None
    copy = copies[0]
    if copy.ability not in {Ability.COPY_CAT, Ability.EGG, Ability.TWIN}:
        return None
    babies = [actor for actor in engine.racers.values()
              if (actor.ability == Ability.HUGE_BABY or actor.ability in {Ability.EGG, Ability.TWIN}
                  and actor.inherited_ability == Ability.HUGE_BABY)
              and authenticated_static_power(engine, actor, Ability.HUGE_BABY)]
    if len(babies) != 1:
        return None
    baby = babies[0]
    spectator = next(actor for power, actor in roles.items()
                     if power not in core and actor is not baby and actor is not copy)
    if spectator.ability not in PASSIVE_SPECTATOR_ABILITIES:
        return None
    return roles[Ability.SCOOCHER], copy, baby, spectator


def _notification(engine, pending, s, c, b):
    if (not isinstance(pending, PendingEffect)
            or any(type(value) is not int for value in (pending.racer_id, pending.source_id,
                                                       pending.steps, pending.trigger_sequence, pending.queued_sequence))
            or (pending.kind, pending.steps, pending.ability, pending.target_id) != ("move", 1, Ability.SCOOCHER.value, None)
            or pending.racer_id not in {s, c}
            or pending.source_id not in ({c, b} if pending.racer_id == s else {s, b})
            or type(pending.trigger_sequence) is not int or type(pending.queued_sequence) is not int
            or not 1 <= pending.trigger_sequence < pending.queued_sequence <= len(engine.events)):
        return False
    trigger = engine.events[pending.trigger_sequence - 1]
    queued = engine.events[pending.queued_sequence - 1]
    power = Ability.HUGE_BABY.value if pending.source_id == b else Ability.SCOOCHER.value
    if ((trigger.kind, trigger.racer_id, trigger.source_id, trigger.target_id, trigger.ability, trigger.amount) != (
            "ability_effect", pending.source_id, pending.source_id, None, power, None if pending.source_id == b else 1)
            or not _event(queued, "effect_queued", pending.racer_id, source_id=pending.source_id,
                          ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)):
        return False
    return True


def _queue_valid(engine, pending, s, c, b, *, require_unconsumed=True):
    if not pending or any(not isinstance(job, PendingEffect) for job in pending):
        return False
    ids = [job.queued_sequence for job in pending]
    if not pending or ids != sorted(set(ids)) or not all(_notification(engine, job, s, c, b) for job in pending):
        return False
    if engine.racers[b].ability != Ability.HUGE_BABY:
        grants = [event.sequence for event in engine.events if event.kind == "inherit"
                  and event.racer_id == b and event.ability == Ability.HUGE_BABY.value]
        if not grants or any(job.source_id == b and job.trigger_sequence <= grants[-1] for job in pending):
            return False
    source = _copy_source(engine, engine.racers[c])
    if source is None or source and any(job.trigger_sequence <= source[0] for job in pending):
        return False
    consumed = {event.parent_sequence for event in engine.events if event.kind in {"ability_effect", "effect_cancelled"}}
    if require_unconsumed and consumed.intersection(ids):
        return False
    # A Copy watcher must really have been Scoocher when qualified. An older
    # already-started C power itself remains valid even if its borrowed power
    # later changes, so only the watcher qualification is checked here.
    original_copy = [item for item in engine.initial_racers if item["racer_id"] == c]
    if len(original_copy) != 1:
        return False
    power = original_copy[0]["copied_ability"]
    wanted = {job.trigger_sequence for job in pending if job.racer_id == c}
    observed = {}
    for event in engine.events:
        if event.kind == "copy_changed" and event.racer_id == c:
            power = event.ability
        if event.sequence in wanted:
            observed[event.sequence] = power
    return all(observed.get(sequence) == Ability.SCOOCHER.value for sequence in wanted)


def _leaptoad_gap_spectator(engine, actors):
    """Exact real first Leaptoad main, outside both closed movement bands.

    Native L8 cannot act while C5 returns through Baby6 and S traverses12..16.
    L's two real main-move bonuses and their four old notifications must have
    happened first. The existing queue validator rejects any still-old L job;
    the unchanged cohort reader then serves every captured S/C/B descendant.
    """
    s, c, b, other = actors
    if ({actor.ability for actor in actors} != {Ability.SCOOCHER, Ability.EGG, Ability.HUGE_BABY, Ability.LEAPTOAD}
            or c.ability != Ability.EGG or c.inherited_ability != Ability.COPY_CAT
            or b.ability != Ability.HUGE_BABY or other.ability != Ability.LEAPTOAD
            or (c.position, b.position, other.position) != (5, 6, 8) or c.tripped is not True
            or engine.turn_order[engine.turn_index] != other.racer_id or other.turns_taken != 1
            or other.tripped is not False or len(engine.turns) != 3
            or {turn.racer_id for turn in engine.turns} != {s.racer_id, c.racer_id, b.racer_id}):
        return False
    mains = [event for event in engine.events if event.kind == "move"
             and event.racer_id == other.racer_id and event.movement_kind == "main"]
    if len(mains) != 1:
        return False
    main = mains[0]
    if (type(main.sequence) is not int or type(main.parent_sequence) is not int
            or not 3 <= main.parent_sequence < main.sequence or main.sequence + 2 >= len(engine.events)):
        return False
    resolved = engine.events[main.parent_sequence - 1]
    if resolved.sequence != main.parent_sequence:
        return False
    locked, roll = engine.events[resolved.sequence - 2], engine.events[resolved.sequence - 3]
    return (_event(roll, "roll", other.racer_id, amount=6)
            and _event(locked, "roll_locked", other.racer_id, amount=6, parent_sequence=roll.sequence)
            and locked.sequence == roll.sequence + 1
            and _event(resolved, "main_move_resolved", other.racer_id, before=6, after=6,
                       parent_sequence=locked.sequence) and resolved.sequence == locked.sequence + 1
            and _event(main, "move", other.racer_id, before=0, after=8, amount=6,
                       movement_kind="main", parent_sequence=resolved.sequence, arrival_batch_id=main.arrival_batch_id)
            and type(main.arrival_batch_id) is int
            and _event(engine.events[main.sequence], "passing", other.racer_id, target_id=c.racer_id,
                       parent_sequence=main.sequence, arrival_batch_id=main.arrival_batch_id)
            and _event(engine.events[main.sequence + 1], "passing", other.racer_id, target_id=b.racer_id,
                       parent_sequence=main.sequence, arrival_batch_id=main.arrival_batch_id)
            and _event(engine.events[main.sequence + 2], "stop", other.racer_id, before=0, after=8,
                       parent_sequence=main.sequence, arrival_batch_id=main.arrival_batch_id))


def _coach_gap_spectator(engine, actors):
    """An original Coach strictly outside both closed movement bands.

    C only visits p/p+1 and S only visits12..16; a Coach at p+1 would be
    really displaced by Baby's resident batch, so that boundary is excluded.
    Coach only modifies a real main roll, absent from these authenticated
    FIFO bodies. Existing principal/source gates and full M still apply.
    """
    _, copy, _, spectator = actors
    return (spectator.ability == Ability.COACH
            and type(copy.position) is int and type(spectator.position) is int
            and copy.position + 1 < spectator.position < 12)


def _coach_tied_spectator(engine, actors):
    """One idle Coach at12; all real Copy choices obey the same U>=2 family.

    H12 may vanish or produce C, H13..15 always produce C; C produces HH
    or HHC according to its actual borrowed power when Baby qualifies it.
    U=#H+2#C+1[S13] never decreases, while an empty FIFO has U<=1.
    Qualified jobs survive Copy changes. The pending-source validator still
    excludes every unexecuted original Coach/main modifier notification.
    """
    s, c, b, spectator = actors
    return (spectator.ability == Ability.COACH and spectator.position == 12
            and type(spectator.position) is int and c.position + 1 < 12
            and s.position in range(12, 16)
            and all(isinstance(job, PendingEffect) for job in engine.pending_effects)
            and sum(1 if job.racer_id == s.racer_id else 2 for job in engine.pending_effects)
                + int(s.position == 13) >= 2)


def key(engine):
    """A full mechanical/control cut at S=12; only scores/FIFO are separate."""
    if (engine.track != WILD_WILDS or len(engine.racers) != 4 or engine.termination_reason is not None
            or engine.finish_order or engine._copy_locks or engine._refreshing_copies or engine._preparing
            or not engine._turn_in_progress or hasattr(engine, "_arrow_visits")
            or hasattr(engine, "_arrow_queue_visits") or getattr(engine, "_movement_depth", 0) != 0
            or getattr(engine, "_duel_offered_epochs", {})
            or type(getattr(engine, "_resolution_depth", 0)) is not int
            or getattr(engine, "_resolution_depth", 0) < 1
            or type(getattr(engine, "_arrival_counter", None)) is not int
            or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
            or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    scoocher, copy, baby, spectator = actors
    s, c, b, g = (actor.racer_id for actor in actors)
    p = copy.position
    leaptoad_gap = _leaptoad_gap_spectator(engine, actors)
    coach_gap = _coach_gap_spectator(engine, actors)
    coach_tied = _coach_tied_spectator(engine, actors)
    if (p not in RETURN_POSITIONS or p == 5 and copy.tripped is not True
            or (scoocher.position != 12 and not ((leaptoad_gap or coach_gap or coach_tied) and scoocher.position in range(12, 16)))
            or baby.position != p + 1
            or not (0 <= spectator.position <= p or leaptoad_gap or coach_gap or coach_tied)
            or spectator.position == p and spectator.ability in {Ability.BABA_YAGA, Ability.SUCKERFISH}
            or (engine._leaders() != (s,) and not (coach_tied and scoocher.position == 12
                and set(engine._leaders()) == {s, g}))
            or (copy.copied_target, copy.copied_ability) != (s, Ability.SCOOCHER)):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE or not actor.pre_race_resolved
                or actor is not baby and actor.inherited_ability is not None
                    and not (actor is copy and actor.ability in {Ability.EGG, Ability.TWIN}
                             and actor.inherited_ability == Ability.COPY_CAT)
                or actor.prediction is not None and (type(actor.prediction) is not int or actor.prediction not in engine.racers)
                or actor.finish_place is not None or actor.finish_places
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or actor is not copy and (actor.copied_target is not None or actor.copied_ability is not None)):
            return None
    next_player = getattr(engine, "next_turn_racer_id", None)
    prediction = getattr(engine, "_genius_prediction", None)
    if (next_player is not None and (type(next_player) is not int or next_player not in engine.racers)
            or prediction is not None and (type(prediction) is not int or not 1 <= prediction <= 6)):
        return None
    if not _queue_valid(engine, tuple(engine.pending_effects), s, c, b):
        return None
    roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                   for rid, actor in sorted(engine.racers.items()))
    initial = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger}, sort_keys=True)
    state = (engine.track, engine._state_key(include_scores=False, include_pending=False, include_decisions=not coach_tied), roster,
             tuple(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
             tuple(engine.egg_drawn), tuple(engine.previous_winners), initial, engine._score_start,
             engine.initial_code_fingerprint, engine.max_events, getattr(engine, "_resolution_depth", 0))
    if baby.ability != Ability.HUGE_BABY:
        inheritance = static_inheritance_evidence(engine, baby, Ability.HUGE_BABY)
        if inheritance is None:
            return None
        state = (*state[:-1], inheritance, state[-1])
    source = _copy_source(engine, copy)
    if source is None:
        return None
    if source:
        # Existing Baby-alias shape and the final depth slot remain unchanged.
        state = (*state[:-1], ("static_copy", c, source), state[-1])
    try:
        hash(state)
    except TypeError:
        return None
    return state


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    original_pending: tuple[PendingEffect, ...]
    score_before: ScoreSnapshot
    event_prefix: tuple
    decision_prefix: tuple
    arrival_counter: int
    initial_scoocher_phase: int = 12


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and actual != state:
        raise ValueError("not a certified growing Star/Baby queue cut")
    return Snapshot(actual, len(engine.events), tuple(engine.pending_effects), engine.score_ledger.snapshot(),
                    tuple(engine.events), tuple(engine.decisions), engine._arrival_counter,
                    _actors(engine)[0].position)


def _event(event, kind, actor, **values):
    """All unlisted event fields must retain their real default None value."""
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _batch(event):
    return type(event.arrival_batch_id) is int and event.arrival_batch_id > 0


def _queued(event, actor, source, power):
    if not _event(event, "effect_queued", actor, source_id=source,
                  ability=Ability.SCOOCHER.value, parent_sequence=power.sequence):
        return None
    return PendingEffect(actor, source, power.sequence, event.sequence)


def _score(event, actor, stop, balances):
    before = balances[actor]
    if not _event(event, "score", actor, before=before, after=before + 1, amount=1,
                  parent_sequence=stop.sequence, reason="star"):
        return False
    balances[actor] += 1
    return True


def _job(events, pending, phase, balances, s, c, b, order, return_position=1):
    """Parse one complete real job; no projected choice or queue mutation."""
    is_s = pending.racer_id == s
    length = (5 if phase == 12 else 6 if phase == 15 else 4) if is_s else 7 + int(return_position == 1)
    period = events[:length]
    if len(period) != length:
        return None
    power = period[0]
    actor = s if is_s else c
    if not _event(power, "ability_effect", actor, source_id=actor, ability=Ability.SCOOCHER.value,
                  amount=1, parent_sequence=pending.queued_sequence):
        return None
    if is_s:
        notification, move, stop = period[1:4]
        child = _queued(notification, c, s, power)
        if (child is None or not _batch(move)
                or not _event(move, "move", s, source_id=s, ability=Ability.SCOOCHER.value,
                              before=phase, after=phase + 1, amount=1, movement_kind="ability",
                              parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
                or not _event(stop, "stop", s, before=phase, after=phase + 1,
                              parent_sequence=move.sequence, arrival_batch_id=move.arrival_batch_id)):
            return None
        if phase == 12 and not _score(period[4], s, stop, balances):
            return None
        if phase == 15:
            arrow, back = period[4:]
            if (not _batch(arrow) or arrow.arrival_batch_id <= move.arrival_batch_id
                    or not _event(arrow, "move", s, before=16, after=12, amount=-4,
                                  movement_kind="track", parent_sequence=stop.sequence,
                                  arrival_batch_id=arrow.arrival_batch_id)
                    or not _event(back, "stop", s, before=16, after=12,
                                  parent_sequence=arrow.sequence, arrival_batch_id=arrow.arrival_batch_id)):
                return None
        return length, (child,), 12 if phase == 15 else phase + 1
    first, move, baby_power, q1, q2, stop = period[1:7]
    child = _queued(first, s, c, power)
    notifications = tuple(_queued(event, target, b, baby_power) for event, target in zip((q1, q2), order))
    if (child is None or None in notifications or not _batch(move) or not _batch(stop)
            or stop.arrival_batch_id <= move.arrival_batch_id
            or not _event(move, "move", c, source_id=c, ability=Ability.SCOOCHER.value,
                          before=return_position, after=return_position + 1, amount=1, movement_kind="ability",
                          parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
            or not _event(baby_power, "ability_effect", b, source_id=b, ability=Ability.HUGE_BABY.value,
                          parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
            or not _event(stop, "stop", c, before=return_position + 1, after=return_position,
                          parent_sequence=baby_power.sequence, arrival_batch_id=stop.arrival_batch_id)
            or return_position == 1 and not _score(period[7], c, stop, balances)):
        return None
    return length, (child, *notifications), phase


class _CoachTieReader:
    """Real choice/refresh suffix around the existing two FIFO job bodies."""
    def __init__(self, engine, snapshot, actors, balances):
        self.engine, self.actors, self.balances = engine, actors, balances
        self.s, self.c, self.b, self.g = (a.racer_id for a in actors)
        self.p = actors[1].position
        self.target = self.s  # key/capture require the actual S binding.
        self.decision = len(snapshot.decision_prefix)
        self.choice_ids, self.change_ids = [], []

    def observation(self, phase):
        positions = {self.s: phase, self.c: self.p, self.b: self.p + 1, self.g: 12}
        public = []
        by_id = {a.racer_id: a for a in self.actors}
        for rid in self.engine.turn_order:
            a = by_id[rid]
            ability = (Ability.SCOOCHER if self.target == self.s else Ability.COACH) if rid == self.c else self.engine._ability(a)
            public.append(PublicRacer(rid, a.name, positions[rid], "active", ability.value,
                a.tripped, None, self.target if rid == self.c else None))
        return Observation(self.engine.turn_order[self.engine.turn_index], len(self.engine.turns) + 1,
            WILD_WILDS.name, WILD_WILDS.finish_position, WILD_WILDS.second_corner_position,
            tuple(sorted(WILD_WILDS.trip_positions)), tuple(public), (), None, None,
            score_totals=tuple((rid, self.balances[rid]) for rid in self.engine.score_ledger.player_ids))

    def refresh(self, events, phase):
        old, used = self.target, 0
        if phase == 12:
            targets = tuple(rid for rid in self.engine._clockwise_order() if rid in {self.s, self.g})
            legal = (Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, self.c, rid) for rid in targets))
            if self.decision >= len(self.engine.decisions) or not events:
                return None
            record = self.engine.decisions[self.decision]
            expected = DecisionRequest(self.decision + 1, self.c, DecisionWindow.COPY_CHOICE,
                                       self.observation(phase), legal)
            # Canonical dataclass values reject bool/int aliases in public data.
            if (type(record.request) is not DecisionRequest or not any(
                    json.dumps(asdict(record.action), sort_keys=True) == json.dumps(asdict(action), sort_keys=True) for action in legal)
                    or json.dumps(asdict(record.request), sort_keys=True) != json.dumps(asdict(expected), sort_keys=True)
                    or type(record.event_sequence) is not int):
                return None
            event = events[0]
            if (not _event(event, "decision", self.c, source_id=record.action.source_id,
                          target_id=record.action.target_id, action_kind=record.action.kind.value,
                          decision_id=expected.decision_id) or record.event_sequence != event.sequence):
                return None
            used = 1
            self.decision += 1
            self.choice_ids.append(event.sequence)
            if record.action.kind != ActionKind.KEEP:
                self.target = record.action.target_id
        else:
            self.target = self.s
        if self.target != old:
            if len(events) <= used or not _event(events[used], "copy_changed", self.c,
                    source_id=self.c, target_id=self.target,
                    ability=(Ability.SCOOCHER if self.target == self.s else Ability.COACH).value):
                return None
            self.change_ids.append(events[used].sequence)
            used += 1
        return used

    def job(self, events, pending, phase, order):
        # Every already-qualified C job still executes Scoocher, even if C now
        # borrows Coach. Only actual watcher qualification can omit a child.
        if self.target == self.s:
            result = _job(events, pending, phase, self.balances, self.s, self.c, self.b, order, self.p)
        elif phase == 12 and pending.racer_id == self.s:
            if len(events) < 4:
                return None
            power, move, stop, score = events[:4]
            if (not _event(power, "ability_effect", self.s, source_id=self.s, ability=Ability.SCOOCHER.value,
                           amount=1, parent_sequence=pending.queued_sequence)
                    or not _batch(move) or not _event(move, "move", self.s, source_id=self.s,
                        ability=Ability.SCOOCHER.value, before=12, after=13, amount=1, movement_kind="ability",
                        parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
                    or not _event(stop, "stop", self.s, before=12, after=13,
                        parent_sequence=move.sequence, arrival_batch_id=move.arrival_batch_id)
                    or not _score(score, self.s, stop, self.balances)):
                return None
            result = (4, (), 13)
        elif phase == 12 and pending.racer_id == self.c:
            length = 6 + int(self.p == 1)
            if len(events) < length:
                return None
            power, first, move, baby, notification, stop = events[:6]
            h1, h2 = _queued(first, self.s, self.c, power), _queued(notification, self.s, self.b, baby)
            if (h1 is None or h2 is None or not _event(power, "ability_effect", self.c,
                    source_id=self.c, ability=Ability.SCOOCHER.value, amount=1, parent_sequence=pending.queued_sequence)
                    or not _batch(move) or not _batch(stop)
                    or not _event(move, "move", self.c, source_id=self.c, ability=Ability.SCOOCHER.value,
                        before=self.p, after=self.p+1, amount=1, movement_kind="ability",
                        parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
                    or not _event(baby, "ability_effect", self.b, source_id=self.b, ability=Ability.HUGE_BABY.value,
                        parent_sequence=power.sequence, arrival_batch_id=move.arrival_batch_id)
                    or not _event(stop, "stop", self.c, before=self.p+1, after=self.p,
                        parent_sequence=baby.sequence, arrival_batch_id=stop.arrival_batch_id)
                    or self.p == 1 and not _score(events[6], self.c, stop, self.balances)):
                return None
            result = (length, (h1, h2), 12)
        else:
            return None
        if result is None:
            return None
        length, children, after = result
        refresh_length = self.refresh(events[length:], after)
        if refresh_length is None:
            return None
        return length + refresh_length, children, after


def completed(engine, snapshot):
    """Authenticate actual FIFO expansion, original consumption and Star drift."""
    if (key(engine) != snapshot.state or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:len(snapshot.decision_prefix)]) != snapshot.decision_prefix):
        return False
    scoocher, copy, baby, spectator = _actors(engine)
    s, c, b = scoocher.racer_id, copy.racer_id, baby.racer_id
    if (type(snapshot.initial_scoocher_phase) is not int
            or snapshot.initial_scoocher_phase not in range(12, 16)
            or snapshot.initial_scoocher_phase != scoocher.position):
        return False
    if not _queue_valid(engine, snapshot.original_pending, s, c, b, require_unconsumed=False):
        return False
    score = engine.score_ledger.snapshot()
    old = snapshot.score_before
    if score.player_ids != old.player_ids or score.entries[:len(old.entries)] != old.entries:
        return False
    coach_tied = _coach_tied_spectator(engine, (scoocher, copy, baby, spectator))
    if coach_tied:
        # The generic detector intentionally forbids decisions. This branch
        # authenticates each actual public choice below, while keeping exactly
        # its original unconditional official Star ledger requirements.
        added = score.entries[len(old.entries):]
        if not added or any(entry.source != "track" or entry.reason != "star" or entry.delta != 1
                or entry.race_index != engine.race_index or entry.phase != "during_race" for entry in added):
            return False
    elif not engine._star_cycle_completed((snapshot.event_start, len(old.entries), dict(old.totals))):
        return False
    events = engine.events[snapshot.event_start:]
    if any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(events)):
        return False
    queue, balances, phase = deque(snapshot.original_pending), dict(old.totals), snapshot.initial_scoocher_phase
    originals = {job.queued_sequence for job in snapshot.original_pending}
    cursor, executed, counts, score_events = 0, [], {s: 0, c: 0}, []
    arrival_counter = snapshot.arrival_counter
    order = tuple(rid for rid in engine._clockwise_order() if rid in {s, c})
    coach_reader = _CoachTieReader(engine, snapshot, (scoocher, copy, baby, spectator), balances) if coach_tied else None
    while cursor < len(events):
        if not queue:
            return False
        pending = queue.popleft()
        result = (coach_reader.job(events[cursor:], pending, phase, order) if coach_reader is not None else
                  _job(events[cursor:], pending, phase, balances, s, c, b, order, copy.position))
        if result is None:
            return False
        length, children, phase = result
        batches = [event.arrival_batch_id for event in events[cursor:cursor + length]
                   if event.kind == "move" or event.kind == "stop" and event.racer_id == c]
        # Each real arrival batch increments exactly once, even Baby's put
        # without a Move event. Do not accept gaps or a partial last arrival.
        unique_batches = list(dict.fromkeys(batches))
        if unique_batches != list(range(arrival_counter + 1, arrival_counter + 1 + len(unique_batches))):
            return False
        arrival_counter += len(unique_batches)
        score_events.extend(event for event in events[cursor:cursor + length] if event.kind == "score")
        cursor += length
        executed.append(pending.queued_sequence)
        counts[pending.racer_id] += 1
        queue.extend(children)
        if coach_reader is not None and sum(1 if job.racer_id == s else 2 for job in queue) + int(phase == 13) < 2:
            return False
    if coach_reader is not None and (coach_reader.decision != len(engine.decisions) or coach_reader.target != copy.copied_target):
        return False
    current_ids = {job.queued_sequence for job in engine.pending_effects}
    if (phase != snapshot.initial_scoocher_phase or counts[s] < 4 or counts[s] % 4 or counts[c] < 1
            or not originals.issubset(executed) or originals.intersection(current_ids)
            or any(sequence <= snapshot.event_start for sequence in current_ids)
            or tuple(queue) != tuple(engine.pending_effects) or balances != engine.score_ledger.totals()
            or arrival_counter != engine._arrival_counter):
        return False
    added = score.entries[len(old.entries):]
    if (len(added) != len(score_events)
            or any((entry.sequence, entry.player_id, entry.event_sequence) != (
                       len(old.entries) + index + 1, event.racer_id, event.sequence)
                   for index, (entry, event) in enumerate(zip(added, score_events)))):
        return False
    races = {index: dict(totals) for index, totals in old.race_totals}
    for entry in added:
        races.setdefault(entry.race_index, {rid: 0 for rid in old.player_ids})[entry.player_id] += entry.delta
    expected_races = tuple((index, tuple((rid, races[index][rid]) for rid, _ in totals))
                           for index, totals in old.race_totals)
    if score.race_totals != expected_races:
        return False
    evidence = {"original_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
            "executed_pending_ids": executed, "new_pending_ids": [job.queued_sequence for job in engine.pending_effects],
            "scoocher_jobs": counts[s], "copy_jobs": counts[c], "scoocher_complete_orbits": counts[s] // 4,
            "star_point_increments": {str(rid): balances[rid] - total for rid, total in old.totals},
            "queue_before": len(snapshot.original_pending), "queue_after": len(engine.pending_effects),
            "baby_notification_order": list(order), "owner_resolution_depth": snapshot.state[-1],
            "genius_position_at_cut": spectator.position,
            "genius_prediction_at_cut": getattr(engine, "_genius_prediction", None),
            "next_turn_racer_id_at_cut": getattr(engine, "next_turn_racer_id", None),
            "proof": "every original FIFO job executed; exact nonempty substitutions and full mechanical phase restored"}
    # Preserve byte-identical certificate metadata for the old proven Genius
    # p=1 domain. New deployments name the actual spectator and geometry.
    if spectator.ability != Ability.GENIUS or copy.position != 1:
        evidence.update(copy_return_position=copy.position, baby_position=baby.position,
                        spectator_id=spectator.racer_id, spectator_ability=spectator.ability.value,
                        spectator_position=spectator.position)
    if baby.ability != Ability.HUGE_BABY:
        evidence["static_baby"] = {"racer_id": baby.racer_id, "original_ability": baby.ability.value,
                                   "inherited_ability": baby.inherited_ability.value,
                                   "acquisition": static_inheritance_evidence(engine, baby, Ability.HUGE_BABY)}
    source = _copy_source(engine, copy)
    if source:
        evidence["static_copy"] = _copy_metadata(copy, source)
    if (_leaptoad_gap_spectator(engine, (scoocher, copy, baby, spectator))
            or _coach_gap_spectator(engine, (scoocher, copy, baby, spectator))
            or coach_reader is not None):
        evidence["initial_scoocher_phase"] = snapshot.initial_scoocher_phase
    if coach_reader is not None:
        evidence.update(coach_tied_leader_choice_profile=True,
                        proof="every original FIFO job executed; all real choices parsed; U>=2 forbids empty FIFO; full M and first complete phase restored",
                        actual_copy_choice_event_ids=coach_reader.choice_ids,
                        actual_copy_change_event_ids=coach_reader.change_ids,
                        invariant="U=#S+2#C+1[S13]>=2, nondecreasing for all legal Copy branches")
    return evidence
