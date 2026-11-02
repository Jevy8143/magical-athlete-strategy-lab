"""A mixed, input-free Romantic22 and Scoocher11/Baby12 FIFO proof.

An original Third Wheel remains with the original Romantic at22. R really
returns22->24->22, appending [Scoocher-from-R,R]. S really returns11->12->11
twice through Baby and Arrow11, retaining the existing local Arrow certificate,
and appending two Baby notifications. Both substitutions are nonempty.

The captured cohort must contain one R and at least one S. Every original
qualification must execute its complete exact body. Only its entirely fresh
descendants may stop, after complete mechanical/control/ledger return. This
module never executes or removes an effect, changes a score, or skips input.
"""

from collections import Counter, deque
from dataclasses import dataclass, fields
import json

from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .static_inheritance import static_inheritance_evidence
from .track import WILD_WILDS


REASON = "romantic_baby_arrow_mixed_queue_cycle_proved"


def _event(event, kind, actor, **values):
    return isinstance(event, RaceEvent) and event == RaceEvent(event.sequence, kind, actor, **values)


def _actors(engine):
    actors = tuple(engine.racers.values())
    if (len(actors) != 4 or any(not isinstance(actor.ability, Ability) for actor in actors)
            or len({actor.ability for actor in actors}) != 4):
        return None
    roles = {actor.ability: actor for actor in actors}
    fixed = {Ability.ROMANTIC, Ability.THIRD_WHEEL, Ability.HUGE_BABY}
    if not fixed.issubset(roles):
        return None
    r, p, b = (roles[power] for power in (Ability.ROMANTIC, Ability.THIRD_WHEEL, Ability.HUGE_BABY))
    s = next(actor for actor in actors if actor not in (r, p, b))
    if (s.ability not in {Ability.SCOOCHER, Ability.EGG, Ability.TWIN}
            or engine._ability(s) != Ability.SCOOCHER
            or any(actor.inherited_ability is not None for actor in (r, p, b))):
        return None
    if static_inheritance_evidence(engine, s, Ability.SCOOCHER) is None:
        return None
    return r, s, b, p


def _queue_valid(engine, pending, actors, *, require_unconsumed=True):
    r, s, b, p = (actor.racer_id for actor in actors)
    if not pending or any(not isinstance(job, PendingEffect) for job in pending):
        return False
    ids = [job.queued_sequence for job in pending]
    if ids != sorted(set(ids)):
        return False
    counts = Counter(job.racer_id for job in pending)
    # In particular a lone R at the original244 cut cannot capture early.
    if set(counts) != {r, s} or counts[r] != 1 or counts[s] < 1:
        return False
    for job in pending:
        if (any(type(value) is not int for value in (job.racer_id, job.source_id, job.steps,
                                                    job.trigger_sequence, job.queued_sequence))
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)):
            return False
        if job.racer_id == r:
            if type(job.target_id) is not int or (job.source_id, job.kind, job.steps, job.ability, job.target_id) != (
                    r, "move", 2, Ability.ROMANTIC.value, r):
                return False
        elif (job.source_id not in {r, b}
              or (job.kind, job.steps, job.ability, job.target_id) != ("move", 1, Ability.SCOOCHER.value, None)):
            return False
    initial = {item["racer_id"]: item for item in engine.initial_racers}
    if set(initial) != set(engine.racers):
        return False
    positions = {rid: item["position"] for rid, item in initial.items()}
    needs = {job.trigger_sequence for job in pending}
    for job in pending:
        if job.racer_id == s and job.source_id == r:
            power = engine.events[job.trigger_sequence - 1]
            if type(power.parent_sequence) is not int or not 1 <= power.parent_sequence < power.sequence:
                return False
            header = engine.events[power.parent_sequence - 1]
            if type(header.parent_sequence) is not int or not 1 <= header.parent_sequence < header.sequence:
                return False
            needs.add(header.parent_sequence)
    history, stops, consumed = {}, Counter(), set()
    live, born = deque(), set()
    grants = [event.sequence for event in engine.events if event.kind == "inherit" and event.racer_id == s]
    grant = max(grants, default=0)
    alias = engine.racers[s].ability != Ability.SCOOCHER
    numeric = ("sequence", "racer_id", "source_id", "target_id", "before", "after", "amount",
               "parent_sequence", "decision_id", "arrival_batch_id")
    for index, event in enumerate(engine.events, 1):
        if (not isinstance(event, RaceEvent) or event.sequence != index
                or any(getattr(event, name) is not None and type(getattr(event, name)) is not int for name in numeric)):
            return False
        if event.kind in {"move", "warp", "stop"}:
            if event.racer_id not in positions:
                return False
            positions[event.racer_id] = event.after
        if event.kind in {"finish", "eliminate", "race_end"}:
            return False
        if event.kind == "inherit" and (not alias or event.racer_id != s):
            return False
        if event.kind in {"ability_effect", "effect_cancelled"}:
            consumed.add(event.parent_sequence)
            if event.parent_sequence in born:
                if not live or live.popleft() != event.parent_sequence:
                    return False
        if event.kind == "effect_queued":
            born.add(event.sequence)
            live.append(event.sequence)
        if event.kind == "loop_stopped" and event.reason not in {
                "forced_arrow_cycle_with_deferred_queue_proved", "complete_arrow_state_repeated"}:
            # Earlier real queue certificates retire their fresh children.
            # Those IDs must never masquerade as still-unconsumed input here.
            consumed.update(live)
            live.clear()
        if event.kind == "stop":
            stops[event.arrival_batch_id] += 1
        if event.sequence in needs:
            history[event.sequence] = dict(positions)
    if positions != {rid: actor.position for rid, actor in engine.racers.items()}:
        return False
    if require_unconsumed and tuple(live) != tuple(ids):
        return False

    def genuine_r_stop(trigger):
        old = history.get(trigger.sequence)
        return (old is not None
                and _event(trigger, "stop", r, before=trigger.before, after=22,
                           parent_sequence=trigger.parent_sequence, arrival_batch_id=trigger.arrival_batch_id)
                and type(trigger.before) is int and type(trigger.parent_sequence) is int
                and 1 <= trigger.parent_sequence < trigger.sequence
                and type(trigger.arrival_batch_id) is int and stops[trigger.arrival_batch_id] == 1
                and old[r] == 22 and old[p] == 22 and sum(value == 22 for value in old.values()) == 2)

    for job in pending:
        if (require_unconsumed and job.queued_sequence in consumed
                or job.racer_id == s and job.trigger_sequence <= grant):
            return False
        trigger, queued = (engine.events[index - 1] for index in (job.trigger_sequence, job.queued_sequence))
        if job.racer_id == r:
            if (not genuine_r_stop(trigger)
                    or not _event(queued, "effect_queued", r, source_id=r, target_id=r,
                                  ability=Ability.ROMANTIC.value, parent_sequence=trigger.sequence)):
                return False
        else:
            if (not _event(queued, "effect_queued", s, source_id=job.source_id,
                           ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence)
                    or type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence):
                return False
            parent = engine.events[trigger.parent_sequence - 1]
            if job.source_id == r:
                if (not _event(trigger, "ability_effect", r, source_id=r, ability=Ability.ROMANTIC.value,
                               amount=2, parent_sequence=parent.sequence)
                        or not _event(parent, "effect_queued", r, source_id=r, target_id=r,
                                      ability=Ability.ROMANTIC.value, parent_sequence=parent.parent_sequence)
                        or not genuine_r_stop(engine.events[parent.parent_sequence - 1])):
                    return False
            elif (not _event(trigger, "ability_effect", b, source_id=b, ability=Ability.HUGE_BABY.value,
                             parent_sequence=parent.sequence, arrival_batch_id=trigger.arrival_batch_id)
                  or type(trigger.arrival_batch_id) is not int or history[trigger.sequence][b] != 12
                  or history[trigger.sequence][s] != 12
                  or not ((parent.kind == "ability_effect" and parent.racer_id == s
                           and parent.source_id == s and parent.ability == Ability.SCOOCHER.value and parent.amount == 1)
                          or (parent.kind == "stop" and parent.racer_id == s and parent.after == 11))):
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
    roles = {actor.ability for actor in engine.racers.values()}
    if not {Ability.ROMANTIC, Ability.THIRD_WHEEL, Ability.HUGE_BABY}.issubset(roles):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    r, s, b, p = actors
    if (r.position, s.position, b.position, p.position) != (22, 11, 12, 22):
        return None
    for actor in actors:
        if (actor.status != RacerStatus.ACTIVE or actor.pre_race_resolved is not True
                or actor.copied_target is not None or actor.copied_ability is not None or actor.prediction is not None
                or actor.finish_place is not None or actor.finish_places
                or type(actor.racer_id) is not int or type(actor.position) is not int
                or type(actor.tripped) is not bool or type(actor.turns_taken) is not int or actor.turns_taken < 0):
            return None
    pending = tuple(engine.pending_effects)
    if not _queue_valid(engine, pending, actors):
        return None
    inheritance = static_inheritance_evidence(engine, s, Ability.SCOOCHER)
    roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                   for rid, actor in sorted(engine.racers.items()))
    state = (engine.track, engine._state_key(include_pending=False), roster, engine.score_ledger.snapshot(),
             len(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
             tuple(engine.egg_drawn), tuple(engine.previous_winners), inheritance,
             json.dumps(engine.initial_racers, sort_keys=True, separators=(",", ":"), allow_nan=False),
             getattr(engine, "_genius_prediction", None), getattr(engine, "next_turn_racer_id", None),
             engine._resolution_depth, tuple(sorted(getattr(engine, "_duel_offered_epochs", {}).items())))
    try:
        hash(state)
    except TypeError:
        return None
    return state


@dataclass(frozen=True, slots=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    original_pending: tuple[PendingEffect, ...]
    event_prefix: tuple[RaceEvent, ...]
    decision_prefix: tuple


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual:
        raise ValueError("not a certified mixed Romantic/Baby/Arrow cut")
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    tuple(engine.events), tuple(engine.decisions))


def _r_body(period, original, r, s):
    if len(period) != 7:
        return None
    power, notify, move, stop, arrow, back, queued = period
    if (not _event(power, "ability_effect", r, source_id=r, ability=Ability.ROMANTIC.value,
                   amount=2, parent_sequence=original.queued_sequence)
            or not _event(notify, "effect_queued", s, source_id=r, ability=Ability.SCOOCHER.value,
                          parent_sequence=power.sequence)
            or type(move.arrival_batch_id) is not int or move.arrival_batch_id < 1
            or type(arrow.arrival_batch_id) is not int or arrow.arrival_batch_id <= move.arrival_batch_id
            or not _event(move, "move", r, source_id=r, ability=Ability.ROMANTIC.value,
                          before=22, after=24, amount=2, movement_kind="ability", parent_sequence=power.sequence,
                          arrival_batch_id=move.arrival_batch_id)
            or not _event(stop, "stop", r, before=22, after=24, parent_sequence=move.sequence,
                          arrival_batch_id=move.arrival_batch_id)
            or not _event(arrow, "move", r, before=24, after=22, amount=-2, movement_kind="track",
                          parent_sequence=stop.sequence, arrival_batch_id=arrow.arrival_batch_id)
            or not _event(back, "stop", r, before=24, after=22, parent_sequence=arrow.sequence,
                          arrival_batch_id=arrow.arrival_batch_id)
            or not _event(queued, "effect_queued", r, source_id=r, target_id=r,
                          ability=Ability.ROMANTIC.value, parent_sequence=back.sequence)):
        return None
    return (PendingEffect(s, r, power.sequence, notify.sequence),
            PendingEffect(r, r, back.sequence, queued.sequence, "move", 2, Ability.ROMANTIC.value, r))


def _s_body(period, original, s, b):
    if len(period) != 10:
        return None
    power, move, baby, notify, stop, arrow, baby_again, notify_again, back, loop = period
    batches = (move.arrival_batch_id, stop.arrival_batch_id, arrow.arrival_batch_id, back.arrival_batch_id)
    if (any(type(value) is not int or value < 1 for value in batches) or list(batches) != sorted(set(batches))
            or not _event(power, "ability_effect", s, source_id=s, ability=Ability.SCOOCHER.value,
                          amount=1, parent_sequence=original.queued_sequence)
            or not _event(move, "move", s, source_id=s, ability=Ability.SCOOCHER.value,
                          before=11, after=12, amount=1, movement_kind="ability", parent_sequence=power.sequence,
                          arrival_batch_id=move.arrival_batch_id)
            or not _event(arrow, "move", s, before=11, after=12, amount=1, movement_kind="track",
                          parent_sequence=stop.sequence, arrival_batch_id=arrow.arrival_batch_id)
            or not _event(loop, "loop_stopped", s, reason="forced_arrow_cycle_with_deferred_queue_proved",
                          parent_sequence=back.sequence)):
        return None
    derived = []
    for effect, queued, landing, parent, arrival in (
            (baby, notify, stop, power.sequence, move.arrival_batch_id),
            (baby_again, notify_again, back, stop.sequence, arrow.arrival_batch_id)):
        if (not _event(effect, "ability_effect", b, source_id=b, ability=Ability.HUGE_BABY.value,
                       parent_sequence=parent, arrival_batch_id=arrival)
                or not _event(queued, "effect_queued", s, source_id=b, ability=Ability.SCOOCHER.value,
                              parent_sequence=effect.sequence)
                or not _event(landing, "stop", s, before=12, after=11, parent_sequence=effect.sequence,
                              arrival_batch_id=landing.arrival_batch_id)):
            return None
        derived.append(PendingEffect(s, b, effect.sequence, queued.sequence))
    return tuple(derived)


def completed(engine, snapshot):
    if (not isinstance(snapshot, Snapshot) or key(engine) != snapshot.state
            or tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions) != snapshot.decision_prefix):
        return False
    actors = _actors(engine)
    if actors is None or not _queue_valid(engine, snapshot.original_pending, actors, require_unconsumed=False):
        return False
    r, s, b, _ = actors
    period, cursor, derived, counts = engine.events[snapshot.event_start:], 0, [], Counter()
    if any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(period)):
        return False
    for original in snapshot.original_pending:
        is_r = original.racer_id == r.racer_id
        size = 7 if is_r else 10
        body = period[cursor:cursor + size]
        children = (_r_body(body, original, r.racer_id, s.racer_id) if is_r
                    else _s_body(body, original, s.racer_id, b.racer_id))
        if children is None:
            return False
        counts["R" if is_r else "S"] += 1
        derived.extend(children)
        cursor += size
    original_ids = {job.queued_sequence for job in snapshot.original_pending}
    if (cursor != len(period) or counts["R"] != 1 or counts["S"] < 1
            or tuple(engine.pending_effects) != tuple(derived)
            or len(derived) != 2 * len(snapshot.original_pending)
            or original_ids.intersection(job.queued_sequence for job in derived)
            or any(job.queued_sequence <= snapshot.event_start for job in derived)):
        return False
    result = {"original_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
              "executed_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
              "new_pending_ids": [job.queued_sequence for job in derived],
              "queue_before": len(snapshot.original_pending), "queue_after": len(derived),
              "job_counts": dict(counts), "mechanical_state_returned": True, "scores_unchanged": True,
              "all_original_jobs_executed": True, "only_fresh_pending_descendants_stopped": True,
              "romantic_id": r.racer_id, "scoocher_id": s.racer_id, "baby_id": b.racer_id,
              "original_third_wheel_id": actors[3].racer_id,
              "proof": "exact mixed FIFO R->S_R,R and S->S_B,S_B; one R and real S in the complete original cohort"}
    if s.ability != Ability.SCOOCHER:
        result["static_scoocher"] = {"racer_id": s.racer_id, "original_ability": s.ability.value,
                                     "inherited_ability": s.inherited_ability.value,
                                     "acquisition": static_inheritance_evidence(engine, s, Ability.SCOOCHER)}
    return result
