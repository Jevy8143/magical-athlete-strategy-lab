"""A forced, zero-score Romantic FIFO closure on official Wild22.

Original Romantic R and Copy C share22, and C's only other leading racer is
R. Each actual Romantic +2 returns22->24->22 and queues the same two genuine
clockwise Romantic jobs. Two static, prefix-passive spectators stay outside
22/23/24. Thus every queue job has a nonempty, exactly verified replacement;
neither a choice, dice, score, nor a racer/control field can change.

The whole pending queue grows, so generic complete-queue repetition cannot
detect this case. All N original jobs must actually run before only their
2N fresh descendants may be stopped. Provenance IDs remain authenticated;
they are not substituted for a complete mechanical/control-state proof.
"""

from dataclasses import dataclass, fields

from .decisions import Action, ActionKind, DecisionWindow
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .static_inheritance import resolved_inert_twin_evidence
from .track import WILD_WILDS


REASON = "homogeneous_romantic_copy_queue_cycle_proved"


def _static_copy_source(engine, actor):
    """Authenticate an original Copy or its real resolved Egg/Twin chain."""
    if actor.ability == Ability.COPY_CAT:
        return actor.inherited_ability is None
    if actor.ability not in {Ability.EGG, Ability.TWIN} or actor.inherited_ability != Ability.COPY_CAT:
        return False
    chain = [event for event in engine.events if event.kind == "inherit" and event.racer_id == actor.racer_id]
    if not chain:
        return False
    choices = {record.event_sequence: record for record in engine.decisions}
    current, visited = actor.ability, set()
    for event in chain:
        if current not in {Ability.EGG, Ability.TWIN} or current in visited:
            return False
        visited.add(current)
        record = choices.get(event.parent_sequence)
        try:
            choice = Action(ActionKind.EGG_CHOOSE if current == Ability.EGG else ActionKind.TWIN_CHOOSE,
                            actor.racer_id, actor.racer_id, value=event.ability)
            following = Ability(event.ability)
        except (ValueError, TypeError):
            return False
        if (record is None or event.source_id != actor.racer_id or event.sequence != record.event_sequence + 1
                or record.request.window != DecisionWindow.BEFORE_RACE
                or record.request.player_id != actor.racer_id or record.action != choice
                or choice not in record.request.legal_actions):
            return False
        current = following
    return current == Ability.COPY_CAT


def _actors(engine):
    if len(engine.racers) != 4 or len({actor.ability for actor in engine.racers.values()}) != 4:
        return None
    romantic = [actor for actor in engine.racers.values()
                if actor.status == RacerStatus.ACTIVE and actor.ability == Ability.ROMANTIC]
    effective = [actor for actor in engine.racers.values()
                 if actor.status == RacerStatus.ACTIVE and engine._ability(actor) == Ability.ROMANTIC]
    if len(romantic) != 1 or len(effective) != 2 or romantic[0] not in effective:
        return None
    anchor = romantic[0]
    copy = next(actor for actor in effective if actor is not anchor)
    if (anchor.inherited_ability is not None or anchor.copied_target is not None or anchor.copied_ability is not None
            or anchor.prediction is not None or not engine._is_copy(copy) or not _static_copy_source(engine, copy)
            or (copy.copied_target, copy.copied_ability) != (anchor.racer_id, Ability.ROMANTIC)):
        return None
    return anchor, copy


def _event(event, kind, actor, **values):
    return event == RaceEvent(event.sequence, kind, actor, **values)


def _queued(event, actor, stop):
    if not _event(event, "effect_queued", actor, source_id=actor, target_id=actor,
                  ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence):
        return None
    return PendingEffect(actor, actor, stop.sequence, event.sequence, "move", 2, Ability.ROMANTIC.value, actor)


def _queue_valid(engine, pending, actors):
    ids = [job.queued_sequence for job in pending]
    if not pending or ids != sorted(set(ids)):
        return False
    for job in pending:
        if ((job.racer_id, job.source_id, job.kind, job.steps, job.ability, job.target_id) != (
                job.racer_id, job.racer_id, "move", 2, Ability.ROMANTIC.value, job.racer_id)
                or job.racer_id not in actors or type(job.steps) is not int
                or type(job.trigger_sequence) is not int or type(job.queued_sequence) is not int
                or not 1 <= job.trigger_sequence < job.queued_sequence <= len(engine.events)):
            return False
        trigger = engine.events[job.trigger_sequence - 1]
        queued = engine.events[job.queued_sequence - 1]
        if (trigger.kind != "stop" or trigger.racer_id not in actors or trigger.after != 22
                or _queued(queued, job.racer_id, trigger) != job):
            return False
    return True


def _idle_original_duelist_below22(engine, actor, core):
    """The exact native C/R/D/Flip-Flop Wild22 profile has no Duel window.

    Only R/C move, between22 and24. Native D stays strictly below22 and
    cannot meet either moving racer; Flip-Flop stays17, already tripped,
    until its next real turn. A preexisting D/Flip-Flop colocation does not
    receive a new arrival or start a new turn in this homogeneous FIFO.
    Current original R owns the outermost real resolution, so there is no
    suspended Duel body to skip. Every original queued R/C job still runs
    through the unchanged seven-event reader and complete state comparison.
    """
    by_power = {racer.ability: racer for racer in engine.racers.values()}
    if (set(by_power) != {Ability.COPY_CAT, Ability.ROMANTIC, Ability.DUELIST, Ability.FLIP_FLOP}
            or actor is not by_power[Ability.DUELIST] or type(actor.position) is not int
            or not 0 <= actor.position < 22
            or actor.status != RacerStatus.ACTIVE
            or actor.inherited_ability is not None or actor.copied_target is not None
            or actor.copied_ability is not None or actor.prediction is not None
            or engine.turn_order[engine.turn_index] != by_power[Ability.ROMANTIC].racer_id
            or getattr(engine, "_resolution_depth", 0) != 1
            or getattr(engine, "_physical_resolution_depth", 0) != 1):
        return False
    flip = by_power[Ability.FLIP_FLOP]
    return (core == {by_power[Ability.ROMANTIC].racer_id, by_power[Ability.COPY_CAT].racer_id}
            and flip.status == RacerStatus.ACTIVE and flip.position == 17 and flip.tripped is True
            and flip.inherited_ability is None and flip.copied_target is None
            and flip.copied_ability is None and flip.prediction is None)


def _spectator_evidence(engine, core):
    """Keep the old passive domain; separately authenticate resolved raw Twin.

    Twin has no new race-time grant after its actual before-race KEEP (or
    legal zero-window resolution with no previous winner). The native static
    power fast path is deliberately insufficient evidence for that fact.
    """
    twins = []
    for actor in engine.racers.values():
        if (actor.racer_id in core or engine._inert_duel_spectator_power(actor)
                or _idle_original_duelist_below22(engine, actor, core)):
            continue
        if (actor.status != RacerStatus.ACTIVE or actor.ability != Ability.TWIN
                or actor.position in {22, 23, 24}):
            return None
        evidence = resolved_inert_twin_evidence(engine, actor)
        if evidence is None:
            return None
        twins.append((actor.racer_id, evidence))
    return tuple(sorted(twins))


def key(engine):
    """Full entities, ledger and controls at a proved clean queue cut."""
    if (engine.track != WILD_WILDS or engine.termination_reason is not None or len(engine.finish_order) > 1
            or len(set(engine.finish_order)) != len(engine.finish_order)
            or any(rid not in engine.racers for rid in engine.finish_order)
            or engine._copy_locks or engine._refreshing_copies or engine._preparing or not engine._turn_in_progress
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0
            or type(getattr(engine, "_resolution_depth", 0)) is not int
            or getattr(engine, "_resolution_depth", 0) < 1):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    romantic, copy = actors
    core = {romantic.racer_id, copy.racer_id}
    if ((romantic.position, copy.position) != (22, 22)
            or set(engine._leaders()) != core
            or any(not engine._inert_duel_racer_status(actor) for actor in engine.racers.values())):
        return None
    spectators = _spectator_evidence(engine, core)
    if spectators is None:
        return None
    if not _queue_valid(engine, tuple(engine.pending_effects), core):
        return None
    roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                   for rid, actor in sorted(engine.racers.items()))
    state = (engine.track, engine._state_key(include_pending=False), roster, engine.score_ledger.snapshot(),
             len(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
             tuple(engine.egg_drawn), tuple(engine.previous_winners), getattr(engine, "_resolution_depth", 0),
             tuple(sorted(getattr(engine, "_duel_offered_epochs", {}).items())))
    if spectators:
        # Preserve the original state shape/operation for every old domain,
        # and retain the final depth slots used by the existing certificate.
        state = (*state[:-2], spectators, *state[-2:])
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


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual:
        raise ValueError("not a certified Romantic/Copy queue cut")
    return Snapshot(actual, len(engine.events), tuple(engine.pending_effects))


def _body(period, original, order):
    if len(period) != 7:
        return None
    power, move, stop, arrow, back, q1, q2 = period
    actor = original.racer_id
    if (not _event(power, "ability_effect", actor, source_id=actor, ability=Ability.ROMANTIC.value,
                   amount=2, parent_sequence=original.queued_sequence)
            or type(move.arrival_batch_id) is not int or move.arrival_batch_id < 1
            or type(arrow.arrival_batch_id) is not int or arrow.arrival_batch_id <= move.arrival_batch_id
            or not _event(move, "move", actor, source_id=actor, ability=Ability.ROMANTIC.value,
                          before=22, after=24, amount=2, movement_kind="ability", parent_sequence=power.sequence,
                          arrival_batch_id=move.arrival_batch_id)
            or not _event(stop, "stop", actor, before=22, after=24, parent_sequence=move.sequence,
                          arrival_batch_id=move.arrival_batch_id)
            or not _event(arrow, "move", actor, before=24, after=22, amount=-2, movement_kind="track",
                          parent_sequence=stop.sequence, arrival_batch_id=arrow.arrival_batch_id)
            or not _event(back, "stop", actor, before=24, after=22, parent_sequence=arrow.sequence,
                          arrival_batch_id=arrow.arrival_batch_id)):
        return None
    children = tuple(_queued(event, target, back) for event, target in zip((q1, q2), order))
    return None if None in children else children


def completed(engine, snapshot):
    if key(engine) != snapshot.state:
        return False
    romantic, copy = _actors(engine)
    actors = {romantic.racer_id, copy.racer_id}
    if not _queue_valid(engine, snapshot.original_pending, actors):
        return False
    events = engine.events[snapshot.event_start:]
    if (len(events) != 7 * len(snapshot.original_pending)
            or any(event.sequence != snapshot.event_start + index + 1 for index, event in enumerate(events))):
        return False
    order = tuple(rid for rid in engine._clockwise_order() if rid in actors)
    derived = []
    for index, original in enumerate(snapshot.original_pending):
        children = _body(events[7 * index:7 * (index + 1)], original, order)
        if children is None:
            return False
        derived.extend(children)
    originals = {job.queued_sequence for job in snapshot.original_pending}
    if (tuple(engine.pending_effects) != tuple(derived)
            or len(derived) != 2 * len(snapshot.original_pending)
            or originals.intersection(job.queued_sequence for job in derived)
            or any(job.queued_sequence <= snapshot.event_start for job in derived)):
        return False
    result = {"original_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
            "executed_pending_ids": [job.queued_sequence for job in snapshot.original_pending],
            "new_pending_ids": [job.queued_sequence for job in derived],
            "queue_before": len(snapshot.original_pending), "queue_after": len(derived),
            "clockwise_romantic_order": list(order), "scores_unchanged": True,
            "owner_resolution_depth": snapshot.state[-2], "retained_finish_order": list(engine.finish_order),
            "original_romantic_id": romantic.racer_id, "copy_id": copy.racer_id,
            "copy_original_ability": copy.ability.value,
            "copy_inherited_ability": copy.inherited_ability.value if copy.inherited_ability else None,
            "proof": "all original jobs executed; each exact seven-event body creates two fresh clockwise Romantic jobs"}
    twins = _spectator_evidence(engine, actors)
    if twins:
        result["resolved_inert_twins"] = [
            {"racer_id": rid, "original_ability": Ability.TWIN.value,
             "before_race_keep_event_sequences": [record.event_sequence for record in engine.decisions
                                                   if record.request.player_id == rid
                                                   and record.request.window == DecisionWindow.BEFORE_RACE],
             "previous_winners": [power.value for power in engine.previous_winners],
             "acquisition_evidence": list(evidence)} for rid, evidence in twins]
    return result
