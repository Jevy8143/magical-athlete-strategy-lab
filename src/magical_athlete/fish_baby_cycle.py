"""A narrow all-choice closed queue; this module never executes or drops jobs.

At Suckerfish=Scoocher=p and Baby=p+1, a genuine Scoocher +1 notification
returns the complete mechanical state to this cut. KEEP produces one new
Baby notification. FOLLOW really uses Fish's power, produces its additional
notification, and sends both direct arrivals into the same Baby batch, then
back to p. Thus either choice leaves a nonempty queue of the same two sourced
notification types. Every original job and every actual choice must execute
before only proved new descendants may be stopped under existing O12.
"""

from dataclasses import dataclass, fields
import json

from .decisions import Action, ActionKind, DecisionWindow
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer
from .state import RacerStatus
from .static_inheritance import authenticated_static_power, static_inheritance_evidence
from .track import MILD_MILE, WILD_WILDS


REASON = "fish_baby_choice_queue_cycle_proved"

_HYPNOTIST_ENGINE_FIELDS = frozenset({
    "track", "max_events", "_turn_in_progress", "_preparing", "initial_remaining_abilities",
    "remaining_abilities", "previous_winners", "egg_drawn", "_copy_locks", "_refreshing_copies",
    "race_index", "score_ledger", "initial_score_ledger", "_score_start", "racers", "turn_order",
    "dice", "seed", "policies", "policy_metadata", "decisions", "finish_order", "turns", "events",
    "pending_effects", "turn_index", "next_turn_racer_id", "termination_reason", "initial_racers",
    "initial_code_fingerprint", "_genius_prediction", "_resolution_depth", "_physical_resolution_depth",
    "_arrival_counter", "_movement_depth", "_inert_duel_romantic", "_duel_offered_epochs", "max_action_events",
})


def _colocated_hypnotist(actors):
    return actors[1].ability == Ability.HYPNOTIST and actors[1].position == actors[3].position


def _hypnotist_control(engine, actors):
    """Extra strict identity/control gates apply only to this new domain."""
    from .fish_baby_star_cycle import _qualified, _event

    try:
        f, h, b, s = actors
        initial = {row["racer_id"]: row for row in engine.initial_racers}
        if (set(vars(engine)) - _HYPNOTIST_ENGINE_FIELDS
                or len(initial) != 4 or set(initial) != set(engine.racers)
                or (f.ability, h.ability, b.ability, s.ability) != (
                    Ability.SUCKERFISH, Ability.HYPNOTIST, Ability.HUGE_BABY, Ability.SCOOCHER)
                or any(type(row["racer_id"]) is not int for row in engine.initial_racers)
                or any(initial[actor.racer_id]["ability"] != actor.ability.value for actor in actors)
                or any(type(actor) is not Racer or set(vars(actor)) != {field.name for field in fields(Racer)}
                       or actor.inherited_ability is not None or type(actor.tripped) is not bool
                       or actor.prediction is not None or type(actor.turns_taken) is not int or actor.turns_taken < 0
                       or actor.pre_race_resolved is not True for actor in actors)
                or engine._preparing is not False or engine._turn_in_progress is not True
                or engine.next_turn_racer_id is not None or engine._genius_prediction is not None
                or getattr(engine, "_inert_duel_romantic", None) is not None
                or getattr(engine, "_duel_offered_epochs", {})
                or any(type(getattr(engine, name, None)) is not int or getattr(engine, name) < 1
                       for name in ("_resolution_depth", "_physical_resolution_depth", "_arrival_counter"))
                or type(engine.turn_index) is not int or not 0 <= engine.turn_index < 4
                or any(type(rid) is not int for rid in engine.turn_order)
                or len(engine.turn_order) != 4 or set(engine.turn_order) != set(engine.racers)
                or any(type(event) is not RaceEvent or set(vars(event)) != {field.name for field in fields(RaceEvent)}
                       or type(event.sequence) is not int or event.sequence != i + 1
                       for i, event in enumerate(engine.events))
                or any(type(job) is not PendingEffect or set(vars(job)) != {field.name for field in fields(PendingEffect)}
                       for job in engine.pending_effects)
                or not _qualified(engine, tuple(engine.pending_effects), f, b, s)):
            return None
        for job in engine.pending_effects:
            trigger = engine.events[job.trigger_sequence - 1]
            if type(trigger.parent_sequence) is not int or not 1 <= trigger.parent_sequence < trigger.sequence:
                return None
            parent = engine.events[trigger.parent_sequence - 1]
            if parent.kind == "ability_effect":
                if (type(parent.parent_sequence) is not int or not 1 <= parent.parent_sequence < parent.sequence
                        or not _event(parent, "ability_effect", s.racer_id, source_id=s.racer_id,
                                      ability=Ability.SCOOCHER.value, amount=1, parent_sequence=parent.parent_sequence)):
                    return None
                queued = engine.events[parent.parent_sequence - 1]
                if (queued.source_id not in {f.racer_id, h.racer_id, b.racer_id}
                        or type(queued.parent_sequence) is not int or not 1 <= queued.parent_sequence < queued.sequence
                        or not _event(queued, "effect_queued", s.racer_id, source_id=queued.source_id,
                                      ability=Ability.SCOOCHER.value, parent_sequence=queued.parent_sequence)):
                    return None
                ancestor = engine.events[queued.parent_sequence - 1]
                if type(ancestor.parent_sequence) is not int or not 1 <= ancestor.parent_sequence < ancestor.sequence:
                    return None
                if queued.source_id == f.racer_id:
                    if (ancestor.target_id not in {h.racer_id, s.racer_id, b.racer_id}
                            or type(ancestor.amount) is not int or ancestor.amount < 1
                            or not _event(ancestor, "ability_effect", f.racer_id, source_id=f.racer_id,
                                          target_id=ancestor.target_id, ability=Ability.SUCKERFISH.value,
                                          amount=ancestor.amount, parent_sequence=ancestor.parent_sequence)):
                        return None
                elif queued.source_id == b.racer_id:
                    if (type(ancestor.arrival_batch_id) is not int or ancestor.arrival_batch_id < 1
                            or not _event(ancestor, "ability_effect", b.racer_id, source_id=b.racer_id,
                                          ability=Ability.HUGE_BABY.value, parent_sequence=ancestor.parent_sequence,
                                          arrival_batch_id=ancestor.arrival_batch_id)):
                        return None
                else:
                    decision = engine.events[ancestor.parent_sequence - 1]
                    if (ancestor.target_id not in {f.racer_id, s.racer_id, b.racer_id}
                            or not _event(ancestor, "ability_effect", h.racer_id, source_id=h.racer_id,
                                          target_id=ancestor.target_id, ability=Ability.HYPNOTIST.value,
                                          parent_sequence=ancestor.parent_sequence)
                            or type(decision.decision_id) is not int or decision.decision_id < 1
                            or not _event(decision, "decision", h.racer_id, source_id=h.racer_id,
                                          target_id=ancestor.target_id, action_kind=ActionKind.HYPNOTIZE.value,
                                          decision_id=decision.decision_id)):
                        return None
                    record = engine.decisions[decision.decision_id - 1]
                    if (type(record.event_sequence) is not int or type(record.request.player_id) is not int
                            or record.event_sequence != decision.sequence or record.request.player_id != h.racer_id
                            or record.request.window != DecisionWindow.TURN_START
                            or record.action != Action(ActionKind.HYPNOTIZE, h.racer_id, ancestor.target_id)
                            or record.action not in record.request.legal_actions):
                        return None
            else:
                if (parent.kind != "main_move_resolved" or parent.racer_id not in {h.racer_id, b.racer_id, s.racer_id}
                        or type(parent.after) is not int or not 1 <= parent.after <= 6
                        or job.source_id == f.racer_id and parent.after != 1
                        or type(parent.parent_sequence) is not int or parent.parent_sequence != parent.sequence - 1
                        or not _event(parent, "main_move_resolved", parent.racer_id, before=parent.after,
                                      after=parent.after, parent_sequence=parent.parent_sequence)):
                    return None
                locked = engine.events[parent.parent_sequence - 1]
                if (type(locked.parent_sequence) is not int or locked.parent_sequence != locked.sequence - 1
                        or not _event(locked, "roll_locked", parent.racer_id, amount=parent.after,
                                      parent_sequence=locked.parent_sequence)
                        or not _event(engine.events[locked.parent_sequence - 1], "roll", parent.racer_id, amount=parent.after)):
                    return None
        immutable = json.dumps({"racers": engine.initial_racers, "scores": engine.initial_score_ledger,
                                "policies": engine.policy_metadata}, sort_keys=True, separators=(",", ":"))
        return (tuple(engine.turns), engine._physical_resolution_depth, tuple(sorted(vars(engine))), immutable,
                engine._score_start, engine.initial_code_fingerprint, engine.max_events,
                getattr(engine, "max_action_events", None), engine.seed,
                tuple(engine.initial_remaining_abilities), tuple(sorted(engine.policies)))
    except (AttributeError, KeyError, ValueError, TypeError, IndexError):
        return None

# These original spectator powers cannot act during the adjacent-space prefix
# when the spectator is elsewhere. In particular Romantic listens globally
# to the returned shared space, so it is deliberately absent.
PASSIVE_SPECTATOR_ABILITIES = frozenset({
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
    Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER,
    Ability.DUELIST, Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE,
    Ability.HECKLER, Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY,
    Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN,
    Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.SKIPPER, Ability.STICKLER, Ability.THIRD_WHEEL,
})


def _actors(engine):
    # Original card identity remains distinct from a fully authenticated,
    # immutable Egg/Twin power. Copy Cat borrowing never qualifies here.
    actors = tuple(engine.racers.values())
    if len(actors) != 4 or len({actor.ability for actor in actors}) != 4:
        return None
    wanted_powers = (Ability.SUCKERFISH, Ability.HUGE_BABY, Ability.SCOOCHER)
    candidates = [[actor for actor in actors
                   if actor.ability == wanted or actor.ability in {Ability.EGG, Ability.TWIN}
                   and actor.inherited_ability == wanted] for wanted in wanted_powers]
    if any(len(matches) != 1 for matches in candidates):
        return None
    principals = [matches[0] for matches in candidates]
    if len({actor.racer_id for actor in principals}) != 3:
        return None
    spectator, = [actor for actor in actors if all(actor is not principal for principal in principals)]
    if spectator.ability not in PASSIVE_SPECTATOR_ABILITIES:
        return None
    # Reject absent/duplicate powers and the spectator before scanning any
    # acquisition history in ordinary unrelated race states.
    if any(not authenticated_static_power(engine, actor, wanted)
           for actor, wanted in zip(principals, wanted_powers)):
        return None
    fish, baby, scoocher = principals
    return fish, spectator, baby, scoocher


def _new_static_principals(fish, scoocher):
    return fish.ability != Ability.SUCKERFISH or scoocher.ability != Ability.SCOOCHER


def _principal_bindings(engine, fish, baby, scoocher):
    """Source evidence and final grant; never follow a Copy Cat target."""
    result = []
    for actor, wanted in ((fish, Ability.SUCKERFISH), (baby, Ability.HUGE_BABY),
                          (scoocher, Ability.SCOOCHER)):
        evidence = static_inheritance_evidence(engine, actor, wanted)
        if evidence is None:
            return None
        if actor.ability == wanted:
            continue
        grants = [event.sequence for event in engine.events
                  if event.kind == "inherit" and event.racer_id == actor.racer_id
                  and event.ability == wanted.value]
        if not grants:
            return None
        grant = grants[-1]
        if any(event.racer_id == actor.racer_id and event.ability == wanted.value
               and event.kind == "ability_effect" and event.sequence <= grant
               for event in engine.events):
            return None
        result.append((actor.racer_id, wanted.value, grant, evidence))
    return tuple(result)


def _principal_packet(engine, fish, baby, scoocher):
    bindings = _principal_bindings(engine, fish, baby, scoocher)
    if bindings is None:
        return None
    return [{"racer_id": rid, "original_ability": engine.racers[rid].ability.value,
             "inherited_ability": wanted, "grant_sequence": grant, "acquisition": evidence}
            for rid, wanted, grant, evidence in bindings]


def _queue_valid(engine, pending, fish, baby, scoocher, *, require_unconsumed=True):
    if (not pending or any(not isinstance(effect, PendingEffect) for effect in pending)
            or len({effect.queued_sequence for effect in pending}) != len(pending)):
        return False
    new_principals = _new_static_principals(fish, scoocher)
    bindings = _principal_bindings(engine, fish, baby, scoocher) if new_principals else ()
    if bindings is None:
        return False
    final_grants = {rid: grant for rid, _, grant, _ in bindings}
    alias = baby.ability != Ability.HUGE_BABY
    grants = [event.sequence for event in engine.events if event.kind == "inherit"
              and event.racer_id == baby.racer_id and event.ability == Ability.HUGE_BABY.value] if alias else []
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}} if (alias or new_principals) and require_unconsumed else set()
    if (alias and not grants or (alias or new_principals)
            and consumed.intersection(effect.queued_sequence for effect in pending)):
        return False
    for effect in pending:
        if ((effect.racer_id, effect.kind, effect.steps, effect.ability, effect.target_id) != (
                scoocher.racer_id, "move", 1, Ability.SCOOCHER.value, None)
                or effect.source_id not in {fish.racer_id, baby.racer_id}
                or any(type(value) is not int for value in (effect.racer_id, effect.source_id,
                                                           effect.trigger_sequence, effect.queued_sequence))
                or not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(engine.events)):
            return False
        trigger, queued = (engine.events[sequence - 1] for sequence in (effect.trigger_sequence, effect.queued_sequence))
        ability = Ability.SUCKERFISH.value if effect.source_id == fish.racer_id else Ability.HUGE_BABY.value
        if ((trigger.kind, trigger.racer_id, trigger.source_id, trigger.ability) != (
                "ability_effect", effect.source_id, effect.source_id, ability)
                or (queued.kind, queued.racer_id, queued.source_id, queued.ability, queued.target_id, queued.parent_sequence) != (
                    "effect_queued", scoocher.racer_id, effect.source_id, Ability.SCOOCHER.value, None, trigger.sequence)
                or alias and effect.source_id == baby.racer_id and effect.trigger_sequence <= grants[-1]):
            return False
        if new_principals and effect.trigger_sequence <= max(
                final_grants.get(scoocher.racer_id, 0), final_grants.get(effect.source_id, 0)):
            return False
        if effect.source_id == fish.racer_id and (trigger.target_id, trigger.amount) != (scoocher.racer_id, 1):
            return False
    return True


def _colocated_tripped_inchworm(engine, actors):
    """Only the original I-main6/Baby6 return5, after all real Trips.

    Inchworm's main movement already happened. No new roll or turn begins
    while the captured Fish/Baby FIFO drains, so I has no new optional window.
    Its actual Trip survives this period and recovers on I's next real turn.
    """
    fish, inchworm, baby, scoocher = actors
    if (engine.track != WILD_WILDS
            or tuple(actor.ability for actor in actors) != (
                Ability.SUCKERFISH, Ability.INCHWORM, Ability.HUGE_BABY, Ability.SCOOCHER)
            or any(actor.inherited_ability is not None for actor in actors)
            or (fish.position, inchworm.position, baby.position, scoocher.position) != (5, 5, 6, 5)
            or any(actor.tripped is not True for actor in (fish, inchworm, scoocher))
            or engine.turn_order[engine.turn_index] != inchworm.racer_id):
        return False
    from .fish_baby_star_cycle import _event

    main = next((event for event in reversed(engine.events) if event.kind == "main_move_resolved"), None)
    if (main is None or type(main.parent_sequence) is not int
            or not 2 <= main.parent_sequence < main.sequence <= len(engine.events)
            or not _event(main, "main_move_resolved", inchworm.racer_id, before=6, after=6,
                          parent_sequence=main.parent_sequence)):
        return False
    locked = engine.events[main.parent_sequence - 1]
    if (locked.parent_sequence != locked.sequence - 1
            or not _event(locked, "roll_locked", inchworm.racer_id, amount=6, parent_sequence=locked.parent_sequence)
            or not _event(engine.events[locked.parent_sequence - 1], "roll", inchworm.racer_id, amount=6)):
        return False
    for actor in (fish, inchworm, scoocher):
        trip = next((event for event in reversed(engine.events)
                     if event.racer_id == actor.racer_id and event.kind in {"trip", "recover"}), None)
        if (trip is None or type(trip.parent_sequence) is not int
                or not main.sequence < trip.parent_sequence < trip.sequence <= len(engine.events)
                or not _event(trip, "trip", actor.racer_id, parent_sequence=trip.parent_sequence)):
            return False
        stop = engine.events[trip.parent_sequence - 1]
        if stop.kind != "stop" or stop.racer_id != actor.racer_id or stop.after != 5:
            return False
    return True


def key(engine):
    """Return complete mechanical/control state only at the proved clean cut."""
    if (engine.track not in (MILD_MILE, WILD_WILDS) or engine.termination_reason is not None or engine.finish_order
            or engine._copy_locks or engine._refreshing_copies or engine._preparing or not engine._turn_in_progress
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0):
        return None
    actors = _actors(engine)
    if actors is None:
        return None
    fish, spectator, baby, scoocher = actors
    position = scoocher.position
    if (type(position) is not int or not 1 <= position <= engine.track.finish_position - 2
            or fish.position != position or baby.position != position + 1
            or (spectator.position == position and spectator.ability not in {Ability.COACH, Ability.STICKLER, Ability.HYPNOTIST}
                and not _colocated_tripped_inchworm(engine, actors))
            or spectator.position == position + 1
            or position in engine.track.star_positions
            or (position in engine.track.trip_positions
                and not (fish.tripped is True and scoocher.tripped is True))
            or position in dict(engine.track.arrows)):
        return None
    # Baby redirects every direct arrival before its stop is resolved. The
    # direct p+1 square may itself be special, but it cannot be the finish:
    # finish resolution precedes Baby. Only the returned p stop can fire.
    if any(actor.status != RacerStatus.ACTIVE or actor is spectator and actor.inherited_ability is not None
           or actor.copied_target is not None or actor.copied_ability is not None
           or actor.finish_place is not None or actor.finish_places or not actor.pre_race_resolved
           or type(actor.position) is not int or not 0 <= actor.position < engine.track.finish_position
           or type(actor.racer_id) is not int for actor in actors):
        return None
    pending = tuple(engine.pending_effects)
    colocated_stickler = spectator.position == position and spectator.ability == Ability.STICKLER
    colocated_hypnotist = _colocated_hypnotist(actors)
    hypnotist_control = _hypnotist_control(engine, actors) if colocated_hypnotist else ()
    if hypnotist_control is None:
        return None
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}} if colocated_stickler or colocated_hypnotist else set()
    if (not _queue_valid(engine, pending, fish, baby, scoocher)
            or (colocated_stickler or colocated_hypnotist) and any(job.queued_sequence in consumed for job in pending)):
        return None
    roster = tuple((rid, tuple((field.name, getattr(actor, field.name)) for field in fields(actor)))
                   for rid, actor in sorted(engine.racers.items()))
    actual = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
              "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
    epoch = (len(engine.turns), next((event.sequence for event in reversed(engine.events) if event.kind in actual), 0))
    offered = getattr(engine, "_duel_offered_epochs", {})
    control = (getattr(engine, "_resolution_depth", 0), tuple(sorted(offered.items())),
               tuple((rid, offered.get(rid) == epoch) for rid in sorted(engine.racers)))
    state = (engine.track, engine._state_key(include_pending=False, include_decisions=False), roster, engine.score_ledger.snapshot(),
             len(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
             tuple(engine.egg_drawn), tuple(engine.previous_winners), control)
    if baby.ability != Ability.HUGE_BABY:
        inheritance = static_inheritance_evidence(engine, baby, Ability.HUGE_BABY)
        if inheritance is None:
            return None
        state = (*state[:-1], inheritance, state[-1])
    if _new_static_principals(fish, scoocher):
        bindings = _principal_bindings(engine, fish, baby, scoocher)
        if bindings is None:
            return None
        state = (*state[:-1], bindings, state[-1])
    if colocated_hypnotist:
        state = (*state[:-1], hypnotist_control, state[-1])
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
    event_prefix: tuple = ()
    decision_prefix: tuple = ()
    score_before: object = None
    arrival_counter: int | None = None


def capture(engine, state=None):
    actual = key(engine)
    if actual is None or state is not None and state != actual:
        raise ValueError("not a certified Fish/Baby queue cut")
    fish, spectator, baby, scoocher = _actors(engine)
    hypnotist = _colocated_hypnotist((fish, spectator, baby, scoocher))
    alias = (_new_static_principals(fish, scoocher) or baby.ability != Ability.HUGE_BABY
             or spectator.ability == Ability.STICKLER and spectator.position == scoocher.position or hypnotist)
    return Snapshot(actual, len(engine.events), len(engine.decisions), tuple(engine.pending_effects),
                    tuple(engine.events) if alias else (), tuple(engine.decisions) if alias else (),
                    engine.score_ledger.snapshot() if hypnotist else None,
                    engine._arrival_counter if hypnotist else None)


def _hypnotist_completed(engine, snapshot, actors):
    """Strict real observation/body parser; no new Hypnotist window in FIFO."""
    from .fish_baby_star_cycle import _Reader, _Reject, _qualified

    class OrdinaryReader(_Reader):
        def job(self, pending):
            record = self.choices[self.choice_index]
            if type(record.request.player_id) is not int or type(record.event_sequence) is not int:
                raise _Reject
            return super().job(pending)

        def star(self, actor, stop):
            pass

    try:
        fish, hypnotist, baby, scoocher = actors
        if (tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
                or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix
                or engine.score_ledger.snapshot() != snapshot.score_before
                or any(type(job) is not PendingEffect or set(vars(job)) != {field.name for field in fields(PendingEffect)}
                       for job in snapshot.original_pending)
                or not _qualified(engine, snapshot.original_pending, fish, baby, scoocher, require_unconsumed=False)):
            return False
        reader, fifo = OrdinaryReader(engine, snapshot, actors), list(snapshot.original_pending)
        if len(reader.choices) != len(fifo):
            return False
        derived = []
        for job in fifo:
            derived.extend(reader.job(job))
        if (reader.offset != len(reader.events) or reader.choice_index != len(reader.choices)
                or reader.batch != engine._arrival_counter or not derived
                or tuple(derived) != tuple(engine.pending_effects)
                or set(job.queued_sequence for job in fifo).intersection(job.queued_sequence for job in derived)):
            return False
        return {"original_pending_ids": [job.queued_sequence for job in fifo],
                "executed_pending_ids": reader.executed,
                "new_pending_ids": [job.queued_sequence for job in derived],
                "queue_before": len(fifo), "queue_after": len(derived),
                "capture_event_count": snapshot.event_start,
                "suckerfish_decision_event_ids": reader.choice_ids,
                "keep_jobs": len(fifo) - reader.follow_count, "follow_jobs": reader.follow_count,
                "mechanical_state_returned": True, "scores_unchanged": True,
                "all_original_jobs_executed": True, "only_fresh_pending_descendants_stopped": True,
                "colocated_original_hypnotist": {"racer_id": hypnotist.racer_id, "ability": hypnotist.ability.value}}
    except (AttributeError, KeyError, ValueError, TypeError, IndexError, _Reject):
        return False


def completed(engine, snapshot):
    """Verify each original FIFO job and either complete, actual choice path."""
    if key(engine) != snapshot.state:
        return False
    fish, spectator, baby, scoocher = _actors(engine)
    if _colocated_hypnotist((fish, spectator, baby, scoocher)):
        return _hypnotist_completed(engine, snapshot, (fish, spectator, baby, scoocher))
    colocated_stickler = spectator.ability == Ability.STICKLER and spectator.position == scoocher.position
    new_principals = _new_static_principals(fish, scoocher)
    if (baby.ability != Ability.HUGE_BABY or colocated_stickler or new_principals) and (
            tuple(engine.events[:snapshot.event_start]) != snapshot.event_prefix
            or tuple(engine.decisions[:snapshot.decision_start]) != snapshot.decision_prefix
            or not _queue_valid(engine, snapshot.original_pending, fish, baby, scoocher, require_unconsumed=False)):
        return False
    f, b, s = fish.racer_id, baby.racer_id, scoocher.racer_id
    position, destination = scoocher.position, baby.position
    choices = engine.decisions[snapshot.decision_start:]
    if len(choices) != len(snapshot.original_pending):
        return False
    events, cursor, derived = engine.events[snapshot.event_start:], 0, []
    returned_order = [rid for rid in engine._clockwise_order() if rid in {f, s}]
    expected_legal = {Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, f, s)}
    for original, choice in zip(snapshot.original_pending, choices):
        request = choice.request
        if (request.player_id != f or request.window != DecisionWindow.MOVE_START
                or len(request.legal_actions) != 2 or set(request.legal_actions) != expected_legal
                or choice.action not in expected_legal or request.observation.current_move_value != 1
                or request.observation.current_player_id != engine.turn_order[engine.turn_index]
                or dict(request.observation.score_totals) != engine.score_ledger.totals()):
            return False
        follow = choice.action.kind == ActionKind.SUCKERFISH_FOLLOW
        period = events[cursor:cursor + (10 if follow else 6)]
        expected = ([("decision", f), ("ability_effect", s), ("ability_effect", f), ("effect_queued", s),
                     ("move", s), ("move", f), ("ability_effect", b), ("effect_queued", s),
                     *(("stop", rid) for rid in returned_order)]
                    if follow else [("decision", f), ("ability_effect", s), ("move", s),
                                    ("ability_effect", b), ("effect_queued", s), ("stop", s)])
        if [(event.kind, event.racer_id) for event in period] != expected:
            return False
        decision, power = period[:2]
        if ((decision.sequence, decision.decision_id, decision.action_kind, decision.source_id,
             decision.target_id, decision.parent_sequence) != (
                 choice.event_sequence, request.decision_id, choice.action.kind.value,
                 choice.action.source_id, choice.action.target_id, original.queued_sequence)
                or (power.source_id, power.ability, power.amount, power.parent_sequence) != (
                    s, Ability.SCOOCHER.value, 1, original.queued_sequence)):
            return False
        if follow:
            fish_power, fish_job, move, fish_move, baby_power, baby_job, first_stop, last_stop = period[2:]
            stops = {event.racer_id: event for event in (first_stop, last_stop)}
            stop, fish_stop = stops[s], stops[f]
            if ((fish_power.source_id, fish_power.target_id, fish_power.ability, fish_power.amount, fish_power.parent_sequence) != (
                    f, s, Ability.SUCKERFISH.value, 1, power.sequence)
                    or (fish_job.source_id, fish_job.target_id, fish_job.ability, fish_job.parent_sequence) != (
                        f, None, Ability.SCOOCHER.value, fish_power.sequence)
                    or (fish_move.before, fish_move.after, fish_move.amount, fish_move.movement_kind,
                        fish_move.source_id, fish_move.ability, fish_move.parent_sequence) != (
                            position, destination, 1, "ability", f, Ability.SUCKERFISH.value, fish_power.sequence)
                    or fish_move.arrival_batch_id != move.arrival_batch_id
                    or (fish_stop.before, fish_stop.after, fish_stop.parent_sequence) != (destination, position, baby_power.sequence)
                    or fish_stop.arrival_batch_id != stop.arrival_batch_id):
                return False
            derived.append(PendingEffect(s, f, fish_power.sequence, fish_job.sequence))
        else:
            move, baby_power, baby_job, stop = period[2:]
        if ((move.before, move.after, move.amount, move.movement_kind, move.source_id, move.ability, move.parent_sequence) != (
                position, destination, 1, "ability", s, Ability.SCOOCHER.value, power.sequence)
                or (baby_power.source_id, baby_power.ability, baby_power.parent_sequence) != (b, Ability.HUGE_BABY.value, power.sequence)
                or baby_power.arrival_batch_id != move.arrival_batch_id
                or (baby_job.source_id, baby_job.target_id, baby_job.ability, baby_job.parent_sequence) != (
                    b, None, Ability.SCOOCHER.value, baby_power.sequence)
                or (stop.before, stop.after, stop.parent_sequence) != (destination, position, baby_power.sequence)
                or stop.arrival_batch_id == move.arrival_batch_id):
            return False
        derived.append(PendingEffect(s, b, baby_power.sequence, baby_job.sequence))
        cursor += len(period)
    valid = (cursor == len(events) and bool(derived) and tuple(engine.pending_effects) == tuple(derived)
             and not {effect.queued_sequence for effect in snapshot.original_pending}.intersection(
                 effect.queued_sequence for effect in engine.pending_effects))
    if not valid or baby.ability == Ability.HUGE_BABY and not colocated_stickler and not new_principals:
        return valid
    result = {"original_pending_ids": [effect.queued_sequence for effect in snapshot.original_pending],
            "executed_pending_ids": [effect.queued_sequence for effect in snapshot.original_pending],
            "new_pending_ids": [effect.queued_sequence for effect in engine.pending_effects],
            "queue_before": len(snapshot.original_pending), "queue_after": len(engine.pending_effects),
            "suckerfish_decision_event_ids": [choice.event_sequence for choice in choices],
            "mechanical_state_returned": True, "scores_unchanged": True}
    if baby.ability != Ability.HUGE_BABY:
        result["static_baby"] = {"racer_id": b, "original_ability": baby.ability.value,
                                 "inherited_ability": baby.inherited_ability.value,
                                 "acquisition": static_inheritance_evidence(engine, baby, Ability.HUGE_BABY)}
    if new_principals:
        result["static_principals"] = _principal_packet(engine, fish, baby, scoocher)
    if colocated_stickler:
        result["colocated_original_stickler"] = {"racer_id": spectator.racer_id, "ability": spectator.ability.value}
    return result
