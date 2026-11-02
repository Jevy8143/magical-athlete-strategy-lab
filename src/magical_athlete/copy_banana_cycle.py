"""A Banana-specific domain for the existing verified Copy/Romantic period.

Banana is not generally passive: copying it at 14 can Trip a passing racer.
At this cut it is borrowed only at 12, before the very next Romantic move
12->14 makes Romantic the unique leader and overwrites the borrowed power.
When Romantic subsequently passes Copy at 14, Copy therefore has Romantic,
not Banana. Both mandatory choices coalesce before any Banana effect can act.
"""

from dataclasses import dataclass, fields

from .events import PendingEffect
from .racer import Ability
from .scoring import ScoreSnapshot
from .state import RacerStatus
from .static_inheritance import static_inheritance_evidence
from .track import WILD_WILDS


REASON = "inert_copy_banana_prefix_cycle_proved"


def copy_source(engine, actor=None):
    """One original Copy, or one fully authenticated static E/T->Copy carrier.

    The native path and certificate stay unchanged. Static acquisition of
    Copy does not follow the original text of a copied Egg/Twin (H02).
    """
    copies = [racer for racer in engine.racers.values() if engine._is_copy(racer)]
    if len(copies) != 1 or actor is not None and copies[0] is not actor:
        return None
    copy = copies[0]
    if copy.ability == Ability.COPY_CAT and copy.inherited_ability is None:
        return copy, ()
    if copy.ability not in {Ability.EGG, Ability.TWIN} or copy.inherited_ability != Ability.COPY_CAT:
        return None
    evidence = static_inheritance_evidence(engine, copy, Ability.COPY_CAT)
    grants = [event.sequence for event in engine.events if event.kind == "inherit"
              and event.racer_id == copy.racer_id and event.ability == Ability.COPY_CAT.value]
    if evidence is None or len(grants) != 1:
        return None
    grant, = grants
    if any(event.racer_id == copy.racer_id and event.kind in {"ability_effect", "copy_changed"}
           and event.sequence <= grant for event in engine.events):
        return None
    return copy, (grant, evidence)


def alias_queue_valid(engine, copy, source, pending, *, require_unconsumed=True):
    if not source:
        return True
    grant, _ = source
    consumed = {event.parent_sequence for event in engine.events
                if event.kind in {"ability_effect", "effect_cancelled"}} if require_unconsumed else set()
    return all((job.racer_id != copy.racer_id or job.trigger_sequence > grant)
               and job.queued_sequence not in consumed for job in pending)


def alias_certificate(engine, snapshot):
    """Extra provenance only for the new static Copy carrier domain."""
    resolved = copy_source(engine)
    if resolved is None or not resolved[1]:
        return None
    copy, (grant, evidence) = resolved
    event_start, _, _, original = snapshot
    period = engine.events[event_start:]
    return {"original_pending_ids": [job.queued_sequence for job in original],
            "executed_pending_ids": [job.queued_sequence for job in original],
            "executed_period_pending_ids": [event.parent_sequence for event in period if event.kind == "ability_effect"],
            "new_pending_ids": [job.queued_sequence for job in engine.pending_effects],
            "queue_before": len(original), "queue_after": len(engine.pending_effects),
            "copy_decision_event_ids": [event.sequence for event in period if event.kind == "decision"],
            "static_copy": {"racer_id": copy.racer_id, "original_ability": copy.ability.value,
                            "inherited_ability": copy.inherited_ability.value,
                            "copy_grant_sequence": grant, "acquisition": evidence},
            "mechanical_state_returned": True, "scores_unchanged": True,
            "all_original_jobs_executed": True, "only_fresh_pending_descendants_stopped": True}

# Original fourth cards below12 cannot respond to these 12/14/16 arrivals.
# Scoocher is deliberately absent: it truly responds globally to the four
# Romantic effects and must keep all of its qualifications. Copy and inherited
# powers are excluded rather than inferred from one observed choice path.
FOURTH_ORIGINAL_ABILITIES = frozenset({
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BLIMP, Ability.CENTAUR,
    Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER, Ability.DUELIST,
    Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE, Ability.HECKLER,
    Ability.HUGE_BABY, Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY,
    Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN,
    Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL,
})


@dataclass(frozen=True)
class Snapshot:
    state: tuple
    event_start: int
    decision_start: int
    score_before: ScoreSnapshot
    original_pending: tuple[PendingEffect, ...]


def key(engine):
    """Recognize the exact R14/C12/B12 domain, including both FIFO phases.

    Existing Trip flags are retained, not erased. Any new Trip, score, input
    or other response must fail the completed-period check or complete-state
    equality. Event, decision and arrival IDs are provenance only; all rule
    controls and the original FIFO semantics are included or explicitly gated.
    """
    if (engine.track != WILD_WILDS or len(engine.racers) != 4
            or engine.termination_reason is not None or engine.finish_order
            or engine._copy_locks or engine._refreshing_copies or engine._preparing
            or not engine._turn_in_progress
            or hasattr(engine, "_arrow_visits") or hasattr(engine, "_arrow_queue_visits")
            or getattr(engine, "_movement_depth", 0) != 0):
        return None
    roles = {racer.ability: racer for racer in engine.racers.values()}
    fixed = {Ability.ROMANTIC, Ability.BANANA}
    if len(roles) != 4 or not fixed.issubset(roles):
        return None
    resolved = copy_source(engine)
    if resolved is None:
        return None
    copy, copy_origin = resolved
    if any(not isinstance(racer.ability, Ability) or racer.status != RacerStatus.ACTIVE
           or racer is not copy and racer.inherited_ability is not None
           or (racer.ability != Ability.MASTERMIND and racer.prediction is not None)
           or (racer.ability == Ability.MASTERMIND and racer.prediction is not None
               and (type(racer.prediction) is not int or racer.prediction not in engine.racers))
           or racer.finish_place is not None or racer.finish_places or not racer.pre_race_resolved
           or type(racer.position) is not int or type(racer.racer_id) is not int
           for racer in engine.racers.values()):
        return None
    romantic, banana = (roles[power] for power in (Ability.ROMANTIC, Ability.BANANA))
    fourth = next(racer for racer in roles.values() if racer not in (romantic, copy, banana))
    if ((romantic.position, copy.position, banana.position) != (14, 12, 12)
            or fourth.ability not in FOURTH_ORIGINAL_ABILITIES or not 0 <= fourth.position < 12
            or (copy.copied_target, copy.copied_ability) != (romantic.racer_id, Ability.ROMANTIC)
            or any(racer.copied_target is not None or racer.copied_ability is not None
                   for racer in (romantic, banana, fourth))):
        return None
    pending = tuple(engine.pending_effects)
    if not alias_queue_valid(engine, copy, copy_origin, pending):
        return None
    clockwise = [rid for rid in engine._clockwise_order() if rid in {copy.racer_id, romantic.racer_id}]
    sources = (copy, romantic) if clockwise[0] == copy.racer_id else (copy,)
    if len(pending) != len(sources):
        return None
    for effect, source in zip(pending, sources):
        if (effect.racer_id, effect.source_id, effect.kind, effect.steps, effect.ability, effect.target_id) != (
            source.racer_id, source.racer_id, "move", 2, Ability.ROMANTIC.value, source.racer_id,
        ) or not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(engine.events):
            return None
        trigger, queued = (engine.events[sequence - 1] for sequence in (
            effect.trigger_sequence, effect.queued_sequence,
        ))
        if (trigger.kind != "stop" or trigger.racer_id not in {copy.racer_id, romantic.racer_id}
                or trigger.after not in {12, 14}
                or (queued.kind, queued.racer_id, queued.source_id, queued.target_id,
                    queued.ability, queued.parent_sequence) != (
                        "effect_queued", source.racer_id, source.racer_id, source.racer_id,
                        Ability.ROMANTIC.value, trigger.sequence)):
            return None
    roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                   for rid, racer in sorted(engine.racers.items()))
    # A fourth Duelist is below12 and has no new sharing window here. Preserve
    # saved raw epochs and current eligibility rather than treating them as inert.
    epoch = engine._duel_actual_epoch()
    offered = getattr(engine, "_duel_offered_epochs", {})
    state = (engine._state_key(include_decisions=False), roster, engine.score_ledger.snapshot(),
             len(engine.turns), engine.race_index, tuple(engine.turn_order), tuple(engine.remaining_abilities),
             tuple(engine.egg_drawn), tuple(engine.previous_winners),
             getattr(engine, "_resolution_depth", 0), tuple(sorted(offered.items())),
             tuple((rid, offered.get(rid) == epoch) for rid in sorted(engine.racers)))
    if copy_origin:
        state = (*state, copy_origin)
    try:
        hash(state)
    except TypeError:
        return None
    return state


def capture(engine, state=None):
    """Freeze a real clean cut; never manufacture a pending qualification."""
    actual = key(engine)
    if actual is None or (state is not None and actual != state):
        return None
    return Snapshot(actual, len(engine.events), len(engine.decisions), engine.score_ledger.snapshot(),
                    tuple(engine.pending_effects))


def completed(engine, snapshot):
    """Require complete M repetition and every original task's actual body.

    Reuse the production validator for the actual six moves, four Romantic
    effects, mandatory {Banana, Romantic} choice, provenance and FIFO. That
    validator projects no Trip/score events away and retains Copy's final
    real borrowed power. Only new descendants can remain after all originals
    and the required derived prefix have genuinely run.
    """
    original = (snapshot.event_start, snapshot.decision_start, snapshot.score_before, snapshot.original_pending) if isinstance(snapshot, Snapshot) else None
    if original is None or key(engine) != snapshot.state or not engine._inert_copy_romantic_cycle_completed(original):
        return False
    return alias_certificate(engine, original) or True
