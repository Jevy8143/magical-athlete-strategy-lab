"""Atomic movement and arrival batches for the incremental race engine.

The host owns turn/roll windows, scoring, public decisions and terminal checks.
This mixin keeps simultaneous arrivals distinct from Huge Baby's causal child
batches, and queues only qualifications captured at the actual trigger event.
"""

from collections import defaultdict
from dataclasses import fields
from functools import wraps
import json

from .decisions import Action, ActionKind, DecisionWindow, DecisionRequest, Observation, PublicRacer
from .static_inheritance import static_inheritance_evidence
from .events import PendingEffect, RaceEvent
from .racer import Ability
from .state import RacerStatus
from .track import WILD_WILDS
from . import copy_banana_cycle, fish_baby_cycle, star_baby_cycle, romantic_queue_cycle, preparing_copy_romantic_cycle, romantic_anchor_cycle, romantic_baby_star_cycle, fish_baby_star_cycle, romantic_baby_arrow_cycle, romantic_duel_copy_cycle
from . import egg_copy_scoocher_cycle, romantic_scoocher_anchor_cycle, romantic_baby_baba_cycle, copy_fish_baby_cycle, centaur_copy_baby_star_cycle
from . import fish_baby_arrow11_cycle, romantic_baby_shared12_cycle
from . import scoocher_baby_duel22_cycle
from . import fish_baby_baba4_cycle, fish_baby_arrow11_scoocher_cycle
from . import duel_romantic_scoocher22_cycle
from . import copy_baby_duel_scoocher_cycle


RESOLUTION_DEPTH_LIMIT = 128
# Heap continuations cannot exhaust the native stack. Retain a separate,
# deterministic diagnostic for private calls outside the normal event budget.
RESOLUTION_FRAME_LIMIT = 10_000

# These powers cannot act between the tied-lead Copy choice and Romantic's
# next isolated 12->14 arrival. Their normal turn/roll/movement windows remain
# available afterwards; this list makes no claim that the powers are inert
# elsewhere. Powers that can respond to this prefix and inherited cards are
# excluded; a power listening only to its own movement cannot act here.
COPY_PREFIX_PASSIVE_ABILITIES = frozenset({
    Ability.SKIPPER, Ability.HECKLER, Ability.INCHWORM, Ability.LACKEY,
    Ability.ALCHEMIST, Ability.BLIMP, Ability.CENTAUR, Ability.CHEERLEADER,
    Ability.COACH, Ability.DICEMONGER, Ability.FLIP_FLOP, Ability.GENIUS,
    Ability.GUNK, Ability.HARE, Ability.HYPNOTIST, Ability.LEAPTOAD,
    Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN, Ability.MASTERMIND,
    Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.STICKLER, Ability.THIRD_WHEEL,
})

# Independent proof domain for *spectators* of 22->24->22. Main-roll,
# turn-start/end and own-movement powers cannot act in this prefix. The four
# arrival/follow/passing listeners listed here are also inert when their
# positions avoid22/23/24. Scoocher/Copy and before-race inheritance are excluded.
# Keep this explicit: a future role must not silently enter the proof domain.
DUEL_PREFIX_PASSIVE_SPECTATOR_ABILITIES = frozenset({
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
    Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER,
    Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE, Ability.HECKLER,
    Ability.HUGE_BABY, Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY,
    Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN,
    Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL,
})


def bounded_resolution(method):
    """Bound synchronous engine nesting, never award places.

    Suspended Duel/Drain generators retain a separate logical depth for
    proof controls. Only actual synchronous calls consume the native stack
    guard, independently of incidental policy/Python caller frames.
    """
    @wraps(method)
    def resolve(self, *args, **kwargs):
        depth = getattr(self, "_resolution_depth", 0)
        physical_depth = getattr(self, "_physical_resolution_depth", 0)
        if physical_depth >= RESOLUTION_DEPTH_LIMIT:
            self._action_diagnostic("resolution_recursion_budget_exceeded")
        self._resolution_depth = depth + 1
        self._physical_resolution_depth = physical_depth + 1
        try:
            return method(self, *args, **kwargs)
        finally:
            self._resolution_depth = depth
            self._physical_resolution_depth = physical_depth
    return resolve


def inert_duel_scope(method):
    """Keep one proved cohort visible across Duelist's nested FIFO drains."""
    @wraps(method)
    def resolve(self, *args, **kwargs):
        owner = not hasattr(self, "_inert_duel_romantic")
        if owner:
            # A finite external prefix may create the clean pair later in
            # this drain. The outer scope still owns its eventual cleanup.
            self._inert_duel_romantic = None
        try:
            yield from method(self, *args, **kwargs)
        finally:
            if owner:
                del self._inert_duel_romantic
    return resolve


class RaceActions:
    def _run_resolution_frames(self, first):
        """Finish each child before resuming its suspended Duel/Drain parent.

        Generators retain the original locals, proof snapshots and post-child
        work. Logical depth remains visible to the existing proof validators;
        only synchronous calls consume the native stack guard.
        """
        entry_depth = getattr(self, "_resolution_depth", 0)
        stack = [(first, entry_depth)]
        failed = False
        try:
            while stack:
                frame, depth = stack[-1]
                self._resolution_depth = depth
                try:
                    child = next(frame)
                except StopIteration:
                    stack.pop()
                    continue
                except RuntimeError as error:
                    # PEP 479 wraps a dice/policy StopIteration crossing a
                    # generator boundary. Restore the old synchronous API;
                    # an ordinary RuntimeError must retain its own meaning.
                    cause = error.__cause__
                    if (type(error) is RuntimeError
                            and str(error) == "generator raised StopIteration"
                            and isinstance(cause, StopIteration)
                            and error.__context__ is cause):
                        raise cause
                    raise
                if len(stack) >= RESOLUTION_FRAME_LIMIT:
                    child.close()
                    self._action_diagnostic("resolution_frame_budget_exceeded")
                # Each yield replaces precisely one former synchronous call.
                stack.append((child, depth + 1))
        except BaseException:
            failed = True
            raise
        finally:
            cleanup_error = None
            # Release inner scopes first, even after a diagnostic or callback
            # error. No suspended parent's post-child job is run on failure.
            while stack:
                frame, depth = stack.pop()
                self._resolution_depth = depth
                try:
                    frame.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            self._resolution_depth = entry_depth
            if not failed and cleanup_error is not None:
                raise cleanup_error

    def _new_batch(self):
        self._arrival_counter = getattr(self, "_arrival_counter", 0) + 1
        return self._arrival_counter

    def _state_key(self, *, include_scores=True, include_pending=True, include_decisions=True):
        # Event IDs are provenance, not state: retaining them would hide cycles.
        return (
            self.turn_index,
            len(self.decisions) if include_decisions else None,
            getattr(self, "next_turn_racer_id", None),
            getattr(self, "_genius_prediction", None),
            tuple((rid, r.position, r.status.value, r.tripped, self._ability(r),
                   r.copied_target, r.prediction, r.turns_taken)
                  for rid, r in sorted(self.racers.items())),
            tuple(sorted(self.score_ledger.totals().items())) if include_scores else None,
            tuple((e.racer_id, e.source_id, e.kind, e.steps, e.ability, e.target_id)
                  for e in self.pending_effects) if include_pending else None,
            tuple(sorted(getattr(self, "_copy_locks", ()))),
            tuple(self.finish_order),
        )

    def _star_cycle_completed(self, snapshot):
        """Prove a forced cycle whose only state drift is printed Star +1s.

        Full-score equality remains the normal cycle test. This extra proof is
        narrow: no player choice or die was consumed, mechanical state and the
        ordered queue repeat, and every balance change is an unconditional
        track Star. Neither these moves nor Star scoring branch on balances.
        Do not apply the projection to penalties, power scores or new inputs.
        """
        event_start, entry_start, totals = snapshot
        entries = self.score_ledger.entries[entry_start:]
        if not entries or any(
            entry.source != "track" or entry.reason != "star" or entry.delta != 1
            or entry.race_index != self.race_index or entry.phase != "during_race"
            for entry in entries
        ):
            return False
        if any(event.kind in {"roll", "reroll", "duel_roll", "decision"}
               for event in self.events[event_start:]):
            return False
        expected = totals.copy()
        for entry in entries:
            expected[entry.player_id] += entry.delta
        return expected == self.score_ledger.totals()

    def _queued_arrow_cycle_completed(self, snapshot):
        """Prove a forced Arrow cycle that only appends deferred qualifications.

        Track actions finish before the pending power queue drains. Baby can
        return a racer to the same arrow while Romantic/Duelist keep adding
        effects; their unprocessed queue cannot break that geometric cycle.
        Keep every real effect from the initial arrival and one complete cycle.
        """
        event_start, entry_start, original_pending = snapshot
        current = tuple(self.pending_effects)
        if current[:len(original_pending)] != original_pending:
            return False
        if len(self.score_ledger.entries) != entry_start:
            return False
        events = self.events[event_start:]
        if any(event.kind in {"roll", "reroll", "duel_roll", "decision", "score", "effect_cancelled"}
               for event in events):
            return False
        appended = current[len(original_pending):]
        queued = {event.sequence for event in events if event.kind == "effect_queued"}
        return (len(appended) == len(queued)
                and {effect.queued_sequence for effect in appended} == queued)

    def _inert_copy_mastermind_prediction(self):
        """Authenticate the one idle original Mastermind in the Party profile.

        A real prediction is persistent mechanical state. It only acts on a
        finish, absent from this verified 12/14/16 period. Party's turn-start
        pull cannot restart inside the FIFO; either copy target therefore
        rejoins the existing all-choice period before it could act.
        """
        roles = {racer.ability: racer for racer in self.racers.values()}
        if (len(self.racers) != 4 or set(roles) != {
                Ability.ROMANTIC, Ability.COPY_CAT, Ability.MASTERMIND, Ability.PARTY_ANIMAL}):
            return None
        mastermind, copy = roles[Ability.MASTERMIND], roles[Ability.COPY_CAT]
        if (mastermind.position != 10 or type(mastermind.prediction) is not int
                or mastermind.prediction not in self.racers
                or type(mastermind.turns_taken) is not int or mastermind.turns_taken < 1
                or self.turn_order[self.turn_index] != copy.racer_id):
            return None
        predictions = [event for event in self.events
                       if event.kind == "prediction" and event.racer_id == mastermind.racer_id]
        if len(predictions) != 1:
            return None
        event, = predictions
        if (type(event) is not RaceEvent or type(event.parent_sequence) is not int
                or type(event.sequence) is not int
                or not 1 <= event.parent_sequence < event.sequence <= len(self.events)):
            return None
        choices = [record for record in self.decisions if record.event_sequence == event.parent_sequence]
        if len(choices) != 1:
            return None
        record, = choices
        actor, target = mastermind.racer_id, mastermind.prediction
        action = Action(ActionKind.MASTERMIND_PREDICT, actor, target)
        observed = record.request.observation.racers
        if (len(observed) != 4 or {racer.racer_id for racer in observed} != set(self.racers)
                or record.request.observation.current_player_id != actor):
            return None
        order = tuple(racer.racer_id for racer in observed)
        start = order.index(actor)
        legal = tuple(Action(ActionKind.MASTERMIND_PREDICT, actor, rid)
                      for rid in order[start:] + order[:start])
        if (event.sequence != record.event_sequence + 1
                or event != RaceEvent(event.sequence, "prediction", actor, source_id=actor,
                                      target_id=target, ability=Ability.MASTERMIND.value,
                                      parent_sequence=record.event_sequence)
                or record.request.window != DecisionWindow.TURN_START
                or record.request.player_id != actor or record.request.legal_actions != legal
                or record.action != action
                or self.events[record.event_sequence - 1] != RaceEvent(
                    record.event_sequence, "decision", actor, source_id=actor, target_id=target,
                    action_kind=ActionKind.MASTERMIND_PREDICT.value, decision_id=record.request.decision_id)):
            return None
        return actor

    def _inert_copy_eliminated_mouth_spectator(self):
        """One original MOUTH, truly eliminated by an already qualified Copy.

        Dead racers are excluded from leaders and all arrival power captures.
        This local profile does not widen either global passive-power set.
        The existing FIFO reader still serves every old live qualification.
        """
        roles = {racer.ability: racer for racer in self.racers.values()}
        if (self.track != WILD_WILDS or len(self.racers) != 4
                or set(roles) != {Ability.ROMANTIC, Ability.COPY_CAT, Ability.MOUTH, Ability.FLIP_FLOP}):
            return None
        r, c, m, f = (roles[a] for a in (Ability.ROMANTIC, Ability.COPY_CAT, Ability.MOUTH, Ability.FLIP_FLOP))
        if (f.position != 12 or m.position != 14
                or m.status != RacerStatus.ELIMINATED
                or any(actor.status != RacerStatus.ACTIVE for actor in (r,c,f))):
            return None
        initial = [actor for actor in self.initial_racers if actor['racer_id'] == m.racer_id]
        if (len(initial) != 1 or initial[0]['ability'] != Ability.MOUTH.value or initial[0]['status'] != 'active'):
            return None
        def exact(event, kind, actor, **values):
            return (all(value is None or type(value) is int for value in (
                        event.sequence,event.racer_id,event.source_id,event.target_id,event.parent_sequence,
                        event.arrival_batch_id,event.before,event.after,event.amount,event.decision_id))
                    and event == RaceEvent(event.sequence,kind,actor,**values))
        def parent(event):
            sequence=event.parent_sequence
            if type(sequence) is not int or not 1<=sequence<event.sequence<=len(self.events):
                return None
            result=self.events[sequence-1]
            return result if result.sequence==sequence else None
        eliminated=[event for event in self.events if event.kind=='eliminate' and event.racer_id==m.racer_id]
        if len(eliminated)!=1:return None
        gone=eliminated[0];power=parent(gone);queued=parent(power) if power is not None else None
        stop=parent(queued) if queued is not None else None;move=parent(stop) if stop is not None else None
        if (power is None or queued is None or stop is None or move is None
                or not exact(gone,'eliminate',m.racer_id,source_id=c.racer_id,ability=Ability.MOUTH.value,parent_sequence=power.sequence)
                or not exact(power,'ability_effect',c.racer_id,source_id=c.racer_id,target_id=m.racer_id,
                             ability=Ability.MOUTH.value,parent_sequence=queued.sequence)
                or not exact(queued,'effect_queued',c.racer_id,source_id=c.racer_id,target_id=m.racer_id,
                             ability=Ability.MOUTH.value,parent_sequence=stop.sequence)
                or type(move.arrival_batch_id) is not int or move.arrival_batch_id<=0
                or not exact(stop,'stop',c.racer_id,before=move.before,after=14,
                             parent_sequence=move.sequence,arrival_batch_id=move.arrival_batch_id)):
            return None
        # Actual root here is an old C Romantic instance that survives borrowing
        # MOUTH; no current borrowed power may retract that old qualification.
        body=parent(move);old=parent(body) if body is not None else None
        if (body is None or old is None or not exact(move,'move',c.racer_id,source_id=c.racer_id,
                    ability=Ability.ROMANTIC.value,before=12,after=14,amount=2,movement_kind='ability',
                    parent_sequence=body.sequence,arrival_batch_id=move.arrival_batch_id)
                or not exact(body,'ability_effect',c.racer_id,source_id=c.racer_id,ability=Ability.ROMANTIC.value,
                             amount=2,parent_sequence=old.sequence)
                or not exact(old,'effect_queued',c.racer_id,source_id=c.racer_id,target_id=c.racer_id,
                             ability=Ability.ROMANTIC.value,parent_sequence=old.parent_sequence)):
            return None
        trigger=parent(old)
        prior=[event for event in self.events[:old.sequence] if event.kind=='copy_changed' and event.racer_id==c.racer_id]
        if (trigger is None or trigger.kind!='stop' or trigger.racer_id!=r.racer_id or trigger.after!=12
                or old.sequence-trigger.sequence not in {1,2}
                or not prior or prior[-1].sequence>=trigger.sequence
                or not exact(prior[-1],'copy_changed',c.racer_id,source_id=c.racer_id,
                                         target_id=r.racer_id,ability=Ability.ROMANTIC.value)):
            return None
        changes=[event for event in self.events[:stop.sequence] if event.kind=='copy_changed' and event.racer_id==c.racer_id]
        if not changes or not exact(changes[-1],'copy_changed',c.racer_id,source_id=c.racer_id,
                                    target_id=m.racer_id,ability=Ability.MOUTH.value):
            return None
        positions={actor['racer_id']:actor['position'] for actor in self.initial_racers}
        for event in self.events[:stop.sequence]:
            if event.kind in {'move','warp','stop'} and event.racer_id in positions and type(event.after) is int:
                positions[event.racer_id]=event.after
        if {rid for rid,p in positions.items() if rid!=c.racer_id and p==14}!={m.racer_id}:
            return None
        if any(event.racer_id==m.racer_id and event.kind in {'move','warp','stop','recover','finish','inherit','copy_changed'}
               for event in self.events[gone.sequence:]):
            return None
        return m.racer_id

    def _inert_copy_romantic_cycle_key(self):
        """A narrow, mechanically proved all-branches-confluent Wild cycle.

        At this cut R=14, C=P=12, C copies R, and FIFO is [C] or [C,R].
        C reaches R at 14; R and then C visit 16->12; the mandatory tied-lead
        copy choice is a prefix-passive original power or Romantic. R then
        reaches 14 alone. Neither chosen power can act before that arrival,
        and its unique lead forces C back to Romantic. The fourth original
        card is below 12; any extra response it produces fails the period
        and queue proof. Both legal choices return to the same complete cut.
        This is not a general permission to ignore decisions in cycle state.
        """
        if (self.track != WILD_WILDS or len(self.racers) != 4
                or self.termination_reason is not None or self.finish_order
                or self._copy_locks or self._refreshing_copies or self._preparing or not self._turn_in_progress):
            return None
        roles = {racer.ability: racer for racer in self.racers.values()}
        if (len(roles) != 4 or Ability.ROMANTIC not in roles
                or any(not isinstance(racer.ability, Ability)
                       for racer in self.racers.values())):
            return None
        resolved = copy_banana_cycle.copy_source(self)
        if resolved is None:
            return None
        copy, copy_origin = resolved
        predicted_spectator = self._inert_copy_mastermind_prediction()
        eliminated_mouth = self._inert_copy_eliminated_mouth_spectator()
        if any((racer.status != RacerStatus.ACTIVE and racer.racer_id != eliminated_mouth) or racer is not copy and racer.inherited_ability is not None
               or racer is not copy and racer.ability in {Ability.EGG, Ability.TWIN}
               or racer.prediction is not None and racer.racer_id != predicted_spectator
               or racer.finish_place is not None or racer.finish_places
               or not racer.pre_race_resolved or type(racer.position) is not int
               or type(racer.racer_id) is not int for racer in self.racers.values()):
            return None
        romantic = roles[Ability.ROMANTIC]
        others = [racer for racer in self.racers.values() if racer not in (romantic, copy)]
        candidates = [racer for racer in others
                      if racer.position == 12 and racer.ability in COPY_PREFIX_PASSIVE_ABILITIES]
        if len(candidates) != 1:
            return None
        passive = candidates[0]
        fourth = next(racer for racer in others if racer is not passive)
        if ((romantic.position, copy.position) != (14, 12)
                or (not 0 <= fourth.position < 12 and fourth.racer_id != eliminated_mouth)
                or copy.copied_target != romantic.racer_id or copy.copied_ability != Ability.ROMANTIC):
            return None
        if any(racer.copied_target is not None or racer.copied_ability is not None
               for racer in (passive, romantic, fourth)):
            return None
        pending = tuple(self.pending_effects)
        if not copy_banana_cycle.alias_queue_valid(self, copy, copy_origin, pending):
            return None
        clockwise = [rid for rid in self._clockwise_order() if rid in {copy.racer_id, romantic.racer_id}]
        sources = (copy, romantic) if clockwise[0] == copy.racer_id else (copy,)
        if len(pending) != len(sources):
            return None
        for effect, source in zip(pending, sources):
            if (effect.racer_id, effect.source_id, effect.kind, effect.steps, effect.ability, effect.target_id) != (
                source.racer_id, source.racer_id, "move", 2, Ability.ROMANTIC.value, source.racer_id,
            ):
                return None
            if not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(self.events):
                return None
            trigger, queued = (self.events[sequence - 1] for sequence in (
                effect.trigger_sequence, effect.queued_sequence,
            ))
            if (trigger.kind != "stop" or trigger.racer_id not in {copy.racer_id, romantic.racer_id}
                    or trigger.after not in {12, 14}
                    or (queued.kind, queued.racer_id, queued.source_id, queued.target_id,
                        queued.ability, queued.parent_sequence) != (
                            "effect_queued", source.racer_id, source.racer_id, source.racer_id,
                            Ability.ROMANTIC.value, trigger.sequence)):
                return None
        # Include every racer field and the complete score account, not only
        # public positions. IDs of queue provenance are checked below separately.
        roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                       for rid, racer in sorted(self.racers.items()))
        key = (self._state_key(include_decisions=False), roster, self.score_ledger.snapshot(),
               len(self.turns), self.race_index, tuple(self.turn_order),
               tuple(self.remaining_abilities), tuple(self.egg_drawn), tuple(self.previous_winners),
               tuple(sorted(getattr(self, "_duel_offered_epochs", {}).items())))
        if copy_origin:
            key = (*key, copy_origin)
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def _inert_copy_romantic_cycle_completed(self, snapshot):
        """Verify one whole actual period against the all-choice proof above."""
        event_start, decision_start, score_before, original_pending = snapshot
        if self.score_ledger.snapshot() != score_before:
            return False
        roles = {racer.ability: racer for racer in self.racers.values()}
        resolved = copy_banana_cycle.copy_source(self)
        if resolved is None:
            return False
        copy, source = resolved
        romantic = roles[Ability.ROMANTIC]
        if (not copy_banana_cycle.alias_queue_valid(self, copy, source, original_pending, require_unconsumed=False)
                or not copy_banana_cycle.alias_queue_valid(self, copy, source, tuple(self.pending_effects))):
            return False
        passive = next(racer for racer in self.racers.values()
                       if racer is not copy and racer.position == 12)
        p, r, c = passive.racer_id, romantic.racer_id, copy.racer_id
        choices = self.decisions[decision_start:]
        if len(choices) != 1:
            return False
        choice = choices[0]
        expected = {Action(ActionKind.COPY_TARGET, c, target) for target in (p, r)}
        if (choice.request.player_id != c or choice.request.window != DecisionWindow.COPY_CHOICE
                or len(choice.request.legal_actions) != 2 or set(choice.request.legal_actions) != expected
                or choice.action not in expected):
            return False
        observed = {racer.racer_id: racer for racer in choice.request.observation.racers}
        if (set(observed) != set(self.racers)
                or any(observed[rid].position != 12 for rid in (p, r, c))
                or observed[p].ability != passive.ability.value
                or observed[r].ability != Ability.ROMANTIC.value
                or observed[c].ability is not None or observed[c].copied_target is not None
                or dict(choice.request.observation.score_totals) != self.score_ledger.totals()):
            return False
        events = self.events[event_start:]
        allowed = {"decision", "copy_changed", "ability_effect", "move", "stop", "passing", "effect_queued"}
        if any(event.kind not in allowed for event in events):
            return False
        decision_events = [event for event in events if event.kind == "decision"]
        if len(decision_events) != 1 or (
            decision_events[0].sequence, decision_events[0].racer_id, decision_events[0].decision_id,
            decision_events[0].action_kind, decision_events[0].source_id, decision_events[0].target_id,
        ) != (
            choice.event_sequence, c, choice.request.decision_id,
            choice.action.kind.value, choice.action.source_id, choice.action.target_id,
        ):
            return False
        copied_powers = {None: None, r: Ability.ROMANTIC.value, p: passive.ability.value}
        if any(event.racer_id != c or event.source_id != c or event.target_id not in copied_powers
               or event.ability != copied_powers[event.target_id]
               for event in events if event.kind == "copy_changed"):
            return False
        moves = [event for event in events if event.kind == "move"]
        if [(event.racer_id, event.before, event.after, event.amount, event.movement_kind) for event in moves] != [
            (c, 12, 14, 2, "ability"), (r, 14, 16, 2, "ability"), (r, 16, 12, -4, "track"),
            (c, 14, 16, 2, "ability"), (c, 16, 12, -4, "track"), (r, 12, 14, 2, "ability"),
        ]:
            return False
        # The single real choice occurs after C loses the sole lead at 14 and
        # before R's isolated arrival overwrites either candidate ability.
        if not moves[4].sequence < choice.event_sequence < moves[5].sequence:
            return False
        effects = [event for event in events if event.kind == "ability_effect"]
        if [(event.racer_id, event.source_id, event.ability, event.amount) for event in effects] != [
            (actor, actor, Ability.ROMANTIC.value, 2) for actor in (c, r, c, r)
        ]:
            return False
        for effect, move in zip(effects, (moves[0], moves[1], moves[3], moves[5])):
            if (move.source_id, move.ability, move.parent_sequence) != (
                effect.racer_id, Ability.ROMANTIC.value, effect.sequence,
            ):
                return False
        stops = [event for event in events if event.kind == "stop"]
        if len(stops) != len(moves) or any(
            (stop.racer_id, stop.before, stop.after, stop.parent_sequence) != (
                move.racer_id, move.before, move.after, move.sequence,
            ) for stop, move in zip(stops, moves)
        ):
            return False
        if any((moves[index].source_id, moves[index].ability, moves[index].parent_sequence) != (
            None, None, stops[index - 1].sequence,
        ) for index in (2, 4)):
            return False
        consumed = [event.parent_sequence for event in effects]
        initial_count = len(original_pending)
        if (initial_count not in {1, 2} or len(set(consumed)) != 4
                or consumed[:initial_count] != [effect.queued_sequence for effect in original_pending]):
            return False
        queued = {event.sequence: event for event in events if event.kind == "effect_queued"}
        clockwise = [rid for rid in self._clockwise_order() if rid in {c, r}]
        if [(event.racer_id, event.source_id, event.target_id, event.ability) for event in queued.values()] != [
            (actor, actor, actor, Ability.ROMANTIC.value) for actor in clockwise * 2
        ]:
            return False
        if [event.parent_sequence for event in queued.values()] != [stops[0].sequence] * 2 + [stops[2].sequence] * 2:
            return False
        if any(sequence not in queued for sequence in consumed[initial_count:]):
            return False
        # Reconstruct the real FIFO, preserving every original qualification
        # and requiring all remaining IDs to be this period's new descendants.
        fifo = [effect.queued_sequence for effect in original_pending]
        for event in events:
            if event.kind == "effect_queued":
                fifo.append(event.sequence)
            elif event.kind == "ability_effect":
                if not fifo or fifo.pop(0) != event.parent_sequence:
                    return False
        current = tuple(self.pending_effects)
        if (len(current) != initial_count or tuple(effect.queued_sequence for effect in current) != tuple(fifo)
                or set(fifo) != set(queued) - set(consumed)):
            return False
        by_sequence = {event.sequence: event for event in events}
        for queued_event in queued.values():
            trigger = by_sequence.get(queued_event.parent_sequence)
            if trigger not in stops or trigger.after not in {12, 14}:
                return False
        for effect in current:
            queued_event = queued.get(effect.queued_sequence)
            trigger = by_sequence.get(effect.trigger_sequence)
            if (queued_event is None or trigger is None or trigger.kind != "stop"
                    or queued_event.parent_sequence != effect.trigger_sequence
                    or queued_event.racer_id != effect.racer_id or queued_event.target_id != effect.target_id):
                return False
        return True

    def _egg_two_anchor_profile(self):
        """Exact idle-Copy closure; no copied state is projected out."""
        try:
            roles = {r.ability: r for r in self.racers.values()}
            if (self.track != WILD_WILDS or set(roles) != {Ability.EGG, Ability.LACKEY, Ability.GENIUS, Ability.COPY_CAT}
                    or len(self.racers) != 4 or self.termination_reason is not None or self.finish_order
                    or self._preparing or not self._turn_in_progress or self._refreshing_copies or self._copy_locks
                    or self._genius_prediction is not None or hasattr(self, "_arrow_visits")
                    or hasattr(self, "_arrow_queue_visits") or getattr(self, "_movement_depth", 0) != 0):
                return None
            egg, lackey, genius, copy = (roles[a] for a in (Ability.EGG, Ability.LACKEY, Ability.GENIUS, Ability.COPY_CAT))
            if (tuple(r.position for r in (egg, lackey, genius, copy)) != (12, 14, 12, 9)
                    or self.turn_order[self.turn_index] != genius.racer_id
                    or (copy.copied_target, copy.copied_ability) != (lackey.racer_id, Ability.LACKEY)
                    or egg.inherited_ability != Ability.ROMANTIC):
                return None
            if any(r.status != RacerStatus.ACTIVE or r.tripped or not r.pre_race_resolved
                   or r.prediction is not None or r.finish_place is not None or r.finish_places
                   or type(r.racer_id) is not int or type(r.position) is not int or type(r.turns_taken) is not int
                   or r is not egg and r.inherited_ability is not None
                   or r is not copy and (r.copied_target is not None or r.copied_ability is not None)
                   for r in (egg, lackey, genius, copy)):
                return None
            evidence = static_inheritance_evidence(self, egg, Ability.ROMANTIC)
            grants = [event for event in self.events if event.kind == "inherit" and event.racer_id == egg.racer_id]
            if not evidence or len(grants) != 1 or grants[0].ability != Ability.ROMANTIC.value:
                return None
            main = next(event for event in reversed(self.events) if event.kind == "main_move_resolved")
            if main.racer_id != genius.racer_id:
                return None
            return egg, lackey, genius, copy, evidence, grants[0].sequence, main.sequence
        except (AttributeError, KeyError, TypeError, ValueError, IndexError, StopIteration):
            return None

    def _egg_two_anchor_pending(self, profile, pending, *, unconsumed=True):
        egg, lackey, genius, copy, _, grant, main = profile
        if len(pending) != 1:
            return False
        job = pending[0]
        if (not romantic_baby_shared12_cycle._known_pending(job)
                or (job.racer_id, job.source_id, job.kind, job.steps, job.ability, job.target_id)
                    != (egg.racer_id, egg.racer_id, "move", 2, Ability.ROMANTIC.value, egg.racer_id)
                or not grant < job.trigger_sequence < job.queued_sequence <= len(self.events)):
            return False
        stop, queued = (self.events[i - 1] for i in (job.trigger_sequence, job.queued_sequence))
        valid = romantic_baby_shared12_cycle._event
        if (stop.racer_id not in {egg.racer_id, genius.racer_id} or stop.after != 12
                or type(stop.before) is not int or type(stop.arrival_batch_id) is not int
                or not valid(stop, "stop", stop.racer_id, before=stop.before, after=12,
                             parent_sequence=stop.parent_sequence, arrival_batch_id=stop.arrival_batch_id)
                or not valid(queued, "effect_queued", egg.racer_id, source_id=egg.racer_id, target_id=egg.racer_id,
                             ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
                or queued.sequence != stop.sequence + 1 or main >= stop.sequence):
            return False
        positions = {r["racer_id"]: r["position"] for r in self.initial_racers}
        for event in self.events[:stop.sequence]:
            if event.kind in {"move", "warp", "stop"}:
                positions[event.racer_id] = event.after
        if ({rid for rid, pos in positions.items() if pos == 12} != {egg.racer_id, genius.racer_id}
                or sum(e.kind == "stop" and e.arrival_batch_id == stop.arrival_batch_id for e in self.events) != 1):
            return False
        return not unconsumed or not any(e.kind in {"ability_effect", "effect_cancelled"}
                                         and e.parent_sequence == job.queued_sequence for e in self.events)

    def _egg_two_anchor_key(self):
        profile = self._egg_two_anchor_profile()
        if profile is None or not self._egg_two_anchor_pending(profile, tuple(self.pending_effects)):
            return None
        roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                       for rid, racer in sorted(self.racers.items()))
        key = ("egg-two-anchor-idle-copy-full-state-v1", self._state_key(include_decisions=False), roster,
               self.score_ledger.snapshot(), len(self.turns), self.race_index, tuple(self.turn_order),
               tuple(self.remaining_abilities), tuple(self.egg_drawn), tuple(self.previous_winners),
               profile[4:], self._resolution_depth, self._physical_resolution_depth,
               getattr(self, "_movement_depth", 0), tuple(sorted(getattr(self, "_duel_offered_epochs", {}).items())))
        try:
            hash(key)
            return key
        except TypeError:
            return None

    def _egg_two_anchor_completed(self, snapshot):
        """Read one genuine full 12->14->16->12 orbit and its Copy choice."""
        try:
            start, decision_start, score_before, original = snapshot
            profile = self._egg_two_anchor_profile()
            if (profile is None or self.score_ledger.snapshot() != score_before
                    or not self._egg_two_anchor_pending(profile, original, unconsumed=False)):
                return False
            egg, lackey, genius, copy = profile[:4]
            e, l, g, c = (r.racer_id for r in (egg, lackey, genius, copy))
            events, choices = self.events[start:], self.decisions[decision_start:]
            if len(choices) != 1:
                return False
            record = choices[0]
            fish_baby_arrow11_cycle._typed_request(record)
            legal = (Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, c, rid)
                     for rid in self._clockwise_order() if rid in {e, l}))
            observation = Observation(self.turn_order[self.turn_index], len(self.turns) + 1, self.track.name,
                self.track.finish_position, self.track.second_corner_position, tuple(sorted(self.track.trip_positions)),
                tuple(PublicRacer(rid, self.racers[rid].name, 14 if rid == e else self.racers[rid].position,
                      "active", Ability.LACKEY.value if rid == c else self._ability(self.racers[rid]).value,
                      False, None, l if rid == c else None) for rid in self.turn_order), (), None, None,
                score_totals=tuple(self.score_ledger.totals().items()))
            request = DecisionRequest(decision_start + 1, c, DecisionWindow.COPY_CHOICE, observation, legal)
            if record.request != request or record.action not in legal:
                return False
            cursor, batch = 0, None
            valid = romantic_baby_shared12_cycle._event
            def take(kind, actor, **values):
                nonlocal cursor
                if cursor >= len(events) or not valid(events[cursor], kind, actor, **values):
                    raise ValueError("two-anchor actual body differs")
                event = events[cursor]
                cursor += 1
                return event
            power = take("ability_effect", e, source_id=e, ability=Ability.ROMANTIC.value, amount=2,
                         parent_sequence=original[0].queued_sequence)
            batch = events[cursor].arrival_batch_id
            prior_batch = max((event.arrival_batch_id for event in self.events[:start]
                               if type(event.arrival_batch_id) is int), default=0)
            if type(batch) is not int or batch != prior_batch + 1:
                return False
            move = take("move", e, source_id=e, ability=Ability.ROMANTIC.value, before=12, after=14,
                        amount=2, movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=batch)
            stop = take("stop", e, before=12, after=14, parent_sequence=move.sequence, arrival_batch_id=batch)
            first_child = take("effect_queued", e, source_id=e, target_id=e, ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
            choice = record.action
            decision = take("decision", c, action_kind=choice.kind.value, source_id=choice.source_id,
                            target_id=choice.target_id, decision_id=request.decision_id)
            if decision.sequence != record.event_sequence:
                return False
            chose_egg = choice.kind == ActionKind.COPY_TARGET and choice.target_id == e
            if chose_egg:
                take("copy_changed", c, source_id=c, target_id=e)
            power = take("ability_effect", e, source_id=e, ability=Ability.ROMANTIC.value, amount=2,
                         parent_sequence=first_child.sequence)
            move = take("move", e, source_id=e, ability=Ability.ROMANTIC.value, before=14, after=16,
                        amount=2, movement_kind="ability", parent_sequence=power.sequence, arrival_batch_id=batch + 1)
            stop = take("stop", e, before=14, after=16, parent_sequence=move.sequence, arrival_batch_id=batch + 1)
            arrow = take("move", e, before=16, after=12, amount=-4, movement_kind="track",
                         parent_sequence=stop.sequence, arrival_batch_id=batch + 2)
            take("passing", e, target_id=l, parent_sequence=arrow.sequence, arrival_batch_id=batch + 2)
            stop = take("stop", e, before=16, after=12, parent_sequence=arrow.sequence, arrival_batch_id=batch + 2)
            child = take("effect_queued", e, source_id=e, target_id=e, ability=Ability.ROMANTIC.value, parent_sequence=stop.sequence)
            if chose_egg:
                take("copy_changed", c, source_id=c, target_id=l, ability=Ability.LACKEY.value)
            expected = PendingEffect(e, e, stop.sequence, child.sequence, "move", 2, Ability.ROMANTIC.value, e)
            if (cursor != len(events) or tuple(self.pending_effects) != (expected,)
                    or self._arrival_counter != batch + 2 or child.sequence <= start):
                return False
            return {"profile": "egg-two-anchor-idle-copy-full-state-v1", "capture_event_count": start,
                "original_pending_ids": [original[0].queued_sequence],
                "executed_pending_ids": [original[0].queued_sequence, first_child.sequence],
                "new_pending_ids": [child.sequence], "actual_copy_choice_event_id": decision.sequence,
                "actual_copy_action": choice.kind.value, "actual_copy_target": choice.target_id,
                "copied_target_and_power_retained_after_real_unique_leader_refresh": True,
                "mechanical_state_returned": True, "scores_unchanged": True,
                "all_original_jobs_really_executed": True, "only_fresh_descendants_stopped": True,
                "proof": "R12->14 and R14->16->12 each replace R; all real Copy choices are idle and unique Lackey14 restores the complete Copy state at12"}
        except (AttributeError, KeyError, ValueError, TypeError, IndexError, StopIteration,
                fish_baby_arrow11_cycle._Reject):
            return False

    def _egg_romantic_idle_copy_roles(self):
        """Two explicit original-card families share one proved seven-event word.

        In the added family Skipper only listens to dice and Sisyphus only
        to its own main roll. The real roll and next-turn requests have already
        happened; this FIFO contains no roll. Neither copied Skipper nor the
        no-power original Egg changes any arrival in the 22->24->22 word.
        """
        roles = {racer.ability: racer for racer in self.racers.values()}
        if set(roles) == {Ability.EGG, Ability.LEAPTOAD, Ability.DUELIST, Ability.COPY_CAT}:
            return tuple(roles[power] for power in (Ability.EGG, Ability.LEAPTOAD, Ability.DUELIST, Ability.COPY_CAT))
        if set(roles) == {Ability.EGG, Ability.SKIPPER, Ability.SISYPHUS, Ability.COPY_CAT}:
            egg, anchor, spectator, copy = (roles[power] for power in (
                Ability.EGG, Ability.SKIPPER, Ability.SISYPHUS, Ability.COPY_CAT))
            # Below the tied leaders, the spectator cannot enter Copy's menu;
            # no Sisyphus movement/roll/score occurs inside this exact word.
            if type(spectator.position) is int and 0 <= spectator.position < 22:
                return egg, anchor, spectator, copy
        return None

    def _inert_egg_romantic_cycle_key(self):
        """Recognize only the closed Egg 22->24->22 Wild self-chain.

        Egg really inherited Romantic; H02 means Copy can borrow no power
        from Egg, or Leaptoad from the other tied leader. Both are passive
        during Egg's forced moves. The copied state matters on later turns:
        retain it, and require that exact state to repeat before stopping.
        Only decision counts and event/arrival provenance are projected out.
        """
        two_anchor = self._egg_two_anchor_key()
        if two_anchor is not None:
            return two_anchor
        if (self.track != WILD_WILDS or len(self.racers) != 4
                or self.termination_reason is not None or self.finish_order
                or self._copy_locks or self._refreshing_copies or self._preparing
                or not self._turn_in_progress or self._genius_prediction is not None):
            return None
        roles = self._egg_romantic_idle_copy_roles()
        if roles is None:
            return None
        egg, leap, duel, copy = roles
        spectator_position = 13 if duel.ability == Ability.DUELIST else duel.position
        if ((egg.position, leap.position, duel.position, copy.position) != (22, 22, spectator_position, 12)
                or self.turn_order[self.turn_index] != egg.racer_id
                or egg.inherited_ability != Ability.ROMANTIC):
            return None
        for racer in self.racers.values():
            if (racer.status != RacerStatus.ACTIVE or racer.tripped
                    or racer.prediction is not None or racer.finish_place is not None or racer.finish_places
                    or not racer.pre_race_resolved or type(racer.position) is not int
                    or type(racer.racer_id) is not int):
                return None
            if racer is not egg and racer.inherited_ability is not None:
                return None
            if racer is not copy and (racer.copied_target is not None or racer.copied_ability is not None):
                return None
        expected_copy = {egg.racer_id: None, leap.racer_id: leap.ability}
        if copy.copied_target not in expected_copy or copy.copied_ability != expected_copy[copy.copied_target]:
            return None
        pending = tuple(self.pending_effects)
        if len(pending) != 1:
            return None
        effect = pending[0]
        if (effect.racer_id, effect.source_id, effect.kind, effect.steps, effect.ability, effect.target_id) != (
            egg.racer_id, egg.racer_id, "move", 2, Ability.ROMANTIC.value, egg.racer_id,
        ):
            return None
        if not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(self.events):
            return None
        trigger, queued = (self.events[sequence - 1] for sequence in (effect.trigger_sequence, effect.queued_sequence))
        if ((trigger.kind, trigger.racer_id, trigger.after) != ("stop", egg.racer_id, 22)
                or (queued.kind, queued.racer_id, queued.source_id, queued.target_id,
                    queued.ability, queued.parent_sequence) != (
                        "effect_queued", egg.racer_id, egg.racer_id, egg.racer_id,
                        Ability.ROMANTIC.value, trigger.sequence)):
            return None
        roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                       for rid, racer in sorted(self.racers.items()))
        key = (self._state_key(include_decisions=False), roster, self.score_ledger.snapshot(),
               len(self.turns), self.race_index, tuple(self.turn_order),
               tuple(self.remaining_abilities), tuple(self.egg_drawn), tuple(self.previous_winners),
               tuple(sorted(getattr(self, "_duel_offered_epochs", {}).items())))
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def _inert_egg_romantic_cycle_completed(self, snapshot):
        """Check every real period and every possible Copy branch is inert.

        No choices are skipped or replaced. A copied Leaptoad affects only
        Copy's own movement, and a copied original Egg has no borrowed power;
        neither can alter this Egg-only chain. Its final actual copied state
        stays intact. A different fourth card, score, roll, or queue fails.
        """
        if self._egg_two_anchor_key() is not None:
            return self._egg_two_anchor_completed(snapshot)
        event_start, decision_start, score_before, original_pending = snapshot
        if self.score_ledger.snapshot() != score_before or len(original_pending) != 1:
            return False
        roles = self._egg_romantic_idle_copy_roles()
        if roles is None:
            return False
        egg, leap, duel, copy = roles
        e, l, d, c = (racer.racer_id for racer in (egg, leap, duel, copy))
        expected = {Action(ActionKind.KEEP), Action(ActionKind.COPY_TARGET, c, e), Action(ActionKind.COPY_TARGET, c, l)}
        choices = self.decisions[decision_start:]
        events = self.events[event_start:]
        if not choices or len(choices) > 2:
            return False
        cursor = 0
        consumed = original_pending[0].queued_sequence
        last_trigger = None
        for choice in choices:
            request = choice.request
            if (request.player_id != c or request.window != DecisionWindow.COPY_CHOICE
                    or len(request.legal_actions) != 3 or set(request.legal_actions) != expected
                    or choice.action not in expected):
                return False
            observed = {racer.racer_id: racer for racer in request.observation.racers}
            if (set(observed) != {e, l, d, c}
                    or any(observed[rid].position != position for rid, position in ((e, 22), (l, 22), (d, duel.position), (c, 12)))
                    or observed[e].ability != Ability.ROMANTIC.value
                    or observed[l].ability != leap.ability.value
                    or observed[d].ability != duel.ability.value
                    or observed[c].copied_target not in {e, l}
                    or observed[c].ability != (None if observed[c].copied_target == e else leap.ability.value)
                    or dict(request.observation.score_totals) != self.score_ledger.totals()):
                return False
            period = events[cursor:cursor + 7]
            if len(period) != 7 or [(event.kind, event.racer_id) for event in period] != [
                ("ability_effect", e), ("move", e), ("stop", e), ("move", e),
                ("stop", e), ("effect_queued", e), ("decision", c),
            ]:
                return False
            power, forward, far_stop, arrow, back_stop, queued, decision = period
            if ((power.source_id, power.ability, power.amount, power.parent_sequence) != (
                    e, Ability.ROMANTIC.value, 2, consumed)
                    or (forward.before, forward.after, forward.amount, forward.movement_kind,
                        forward.source_id, forward.ability, forward.parent_sequence) != (
                            22, 24, 2, "ability", e, Ability.ROMANTIC.value, power.sequence)
                    or (far_stop.before, far_stop.after, far_stop.parent_sequence) != (22, 24, forward.sequence)
                    or (arrow.before, arrow.after, arrow.amount, arrow.movement_kind,
                        arrow.source_id, arrow.ability, arrow.parent_sequence) != (24, 22, -2, "track", None, None, far_stop.sequence)
                    or (back_stop.before, back_stop.after, back_stop.parent_sequence) != (24, 22, arrow.sequence)
                    or (queued.source_id, queued.target_id, queued.ability, queued.parent_sequence) != (
                        e, e, Ability.ROMANTIC.value, back_stop.sequence)
                    or (decision.sequence, decision.decision_id, decision.action_kind,
                        decision.source_id, decision.target_id) != (
                            choice.event_sequence, request.decision_id, choice.action.kind.value,
                            choice.action.source_id, choice.action.target_id)):
                return False
            cursor += 7
            if cursor < len(events) and events[cursor].kind == "copy_changed":
                changed = events[cursor]
                target = choice.action.target_id
                if (choice.action.kind != ActionKind.COPY_TARGET
                        or (changed.racer_id, changed.source_id, changed.target_id, changed.ability) != (
                            c, c, target, None if target == e else leap.ability.value)):
                    return False
                cursor += 1
            consumed, last_trigger = queued.sequence, back_stop.sequence
        current = tuple(self.pending_effects)
        return (cursor == len(events) and len(current) == 1
                and current[0].queued_sequence == consumed and current[0].trigger_sequence == last_trigger)

    def _homogeneous_baby_scoocher_cycle_key(self):
        """An actual closed queue: T(M, q) = (M, [q, q]).

        Read the actors from the real notification, not their physical cards:
        a legally inherited/copied power is subject to the same proof. At the
        clean cut S=11/B=12, executing q must restore *all* mechanical state M
        while creating exactly two identical notifications, without any input.
        The completed-period check below verifies this for every original job.

        Pending length and provenance IDs are not inputs to role/track rules;
        they are read only by engineering cycle recognition and safety budgets.
        Each new q therefore repeats the same transformation, independently of
        the number of identical jobs behind it. Stop only the new descendants
        after all original jobs actually ran. Other racers may be anywhere or
        hold inherited powers; any real response that breaks this closure is
        rejected by the complete state, event, score and FIFO checks.
        """
        if (self.track != WILD_WILDS or len(self.racers) != 4
                or self.termination_reason is not None
                or self._copy_locks or self._refreshing_copies or self._preparing
                or not self._turn_in_progress
                or hasattr(self, "_arrow_visits") or hasattr(self, "_arrow_queue_visits")
                or getattr(self, "_movement_depth", 0) != 0):
            return None
        if (len({racer.ability for racer in self.racers.values()}) != 4
                or any(not isinstance(racer.ability, Ability) or not racer.pre_race_resolved
                       for racer in self.racers.values())):
            return None
        if any(type(racer.position) is not int
               or type(racer.racer_id) is not int for racer in self.racers.values()):
            return None
        pending = tuple(self.pending_effects)
        if not pending or len({effect.queued_sequence for effect in pending}) != len(pending):
            return None
        scoocher = self.racers.get(pending[0].racer_id)
        baby = self.racers.get(pending[0].source_id)
        if (scoocher is None or baby is None or scoocher is baby
                or scoocher.status != RacerStatus.ACTIVE or baby.status != RacerStatus.ACTIVE
                or (scoocher.position, baby.position) != (11, 12)
                or self._ability(scoocher) != Ability.SCOOCHER or self._ability(baby) != Ability.HUGE_BABY):
            return None
        for effect in pending:
            if (effect.racer_id, effect.source_id, effect.kind, effect.steps, effect.ability, effect.target_id) != (
                scoocher.racer_id, baby.racer_id, "move", 1, Ability.SCOOCHER.value, None,
            ):
                return None
            if not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(self.events):
                return None
            trigger, queued = (self.events[sequence - 1] for sequence in (
                effect.trigger_sequence, effect.queued_sequence,
            ))
            if ((trigger.kind, trigger.racer_id, trigger.source_id, trigger.ability) != (
                    "ability_effect", baby.racer_id, baby.racer_id, Ability.HUGE_BABY.value)
                    or (queued.kind, queued.racer_id, queued.source_id, queued.ability, queued.parent_sequence) != (
                        "effect_queued", scoocher.racer_id, baby.racer_id, Ability.SCOOCHER.value, trigger.sequence)):
                return None
        roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                       for rid, racer in sorted(self.racers.items()))
        # _offer_duel compares the saved epoch with the latest actual event.
        # Preserve that equality relation as well as the raw saved dictionary:
        # event IDs may drift without changing a window's eligibility.
        actual_kinds = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                        "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
        epoch = (len(self.turns), next((event.sequence for event in reversed(self.events)
                                       if event.kind in actual_kinds), 0))
        offered = getattr(self, "_duel_offered_epochs", {})
        control = (self._turn_in_progress, self._preparing, self._refreshing_copies,
                   getattr(self, "_movement_depth", 0), getattr(self, "_resolution_depth", 0),
                   tuple(sorted(offered.items())),
                   tuple((rid, offered.get(rid) == epoch) for rid in sorted(self.racers)))
        key = (self._state_key(include_pending=False), roster, self.score_ledger.snapshot(),
               len(self.turns), self.race_index, tuple(self.turn_order),
               tuple(self.remaining_abilities), tuple(self.egg_drawn), tuple(self.previous_winners),
               control)
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def _homogeneous_baby_scoocher_cycle_completed(self, snapshot):
        """Keep every original notification; prove only its new descendants cyclic.

        Each original FIFO item must actually execute its own complete period.
        The remaining queue must consist exactly of the two genuine descendants
        of each such period, in order. An unrelated suffix cannot be discarded.
        """
        _, event_start, original_pending = snapshot
        events = self.events[event_start:]
        if len(events) != 10 * len(original_pending):
            return False
        if not original_pending:
            return False
        s, b = original_pending[0].racer_id, original_pending[0].source_id
        queued_ids = []
        for index, original in enumerate(original_pending):
            effect, move, baby, queued, stop, arrow, baby_again, queued_again, stop_again, loop = events[index * 10:(index + 1) * 10]
            if [(event.kind, event.racer_id) for event in (
                effect, move, baby, queued, stop, arrow, baby_again, queued_again, stop_again, loop,
            )] != [
                ("ability_effect", s), ("move", s), ("ability_effect", b), ("effect_queued", s),
                ("stop", s), ("move", s), ("ability_effect", b), ("effect_queued", s),
                ("stop", s), ("loop_stopped", s),
            ]:
                return False
            if ((effect.source_id, effect.ability, effect.amount, effect.parent_sequence) != (
                    s, Ability.SCOOCHER.value, 1, original.queued_sequence)
                    or (move.before, move.after, move.amount, move.movement_kind, move.source_id,
                        move.ability, move.parent_sequence) != (11, 12, 1, "ability", s, Ability.SCOOCHER.value, effect.sequence)
                    or (arrow.before, arrow.after, arrow.amount, arrow.movement_kind, arrow.source_id,
                        arrow.ability, arrow.parent_sequence) != (11, 12, 1, "track", None, None, stop.sequence)):
                return False
            for power, qualification, landing, parent in (
                (baby, queued, stop, effect.sequence), (baby_again, queued_again, stop_again, stop.sequence),
            ):
                if ((power.source_id, power.ability, power.parent_sequence) != (b, Ability.HUGE_BABY.value, parent)
                        or (qualification.source_id, qualification.ability, qualification.parent_sequence) != (
                            b, Ability.SCOOCHER.value, power.sequence)
                        or (landing.before, landing.after, landing.parent_sequence) != (12, 11, power.sequence)):
                    return False
                queued_ids.append(qualification.sequence)
            if (loop.reason, loop.parent_sequence) != ("forced_arrow_cycle_with_deferred_queue_proved", stop_again.sequence):
                return False
        return tuple(effect.queued_sequence for effect in self.pending_effects) == tuple(queued_ids)

    def _baba_baby_queue_cycle_key(self, actors=None):
        """A closed two-letter queue, including genuinely inert Trip jobs.

        q is Baby's notification to Scoocher; t is Baba's already-qualified
        Trip of that *already tripped* Scoocher. At the clean cut, the actual
        step proof must establish q -> [q,t,q,t] and t -> [], with identical
        complete mechanical state and no input or score. Pending count/IDs do
        not control role/track rules. At least one q is required: a t-only
        queue is finite and simply drains. No initial qualification is erased.
        """
        if (self.track != WILD_WILDS or len(self.racers) != 4 or self.termination_reason is not None
                or self._copy_locks or self._refreshing_copies or self._preparing or not self._turn_in_progress
                or hasattr(self, "_arrow_visits") or hasattr(self, "_arrow_queue_visits")
                or getattr(self, "_movement_depth", 0) != 0):
            return None
        if (len({racer.ability for racer in self.racers.values()}) != 4
                or any(not isinstance(racer.ability, Ability) or not racer.pre_race_resolved
                       or type(racer.position) is not int or type(racer.racer_id) is not int
                       for racer in self.racers.values())):
            return None
        pending = tuple(self.pending_effects)
        if not pending or len({effect.queued_sequence for effect in pending}) != len(pending):
            return None
        if actors is None:
            q = next((effect for effect in pending if effect.kind == "move"), None)
            t = next((effect for effect in pending if effect.kind == "trip"), None)
            if q is None or t is None:
                return None
            actors = (q.racer_id, q.source_id, t.racer_id)
        s, b, y = actors
        if len({s, b, y}) != 3 or any(rid not in self.racers for rid in actors):
            return None
        scoocher, baby, baba = (self.racers[rid] for rid in actors)
        if (any(racer.status != RacerStatus.ACTIVE for racer in (scoocher, baby, baba))
                or (scoocher.position, baby.position, baba.position) != (11, 12, 11)
                or not scoocher.tripped
                or tuple(self._ability(racer) for racer in (scoocher, baby, baba)) != (
                    Ability.SCOOCHER, Ability.HUGE_BABY, Ability.BABA_YAGA)):
            return None
        q_semantics = (s, b, "move", 1, Ability.SCOOCHER.value, None)
        t_semantics = (y, y, "trip", 1, Ability.BABA_YAGA.value, s)
        q_count = 0
        for effect in pending:
            semantics = (effect.racer_id, effect.source_id, effect.kind, effect.steps, effect.ability, effect.target_id)
            if semantics not in {q_semantics, t_semantics}:
                return None
            if not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(self.events):
                return None
            trigger, queued = (self.events[sequence - 1] for sequence in (effect.trigger_sequence, effect.queued_sequence))
            if (queued.kind, queued.racer_id, queued.source_id, queued.target_id,
                    queued.ability, queued.parent_sequence) != (
                        "effect_queued", effect.racer_id, effect.source_id, effect.target_id,
                        effect.ability, trigger.sequence):
                return None
            if semantics == q_semantics:
                q_count += 1
                if (trigger.kind, trigger.racer_id, trigger.source_id, trigger.ability) != (
                    "ability_effect", b, b, Ability.HUGE_BABY.value,
                ):
                    return None
            elif trigger.kind != "stop" or trigger.racer_id not in {s, y} or trigger.after != 11:
                return None
        if not q_count:
            return None
        roster = tuple((rid, tuple((item.name, getattr(racer, item.name)) for item in fields(racer)))
                       for rid, racer in sorted(self.racers.items()))
        actual_kinds = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                        "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
        epoch = (len(self.turns), next((event.sequence for event in reversed(self.events)
                                       if event.kind in actual_kinds), 0))
        offered = getattr(self, "_duel_offered_epochs", {})
        control = (self._turn_in_progress, self._preparing, self._refreshing_copies,
                   getattr(self, "_movement_depth", 0), getattr(self, "_resolution_depth", 0),
                   tuple(sorted(offered.items())),
                   tuple((rid, offered.get(rid) == epoch) for rid in sorted(self.racers)))
        key = (actors, self._state_key(include_pending=False), roster, self.score_ledger.snapshot(),
               len(self.turns), self.race_index, tuple(self.turn_order), tuple(self.remaining_abilities),
               tuple(self.egg_drawn), tuple(self.previous_winners), control)
        try:
            hash(key)
        except TypeError:
            return None
        return key

    def _baba_baby_queue_step(self, pending, queue_before, event_start, actors):
        """Verify the job that really popped, including silent t consumption."""
        s, b, y = actors
        events, current = self.events[event_start:], tuple(self.pending_effects)
        if pending.kind == "trip":
            # The key proved active source/target, correct identity/power and
            # tripped=True *before* execution. _trip must now be a real no-op.
            return () if not events and current == queue_before[1:] else None
        if len(events) != 12:
            return None
        effect, move, baby, q, stop, t, arrow, baby_again, q_again, stop_again, t_again, loop = events
        if [(event.kind, event.racer_id) for event in events] != [
            ("ability_effect", s), ("move", s), ("ability_effect", b), ("effect_queued", s),
            ("stop", s), ("effect_queued", y), ("move", s), ("ability_effect", b),
            ("effect_queued", s), ("stop", s), ("effect_queued", y), ("loop_stopped", s),
        ]:
            return None
        if ((effect.source_id, effect.ability, effect.amount, effect.parent_sequence) != (
                s, Ability.SCOOCHER.value, 1, pending.queued_sequence)
                or (move.before, move.after, move.amount, move.movement_kind, move.source_id,
                    move.ability, move.parent_sequence) != (11, 12, 1, "ability", s, Ability.SCOOCHER.value, effect.sequence)
                or (arrow.before, arrow.after, arrow.amount, arrow.movement_kind, arrow.source_id,
                    arrow.ability, arrow.parent_sequence) != (11, 12, 1, "track", None, None, stop.sequence)
                or (loop.reason, loop.parent_sequence) != ("forced_arrow_cycle_with_deferred_queue_proved", stop_again.sequence)):
            return None
        derived = []
        for power, notification, landing, trip_job, parent in (
            (baby, q, stop, t, effect.sequence), (baby_again, q_again, stop_again, t_again, stop.sequence),
        ):
            if ((power.source_id, power.ability, power.parent_sequence) != (b, Ability.HUGE_BABY.value, parent)
                    or (notification.source_id, notification.target_id, notification.ability, notification.parent_sequence) != (
                        b, None, Ability.SCOOCHER.value, power.sequence)
                    or (landing.before, landing.after, landing.parent_sequence) != (12, 11, power.sequence)
                    or (trip_job.source_id, trip_job.target_id, trip_job.ability, trip_job.parent_sequence) != (
                        y, s, Ability.BABA_YAGA.value, landing.sequence)):
                return None
            derived.extend((PendingEffect(s, b, power.sequence, notification.sequence),
                            PendingEffect(y, y, landing.sequence, trip_job.sequence, "trip", 1, Ability.BABA_YAGA.value, s)))
        return tuple(derived) if current == (*queue_before[1:], *derived) else None

    def _inert_duel_spectator_power(self, actor):
        """Accept only a static resolved spectator, never an active Copy.

        Finished/eliminated powers are excluded by every prefix listener's
        active-racer filter. Their complete original fields still enter the
        state key. Active Egg/Twin may carry a completed, genuinely recorded
        inheritance chain ending in one of the proved passive powers.
        """
        if actor.status != RacerStatus.ACTIVE:
            return True
        if (self._is_copy(actor) or actor.copied_target is not None or actor.copied_ability is not None
                or actor.position in {22, 23, 24}):
            return False
        effective = self._ability(actor)
        if effective not in DUEL_PREFIX_PASSIVE_SPECTATOR_ABILITIES:
            return False
        if actor.ability in {Ability.EGG, Ability.TWIN}:
            inherited = next((event for event in reversed(self.events)
                              if event.kind == "inherit" and event.racer_id == actor.racer_id), None)
            if (actor.inherited_ability != effective or inherited is None
                    or inherited.source_id != actor.racer_id or inherited.ability != effective.value):
                return False
        elif actor.inherited_ability is not None:
            return False
        return (actor.prediction is None or effective == Ability.MASTERMIND
                and type(actor.prediction) is int and actor.prediction in self.racers)

    def _inert_duel_racer_status(self, actor):
        """Retain a real existing first place/elimination; never invent one."""
        if (not isinstance(actor.ability, Ability) or not actor.pre_race_resolved
                or type(actor.position) is not int or type(actor.racer_id) is not int
                or not 0 <= actor.position <= self.track.finish_position
                or actor.prediction is not None and (type(actor.prediction) is not int or actor.prediction not in self.racers)):
            return False
        places = tuple(index + 1 for index, rid in enumerate(self.finish_order) if rid == actor.racer_id)
        if actor.status == RacerStatus.ACTIVE:
            return (not places and actor.finish_place is None and not actor.finish_places
                    and actor.position < self.track.finish_position)
        if actor.status == RacerStatus.FINISHED:
            return (places == (1,) and actor.finish_place == 1 and actor.finish_places == places
                    and actor.position == self.track.finish_position
                    and any(event.kind == "finish" and event.racer_id == actor.racer_id and event.amount == 1
                            for event in self.events))
        if actor.status == RacerStatus.ELIMINATED:
            return (not places and actor.finish_place is None and not actor.finish_places
                    and any(event.kind == "eliminate" and event.racer_id == actor.racer_id for event in self.events))
        return False

    def _inert_duel_principal_power(self, actor, power):
        """A principal is original or has a fully recorded static inheritance.

        Egg/Twin resolve before the race; their original card and complete
        chain remain in the state. Copy is dynamic, even when its current
        borrowed ability happens to equal D or R, and is never admitted.
        """
        if (actor.status != RacerStatus.ACTIVE or not actor.pre_race_resolved
                or self._ability(actor) != power or self._is_copy(actor)
                or actor.copied_target is not None or actor.copied_ability is not None
                or actor.prediction is not None):
            return False
        if actor.ability == power:
            return actor.inherited_ability is None
        if actor.ability not in {Ability.EGG, Ability.TWIN} or actor.inherited_ability != power:
            return False
        chain = [event for event in self.events if event.kind == "inherit" and event.racer_id == actor.racer_id]
        if not chain:
            return False
        decisions = {record.event_sequence: record for record in self.decisions}
        current, visited = actor.ability, set()
        for event in chain:
            if current not in {Ability.EGG, Ability.TWIN} or current in visited:
                return False
            visited.add(current)
            record = decisions.get(event.parent_sequence)
            kind = ActionKind.EGG_CHOOSE if current == Ability.EGG else ActionKind.TWIN_CHOOSE
            choice = Action(kind, actor.racer_id, actor.racer_id, value=event.ability)
            if (event.source_id != actor.racer_id or record is None
                    or record.request.window != DecisionWindow.BEFORE_RACE
                    or record.request.player_id != actor.racer_id or record.action != choice
                    or choice not in record.request.legal_actions
                    or event.sequence != record.event_sequence + 1):
                return False
            try:
                current = Ability(event.ability)
            except (ValueError, TypeError):
                return False
        return current == power

    def _inert_duel_principals(self):
        # Physical card uniqueness is distinct from effective dispatch. Never
        # let a dict silently overwrite a second effective D or R.
        if len({actor.ability for actor in self.racers.values()}) != 4:
            return None
        duelists = [actor for actor in self.racers.values()
                    if actor.status == RacerStatus.ACTIVE and self._ability(actor) == Ability.DUELIST]
        romantics = [actor for actor in self.racers.values()
                     if actor.status == RacerStatus.ACTIVE and self._ability(actor) == Ability.ROMANTIC]
        if len(duelists) != 1 or len(romantics) != 1:
            return None
        d, r = duelists[0], romantics[0]
        if not self._inert_duel_principal_power(d, Ability.DUELIST) or not self._inert_duel_principal_power(r, Ability.ROMANTIC):
            return None
        return d, r

    def _inert_duel_romantic_state(self, owner_depth):
        """The verified all-choice/all-dice-inert Wild mechanical class.

        D and R share22. KEEP changes no mechanics. Each of the 36 duel dice
        pairs moves D or R 22->24->22. Each active spectator's static original
        or resolved inherited power must be in the proved passive set, away
        from22/23/24. Already inactive spectators cannot listen to this prefix.
        D/R themselves may also carry a real immutable Egg/Twin inheritance.
        All original identities, inheritance, places and scores are retained.
        Duel rolls are not main rolls and do not dispatch main-roll responses.
        R's move also returns22 and recreates the clockwise D/R pair. Either
        FIFO order therefore stays closed with at least one R task.

        Decision/dice history is retained, never fabricated. The only control
        projections are provenance IDs, the already-proved D epoch equality
        relation, and the known owner/one-nested-Duel drain depth. All actual
        racer fields, score entries and other controls must remain identical.
        """
        if (self.track != WILD_WILDS or len(self.racers) != 4 or self.termination_reason is not None
                or len(self.finish_order) > 1 or len(set(self.finish_order)) != len(self.finish_order)
                or any(rid not in self.racers for rid in self.finish_order)
                or self._copy_locks or self._refreshing_copies or self._preparing
                or not self._turn_in_progress or hasattr(self, "_arrow_visits")
                or hasattr(self, "_arrow_queue_visits") or getattr(self, "_movement_depth", 0) != 0
                or getattr(self, "_resolution_depth", 0) not in {owner_depth, owner_depth + 1, owner_depth + 2}):
            return None
        principals = self._inert_duel_principals()
        if principals is None:
            return None
        d, r = principals
        spectators = [actor for actor in self.racers.values() if actor is not d and actor is not r]
        if ((d.position, r.position) != (22, 22)
                or any(not self._inert_duel_spectator_power(actor) for actor in spectators)):
            return None
        if any(not self._inert_duel_racer_status(actor) for actor in self.racers.values()):
            return None
        if set(getattr(self, "_duel_offered_epochs", {})) - {d.racer_id}:
            return None
        roster = tuple((rid, tuple((item.name, getattr(actor, item.name)) for item in fields(actor)))
                       for rid, actor in sorted(self.racers.items()))
        return (self._state_key(include_pending=False, include_decisions=False), roster, self.score_ledger.snapshot(),
                len(self.turns), self.race_index, tuple(self.turn_order), tuple(self.remaining_abilities),
                tuple(self.egg_drawn), tuple(self.previous_winners), owner_depth)

    def _duel_actual_epoch(self):
        kinds = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                 "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
        return (len(self.turns), next((event.sequence for event in reversed(self.events) if event.kind in kinds), 0))

    def _duel_romantic_pair(self, d, r, trigger, queued):
        if len(queued) != 2 or trigger.kind != "stop" or trigger.racer_id not in {d, r} or trigger.after != 22:
            return None
        result = []
        order = [rid for rid in self._clockwise_order() if rid in {d, r}]
        for event, actor in zip(queued, order):
            kind, steps, ability = (("duel", 0, Ability.DUELIST.value) if actor == d
                                    else ("move", 2, Ability.ROMANTIC.value))
            if (event.kind, event.racer_id, event.source_id, event.target_id, event.ability, event.parent_sequence) != (
                "effect_queued", actor, actor, actor, ability, trigger.sequence,
            ):
                return None
            result.append(PendingEffect(actor, actor, trigger.sequence, event.sequence, kind, steps, ability, actor))
        return tuple(result)

    def _new_inert_duel_romantic_cohort(self):
        depth = getattr(self, "_resolution_depth", 0)
        state = self._inert_duel_romantic_state(depth)
        pending = tuple(self.pending_effects)
        if state is None or len(pending) != 2:
            return None
        principals = self._inert_duel_principals()
        if principals is None:
            return None
        d, r = (actor.racer_id for actor in principals)
        if getattr(self, "_duel_offered_epochs", {}).get(d) == self._duel_actual_epoch():
            return None
        if pending[0].trigger_sequence != pending[1].trigger_sequence:
            return None
        if any(not 1 <= effect.trigger_sequence < effect.queued_sequence <= len(self.events) for effect in pending):
            return None
        trigger = self.events[pending[0].trigger_sequence - 1]
        queued = [self.events[effect.queued_sequence - 1] for effect in pending]
        if self._duel_romantic_pair(d, r, trigger, queued) != pending:
            return None
        return {"state": state, "depth": depth, "d": d, "r": r, "originals": pending,
                "d_job": next(effect for effect in pending if effect.racer_id == d),
                "r_job": next(effect for effect in pending if effect.racer_id == r),
                "popped": [], "finished": set(), "derived": [], "event_start": len(self.events),
                "raw_epoch_before": dict(getattr(self, "_duel_offered_epochs", {})),
                "expected_epoch": dict(getattr(self, "_duel_offered_epochs", {})),
                "duel_start": None, "choice": None, "branch": None, "invalid": False, "closed": False}

    def _inert_duel_job_started(self, pending, queue_before, event_start):
        proof = getattr(self, "_inert_duel_romantic", None)
        if proof is None or proof["invalid"] or proof["closed"]:
            return
        index = len(proof["popped"])
        if (index >= 2 or pending != proof["originals"][index]
                or queue_before != (*proof["originals"][index:], *proof["derived"])):
            proof["invalid"] = True
            return
        proof["popped"].append(pending)
        if pending == proof["d_job"]:
            proof["duel_start"] = event_start

    def _inert_duel_choice_prefix(self, source, trigger_sequence, event_start, choice, rolls=(), winner=None):
        """A Duel job's prefix finishes before its recursive drain begins."""
        proof = getattr(self, "_inert_duel_romantic", None)
        if proof is None or proof["invalid"] or proof["closed"]:
            return
        d, r = proof["d"], proof["r"]
        events = self.events[event_start:]
        record = self.decisions[-1]
        expected = {Action(ActionKind.KEEP), Action(ActionKind.DUEL, d, r)}
        if (source.racer_id != d or trigger_sequence != proof["d_job"].queued_sequence
                or event_start != proof["duel_start"] or not proof["popped"] or proof["popped"][-1] != proof["d_job"]
                or getattr(self, "_resolution_depth", 0) != proof["depth"] + 1
                or self._inert_duel_romantic_state(proof["depth"]) != proof["state"]
                or record.request.player_id != d or record.request.window != DecisionWindow.DUEL
                or set(record.request.legal_actions) != expected or len(record.request.legal_actions) != 2
                or record.action != choice or not events
                or (events[0].sequence, events[0].kind, events[0].racer_id, events[0].decision_id,
                    events[0].action_kind, events[0].source_id, events[0].target_id, events[0].parent_sequence) != (
                        record.event_sequence, "decision", d, record.request.decision_id,
                        choice.kind.value, choice.source_id, choice.target_id, trigger_sequence)):
            proof["invalid"] = True
            return
        derived = ()
        offered = getattr(self, "_duel_offered_epochs", {})
        if choice.kind == ActionKind.KEEP:
            valid = (len(events) == 1 and events[0].kind == "decision" and not rolls and winner is None
                     and offered == {**proof["expected_epoch"], d: self._duel_actual_epoch()})
            proof["expected_epoch"] = dict(offered)
            branch = "keep"
        else:
            valid = (len(events) == 10 and len(rolls) == 2 and all(type(value) is int and 1 <= value <= 6 for value in rolls)
                     and winner == (d if rolls[0] >= rolls[1] else r) and offered == proof["expected_epoch"])
            if valid:
                decision, effect, first, second, move, stop, arrow, back_stop, qd, qr = events
                valid = ((effect.kind, effect.racer_id, effect.source_id, effect.target_id, effect.ability, effect.parent_sequence) == (
                    "ability_effect", d, d, r, Ability.DUELIST.value, decision.sequence)
                    and [(roll.kind, roll.racer_id, roll.source_id, roll.ability, roll.amount, roll.parent_sequence)
                         for roll in (first, second)] == [
                             ("duel_roll", actor, d, Ability.DUELIST.value, value, effect.sequence)
                             for actor, value in zip((d, r), rolls)]
                    and self._inert_duel_return_move((move, stop, arrow, back_stop), winner, d, Ability.DUELIST.value, effect.sequence))
                derived = self._duel_romantic_pair(d, r, back_stop, (qd, qr))
                valid = valid and derived is not None
            branch = "duel"
        remaining_originals = proof["originals"][len(proof["popped"]):]
        if not valid or tuple(self.pending_effects) != (*remaining_originals, *proof["derived"], *derived):
            proof["invalid"] = True
            return
        proof["finished"].add(proof["d_job"].queued_sequence)
        proof["derived"].extend(derived)
        proof["choice"], proof["branch"] = record, branch
        self._finish_inert_duel_romantic_cohort()

    def _inert_duel_return_move(self, events, actor, source, ability, parent):
        move, stop, arrow, back_stop = events
        return ([(event.kind, event.racer_id, event.before, event.after, event.amount, event.movement_kind,
                  event.source_id, event.ability, event.parent_sequence) for event in events] == [
            ("move", actor, 22, 24, 2, "ability", source, ability, parent),
            ("stop", actor, 22, 24, None, None, None, None, move.sequence),
            ("move", actor, 24, 22, -2, "track", None, None, stop.sequence),
            ("stop", actor, 24, 22, None, None, None, None, arrow.sequence),
        ])

    def _inert_duel_job_finished(self, pending, queue_before, event_start):
        proof = getattr(self, "_inert_duel_romantic", None)
        if proof is None or proof["invalid"] or proof["closed"]:
            return
        if pending == proof["d_job"]:
            if pending.queued_sequence not in proof["finished"]:
                proof["invalid"] = True
            return
        events = self.events[event_start:]
        d, r = proof["d"], proof["r"]
        d_first = proof["originals"][0] == proof["d_job"]
        valid = (pending == proof["r_job"] and len(events) == 7
                 and self._inert_duel_romantic_state(proof["depth"]) == proof["state"]
                 and getattr(self, "_resolution_depth", 0) == proof["depth"] + (2 if d_first and proof["branch"] == "duel" else 0)
                 and getattr(self, "_duel_offered_epochs", {}) == proof["expected_epoch"])
        if not valid:
            proof["invalid"] = True
            return
        effect, move, stop, arrow, back_stop, qd, qr = events
        derived = self._duel_romantic_pair(d, r, back_stop, (qd, qr))
        if ((effect.kind, effect.racer_id, effect.source_id, effect.ability, effect.amount, effect.parent_sequence) != (
                "ability_effect", r, r, Ability.ROMANTIC.value, 2, pending.queued_sequence)
                or not self._inert_duel_return_move((move, stop, arrow, back_stop), r, r, Ability.ROMANTIC.value, effect.sequence)
                or derived is None or tuple(self.pending_effects) != (
                    *proof["originals"][len(proof["popped"]):], *proof["derived"], *derived)):
            proof["invalid"] = True
            return
        proof["finished"].add(pending.queued_sequence)
        proof["derived"].extend(derived)
        self._finish_inert_duel_romantic_cohort()

    def _finish_inert_duel_romantic_cohort(self):
        proof = getattr(self, "_inert_duel_romantic", None)
        if proof is None or proof["invalid"] or proof["closed"]:
            return
        original_ids = {effect.queued_sequence for effect in proof["originals"]}
        if proof["finished"] != original_ids:
            return
        d, r = proof["d"], proof["r"]
        d_first = proof["originals"][0] == proof["d_job"]
        # DR finishes after R's body (one nested drain only for DUEL).
        # RD finishes inside D's actual prefix before its recursive drain.
        # RD+KEEP has already resolved the latest window; every other branch
        # has a new actual move since the saved epoch. Both cases are closed:
        # the new queue contains R, whose next real move refreshes D's epoch.
        expected_depth = proof["depth"] + ((2 if proof["branch"] == "duel" else 0) if d_first else 1)
        expected_resolved_epoch = not d_first and proof["branch"] == "keep"
        epoch = self._duel_actual_epoch()
        offered = getattr(self, "_duel_offered_epochs", {})
        if (proof["popped"] != list(proof["originals"])
                or proof["choice"] is None or self._inert_duel_romantic_state(proof["depth"]) != proof["state"]
                or getattr(self, "_resolution_depth", 0) != expected_depth
                or offered != proof["expected_epoch"] or (offered.get(d) == epoch) != expected_resolved_epoch
                or tuple(self.pending_effects) != tuple(proof["derived"])
                or len(proof["derived"]) != (2 if proof["branch"] == "keep" else 4)
                or original_ids.intersection(effect.queued_sequence for effect in self.pending_effects)):
            proof["invalid"] = True
            return
        evidence = {"projection": "Duel epoch IDs matter only through equality with the latest actual event",
                    "raw_epoch_at_start": proof["raw_epoch_before"], "raw_epoch_before_resolution": dict(offered),
                    "raw_epoch_after_resolution": {**offered, d: epoch},
                    "original_pending_ids": [effect.queued_sequence for effect in proof["originals"]],
                    "new_pending_ids": [effect.queued_sequence for effect in self.pending_effects],
                    "owner_resolution_depth": proof["depth"], "actual_resolution_depth": getattr(self, "_resolution_depth", 0),
                    "fifo_order": "DR" if d_first else "RD",
                    "resolved_epoch_before_resolution": expected_resolved_epoch,
                    "retained_finish_order": list(self.finish_order),
                    "spectators": [{"racer_id": actor.racer_id, "original_ability": actor.ability.value,
                                    "inherited_ability": actor.inherited_ability.value if actor.inherited_ability else None,
                                    "effective_ability": self._ability(actor).value if self._ability(actor) else None,
                                    "status": actor.status.value}
                                   for actor in self.racers.values() if actor.racer_id not in {d, r}]}
        self._emit("loop_stopped", d, source_id=d, target_id=r, reason="inert_duel_romantic_cycle_proved",
                   parent_sequence=proof["event_start"] + 1, decision_id=proof["choice"].request.decision_id,
                   amount=len(self.pending_effects), operation=json.dumps(evidence, sort_keys=True, separators=(",", ":")))
        # This is an O12-resolved opportunity, not an invented KEEP. All
        # ancestor Duel winner moves already returned before their drains;
        # their remaining while guards now unwind this same repeated window.
        # A subsequent real event (or turn) makes epoch() different again.
        offered[d] = epoch
        self._duel_offered_epochs = offered
        self.pending_effects.clear()
        proof["closed"] = True

    def _schedule(self, source, trigger_sequence, *, kind="move", steps=1,
                  target_id=None, ability=None):
        ability = ability or self._ability(source)
        name = ability.value if isinstance(ability, Ability) else ability
        event = self._emit("effect_queued", source.racer_id,
                           source_id=source.racer_id, target_id=target_id,
                           ability=name, parent_sequence=trigger_sequence)
        self.pending_effects.append(PendingEffect(
            source.racer_id, source.racer_id, trigger_sequence, event.sequence,
            kind=kind, steps=steps, ability=name, target_id=target_id,
        ))

    @bounded_resolution
    def _drain_effects(self):
        self._run_resolution_frames(self._drain_effects_frame())

    @copy_baby_duel_scoocher_cycle.drain_scope
    @duel_romantic_scoocher22_cycle.drain_scope
    @scoocher_baby_duel22_cycle.drain_scope
    @inert_duel_scope
    def _drain_effects_frame(self):
        seen = set()
        proved_prefix_snapshots = {copy_banana_cycle: {}, fish_baby_cycle: {}, star_baby_cycle: {},
                                   romantic_queue_cycle: {}, preparing_copy_romantic_cycle: {}, romantic_anchor_cycle: {},
                                   romantic_baby_star_cycle: {}, fish_baby_star_cycle: {}, romantic_baby_arrow_cycle: {},
                                   romantic_duel_copy_cycle: {}, egg_copy_scoocher_cycle: {},
                                   romantic_scoocher_anchor_cycle: {}, romantic_baby_baba_cycle: {},
                                   copy_fish_baby_cycle: {}, centaur_copy_baby_star_cycle: {},
                                   romantic_baby_shared12_cycle: {}, fish_baby_baba4_cycle: {}, fish_baby_arrow11_scoocher_cycle: {}}
        star_snapshots = {}
        inert_copy_snapshots = {}
        inert_egg_snapshots = {}
        baby_scoocher_snapshot = None
        baby_scoocher_proof_rejected = False
        baba_snapshot = None
        baba_consumed, baba_q_count, baba_derived = 0, 0, []
        count = 0
        while self.pending_effects and self.termination_reason is None:
            copy_baby_duel_scoocher_cycle.probe(self)
            duel_romantic_scoocher22_cycle.probe(self)
            scoocher_baby_duel22_cycle.probe(self)
            if not self.pending_effects:
                return
            if getattr(self, "_inert_duel_romantic", None) is None:
                snapshot = self._new_inert_duel_romantic_cohort()
                if snapshot is not None:
                    self._inert_duel_romantic = snapshot
            for proof, snapshots in proved_prefix_snapshots.items():
                state = proof.key(self)
                if state is None:
                    continue
                snapshot = snapshots.get(state)
                certificate = proof.completed(self, snapshot) if snapshot is not None else False
                if certificate:
                    self._emit("loop_stopped", self.turn_order[self.turn_index],
                               reason=proof.REASON, parent_sequence=snapshot.event_start + 1,
                               amount=len(self.pending_effects),
                               operation=json.dumps(certificate, sort_keys=True, separators=(",", ":"))
                               if isinstance(certificate, dict) else None)
                    # Real choices, effects and every original FIFO job ran.
                    # Only the verified fresh cyclic descendants remain.
                    self.pending_effects.clear()
                    return
                if snapshot is None:
                    captured = proof.capture(self, state)
                    if captured is not None:
                        snapshots[state] = captured
            if baba_snapshot is not None and self._baba_baby_queue_cycle_key(baba_snapshot[0][0]) != baba_snapshot[0]:
                baba_snapshot = None
            if baba_snapshot is None:
                baba_key = self._baba_baby_queue_cycle_key()
                if baba_key is not None:
                    baba_snapshot = (baba_key, len(self.events), tuple(self.pending_effects))
                    baba_consumed, baba_q_count, baba_derived = 0, 0, []
            baby_key = self._homogeneous_baby_scoocher_cycle_key()
            if baby_scoocher_snapshot is not None:
                if baby_key != baby_scoocher_snapshot[0]:
                    baby_scoocher_snapshot = None
                    baby_scoocher_proof_rejected = True
                elif len(self.events) - baby_scoocher_snapshot[1] >= 10 * len(baby_scoocher_snapshot[2]):
                    if self._homogeneous_baby_scoocher_cycle_completed(baby_scoocher_snapshot):
                        self._emit("loop_stopped", self.turn_order[self.turn_index],
                                   reason="homogeneous_scoocher_baby_queue_cycle_proved",
                                   parent_sequence=baby_scoocher_snapshot[1] + 1,
                                   amount=len(self.pending_effects))
                        # All original qualifications actually ran. Only their
                        # proved, freshly derived cyclic descendants remain.
                        self.pending_effects.clear()
                        return
                    baby_scoocher_snapshot = None
                    baby_scoocher_proof_rejected = True
            if baby_key is not None and baby_scoocher_snapshot is None and not baby_scoocher_proof_rejected:
                baby_scoocher_snapshot = (baby_key, len(self.events), tuple(self.pending_effects))
            state = self._state_key()
            if state in seen:
                self._emit("loop_stopped", self.turn_order[self.turn_index],
                           reason="complete_state_and_pending_effects_repeated")
                self.pending_effects.clear()
                return
            seen.add(state)
            mechanical = self._state_key(include_scores=False)
            snapshot = star_snapshots.get(mechanical)
            if snapshot is not None and self._star_cycle_completed(snapshot):
                self._emit("loop_stopped", self.turn_order[self.turn_index],
                           reason="forced_star_cycle_proved",
                           parent_sequence=snapshot[0] + 1,
                           amount=sum(self.score_ledger.totals()[rid] - total
                                      for rid, total in snapshot[2].items()))
                self.pending_effects.clear()
                return
            if snapshot is None:
                star_snapshots[mechanical] = (
                    len(self.events), len(self.score_ledger.entries),
                    self.score_ledger.totals(),
                )
            copy_key = self._inert_copy_romantic_cycle_key()
            if copy_key is not None:
                snapshot = inert_copy_snapshots.get(copy_key)
                if snapshot is not None and self._inert_copy_romantic_cycle_completed(snapshot):
                    certificate = copy_banana_cycle.alias_certificate(self, snapshot)
                    self._emit("loop_stopped", self.turn_order[self.turn_index],
                               reason="inert_copy_choice_cycle_proved", parent_sequence=snapshot[0] + 1,
                               **({"amount": len(self.pending_effects),
                                   "operation": json.dumps(certificate, sort_keys=True, separators=(",", ":"))}
                                  if certificate is not None else {}))
                    # Only this period's one or two cyclic qualifications
                    # remain; the original FIFO was consumed in full.
                    self.pending_effects.clear()
                    return
                if snapshot is None:
                    inert_copy_snapshots[copy_key] = (
                        len(self.events), len(self.decisions), self.score_ledger.snapshot(), tuple(self.pending_effects),
                    )
            egg_key = self._inert_egg_romantic_cycle_key()
            if egg_key is not None:
                snapshot = inert_egg_snapshots.get(egg_key)
                certificate = self._inert_egg_romantic_cycle_completed(snapshot) if snapshot is not None else False
                if certificate:
                    self._emit("loop_stopped", self.turn_order[self.turn_index],
                               reason="inert_egg_romantic_copy_choice_cycle_proved",
                               parent_sequence=snapshot[0] + 1,
                               **({"amount": len(self.pending_effects),
                                   "operation": json.dumps(certificate, sort_keys=True, separators=(",", ":"))}
                                  if isinstance(certificate, dict) else {}))
                    # The original qualification ran; only its proved cyclic
                    # descendant is removed. Preserve the actual Copy choice.
                    self.pending_effects.clear()
                    return
                if snapshot is None:
                    inert_egg_snapshots[egg_key] = (
                        len(self.events), len(self.decisions), self.score_ledger.snapshot(), tuple(self.pending_effects),
                    )
            count += 1
            if count > getattr(self, "max_action_events", 100_000):
                self._action_diagnostic("action_event_budget_exceeded")
            baba_before = tuple(self.pending_effects) if baba_snapshot is not None else None
            duel_before = tuple(self.pending_effects) if getattr(self, "_inert_duel_romantic", None) is not None else None
            duel_copy_snapshots = proved_prefix_snapshots[romantic_duel_copy_cycle]
            duel_copy_before = tuple(self.pending_effects) if duel_copy_snapshots else None
            romantic_baba_snapshots = proved_prefix_snapshots[romantic_baby_baba_cycle]
            romantic_baba_before = tuple(self.pending_effects) if romantic_baba_snapshots else None
            fish_baba_snapshots = proved_prefix_snapshots[fish_baby_baba4_cycle]
            fish_baba_before = tuple(self.pending_effects) if fish_baba_snapshots else None
            arrow_fifo_snapshots = proved_prefix_snapshots[fish_baby_arrow11_scoocher_cycle]
            arrow_fifo_before = tuple(self.pending_effects) if arrow_fifo_snapshots else None
            baba_event_start = len(self.events)
            pending = self.pending_effects.popleft()
            self._inert_duel_job_started(pending, duel_before, baba_event_start)
            for snapshot in duel_copy_snapshots.values():
                romantic_duel_copy_cycle.job_started(self, snapshot, pending, duel_copy_before, baba_event_start)
            for snapshot in romantic_baba_snapshots.values():
                romantic_baby_baba_cycle.job_started(self, snapshot, pending, romantic_baba_before, baba_event_start)
            for snapshot in fish_baba_snapshots.values():
                fish_baby_baba4_cycle.job_started(self, snapshot, pending, fish_baba_before, baba_event_start)
            for snapshot in arrow_fifo_snapshots.values():
                fish_baby_arrow11_scoocher_cycle.job_started(self, snapshot, pending, arrow_fifo_before, baba_event_start)
            source = self.racers[pending.racer_id]
            target = self.racers[pending.target_id if pending.target_id is not None else pending.racer_id]
            if source.status != RacerStatus.ACTIVE or target.status != RacerStatus.ACTIVE:
                self._cancel_effect(pending, "racer_inactive")
                continue
            if pending.kind == "move":
                self._move(target, pending.steps, movement_kind="ability",
                           source_id=source.racer_id, ability=pending.ability,
                           parent_sequence=pending.queued_sequence, notify_ability=True)
            elif pending.kind == "trip":
                self._trip(target, source_id=source.racer_id, ability=pending.ability,
                           parent_sequence=pending.queued_sequence, notify_ability=True)
            elif pending.kind == "eliminate":
                effect = self._emit("ability_effect", source.racer_id,
                                    source_id=source.racer_id, target_id=target.racer_id,
                                    ability=pending.ability, parent_sequence=pending.queued_sequence)
                self._queue_scoochers(effect)
                target.status = RacerStatus.ELIMINATED
                self._emit("eliminate", target.racer_id, source_id=source.racer_id,
                           ability=pending.ability, parent_sequence=effect.sequence)
                self._check_end()
            elif pending.kind == "duel" and hasattr(self, "_offer_duel"):
                yield self._offer_duel_frame(source, pending.queued_sequence)
            else:
                raise RuntimeError(f"Unsupported pending effect: {pending.kind}")
            self._inert_duel_job_finished(pending, duel_before, baba_event_start)
            for snapshot in duel_copy_snapshots.values():
                romantic_duel_copy_cycle.job_finished(self, snapshot, pending, duel_copy_before, baba_event_start)
            for snapshot in romantic_baba_snapshots.values():
                romantic_baby_baba_cycle.job_finished(self, snapshot, pending, romantic_baba_before, baba_event_start)
            for snapshot in fish_baba_snapshots.values():
                fish_baby_baba4_cycle.job_finished(self, snapshot, pending, fish_baba_before, baba_event_start)
            for snapshot in arrow_fifo_snapshots.values():
                fish_baby_arrow11_scoocher_cycle.job_finished(self, snapshot, pending, arrow_fifo_before, baba_event_start)
            if baba_snapshot is not None:
                derived = self._baba_baby_queue_step(pending, baba_before, baba_event_start, baba_snapshot[0][0])
                if (self._baba_baby_queue_cycle_key(baba_snapshot[0][0]) != baba_snapshot[0] or derived is None
                        or pending != baba_snapshot[2][baba_consumed]):
                    baba_snapshot = None
                    continue
                baba_consumed += 1
                baba_q_count += pending.kind == "move"
                baba_derived.extend(derived)
                if baba_consumed == len(baba_snapshot[2]):
                    # Count actual FIFO pops, not only the 12*nq visible
                    # events: an original silent t may still follow the last q.
                    if (baba_q_count > 0 and len(baba_derived) == 4 * baba_q_count
                            and tuple(self.pending_effects) == tuple(baba_derived)
                            and not {effect.queued_sequence for effect in baba_snapshot[2]}.intersection(
                                effect.queued_sequence for effect in self.pending_effects)):
                        self._emit("loop_stopped", self.turn_order[self.turn_index],
                                   reason="baba_baby_queue_cycle_proved", parent_sequence=baba_snapshot[1] + 1,
                                   amount=len(self.pending_effects))
                        self.pending_effects.clear()
                        return
                    baba_snapshot = None

    def _action_diagnostic(self, reason):
        from .engine import DiagnosticLimitExceeded

        trace = self.trace()
        trace["diagnostic_failure"] = reason
        raise DiagnosticLimitExceeded(trace)

    def _path(self, racer, steps):
        if not steps:
            return racer.position, ()
        direction = 1 if steps > 0 else -1
        position = racer.position
        skipped = []
        occupied = {r.position for r in self._active() if r.racer_id != racer.racer_id}
        for _ in range(abs(steps)):
            position += direction
            if self._ability(racer) == Ability.LEAPTOAD:
                while 0 < position < self.track.finish_position and position in occupied:
                    skipped.append(position)
                    position += direction
        return position, tuple(skipped)

    def _blocked(self, racer, raw_end, steps, *, parent_sequence, movement_kind):
        if steps <= 0 or raw_end <= self.track.finish_position:
            return False
        for rid in self._clockwise_order():
            source = self.racers[rid]
            if source.racer_id != racer.racer_id and source.status == RacerStatus.ACTIVE and self._ability(source) == Ability.STICKLER:
                effect = self._emit("move_blocked", racer.racer_id,
                                    source_id=source.racer_id, ability=Ability.STICKLER.value,
                                    before=racer.position, after=racer.position, amount=steps,
                                    parent_sequence=parent_sequence, movement_kind=movement_kind)
                self._queue_scoochers(effect)
                return True
        return False

    def _followers(self, leader, steps, parent_sequence, excluded=()):
        followers = []
        for rid in self._clockwise_order():
            fish = self.racers[rid]
            if (rid == leader.racer_id or rid in excluded or fish.status != RacerStatus.ACTIVE
                    or self._ability(fish) != Ability.SUCKERFISH or fish.position != leader.position):
                continue
            choice = self._choose(
                fish, DecisionWindow.MOVE_START,
                (Action(ActionKind.KEEP), Action(ActionKind.SUCKERFISH_FOLLOW, rid, leader.racer_id)),
                roll=None, move_value=steps, parent_sequence=parent_sequence,
            )
            if choice.kind == ActionKind.SUCKERFISH_FOLLOW:
                followers.append(fish)
        return followers

    @bounded_resolution
    def _move(self, racer, steps, *, movement_kind="main", source_id=None,
              ability=None, parent_sequence=None, notify_ability=False):
        if self.termination_reason is not None or racer.status != RacerStatus.ACTIVE or not steps:
            return
        depth = getattr(self, "_movement_depth", 0)
        if depth >= 250:
            self._action_diagnostic("nested_movement_budget_exceeded")
        self._movement_depth = depth + 1
        try:
            raw_end, skipped = self._path(racer, steps)
            if self._blocked(racer, raw_end, steps, parent_sequence=parent_sequence, movement_kind=movement_kind):
                return
            end = max(0, min(raw_end, self.track.finish_position))
            if end == racer.position:
                return
            followers = self._followers(racer, steps, parent_sequence)
            if notify_ability:
                effect = self._emit("ability_effect", racer.racer_id, source_id=source_id,
                                    ability=ability, amount=steps, parent_sequence=parent_sequence)
                self._queue_scoochers(effect)
                parent_sequence = effect.sequence
            for space in skipped:
                effect = self._emit("ability_effect", racer.racer_id, source_id=racer.racer_id,
                                    ability=Ability.LEAPTOAD.value, amount=1, after=space,
                                    parent_sequence=parent_sequence)
                self._queue_scoochers(effect)
            mappings = {racer.racer_id: end}
            movement = {racer.racer_id: (steps, movement_kind, source_id, ability, parent_sequence)}
            for fish in followers:
                effect = self._emit("ability_effect", fish.racer_id, source_id=fish.racer_id,
                                    target_id=racer.racer_id, ability=Ability.SUCKERFISH.value,
                                    amount=steps, parent_sequence=parent_sequence)
                self._queue_scoochers(effect)
                mappings[fish.racer_id] = end
                movement[fish.racer_id] = (steps, "ability", fish.racer_id, Ability.SUCKERFISH.value, effect.sequence)
            self._arrive(mappings, movement=movement, parent_sequence=parent_sequence)
        finally:
            self._movement_depth = depth
        if depth == 0:
            self._refresh_copies(allow_choice=True)

    def _move_batch(self, positions, *, source_id, ability, parent_sequence=None,
                    mode="move", notify_ability=False, movement_kind="ability"):
        if mode not in {"move", "warp", "put"}:
            raise ValueError("batch mode must be move, warp, or put")
        targets = {}
        movement = {}
        for rid, requested in positions.items():
            racer = self.racers[rid]
            if racer.status != RacerStatus.ACTIVE:
                continue
            if mode != "move":
                destination = max(0, min(requested, self.track.finish_position))
                if destination != racer.position:
                    targets[rid] = destination
                continue
            steps = requested - racer.position
            raw_end, skipped = self._path(racer, steps)
            if not steps or self._blocked(racer, raw_end, steps, parent_sequence=parent_sequence, movement_kind=movement_kind):
                continue
            end = max(0, min(raw_end, self.track.finish_position))
            if end == racer.position:
                continue
            for space in skipped:
                effect = self._emit("ability_effect", rid, source_id=rid,
                                    ability=Ability.LEAPTOAD.value, amount=1, after=space,
                                    parent_sequence=parent_sequence)
                self._queue_scoochers(effect)
            targets[rid] = end
            movement[rid] = (steps, movement_kind, source_id, ability, parent_sequence)
        if not targets:
            return
        if notify_ability:
            effect = self._emit("ability_effect", source_id, source_id=source_id,
                                ability=ability, parent_sequence=parent_sequence)
            self._queue_scoochers(effect)
            parent_sequence = effect.sequence
            movement = {rid: (*data[:4], effect.sequence) for rid, data in movement.items()}
        if mode == "warp":
            for rid, destination in targets.items():
                self._emit("warp", rid, source_id=source_id, ability=ability,
                           before=self.racers[rid].position, after=destination,
                           parent_sequence=parent_sequence)
        self._arrive(targets, movement=movement if mode == "move" else {}, parent_sequence=parent_sequence)
        self._refresh_copies(allow_choice=True)

    def _party_pull(self, actor):
        positions = {}
        for racer in self._active():
            if racer.racer_id == actor.racer_id or racer.position == actor.position:
                continue
            positions[racer.racer_id] = racer.position + (1 if racer.position < actor.position else -1)
        self._move_batch(positions, source_id=actor.racer_id, ability=Ability.PARTY_ANIMAL.value,
                         notify_ability=True)

    def _warp(self, racer, position, *, source_id, ability, parent_sequence=None):
        if self.termination_reason is not None or racer.status != RacerStatus.ACTIVE:
            return False
        position = max(0, min(position, self.track.finish_position))
        if position == racer.position:
            return False
        before = racer.position
        event = self._emit("warp", racer.racer_id, source_id=source_id, ability=ability,
                           before=before, after=position, parent_sequence=parent_sequence)
        self._arrive({racer.racer_id: position}, movement={}, parent_sequence=event.sequence)
        self._refresh_copies(allow_choice=True)
        return True

    @bounded_resolution
    def _arrive(self, positions, *, movement, parent_sequence=None):
        if self.termination_reason is not None:
            return
        batch_id = self._new_batch()
        finish_context = self._finish_context()
        before = {rid: r.position for rid, r in self.racers.items()}
        passed = {}
        for rid, destination in positions.items():
            if rid in movement:
                passed[rid] = tuple(other.racer_id for other in self._active()
                                    if (before[rid] < before[other.racer_id]
                                        and destination > positions.get(other.racer_id, before[other.racer_id]))
                                    or (before[rid] > before[other.racer_id]
                                        and destination < positions.get(other.racer_id, before[other.racer_id])))
        # Place all direct arrivals before any of their sharing powers can fire.
        for rid, destination in positions.items():
            self.racers[rid].position = destination
        causes = {}
        for rid in positions:
            if rid in movement:
                steps, kind, source_id, ability, parent = movement[rid]
                move = self._emit("move", rid, source_id=source_id, ability=ability,
                                  before=before[rid], after=positions[rid], amount=steps,
                                  parent_sequence=parent, movement_kind=kind, arrival_batch_id=batch_id)
                causes[rid] = move.sequence
                for other_id in passed[rid]:
                    self._emit("passing", rid, target_id=other_id, parent_sequence=move.sequence,
                               arrival_batch_id=batch_id)
            else:
                causes[rid] = parent_sequence
        # Dict insertion order puts a Suckerfish's followed racer first.
        for rid in positions:
            racer = self.racers[rid]
            if racer.status == RacerStatus.ACTIVE and self.track.has_finished(racer.position):
                self._award_place(racer, parent_sequence=causes[rid], finish_context=finish_context)
                self._check_end()
                if self.termination_reason is not None:
                    return
        displaced = set()
        babies = sorted((r for r in self._active() if self._ability(r) == Ability.HUGE_BABY and r.position != 0),
                        key=lambda r: (r.ability != Ability.HUGE_BABY, self._clockwise_order().index(r.racer_id)))
        for baby in babies:
            # Earlier child arrivals may already have put this queued source
            # on Start. Start is explicitly exempt from Huge Baby, including
            # a Copy whose already-started main move still holds that power.
            if baby.status != RacerStatus.ACTIVE or baby.position == 0:
                continue
            # A child arrival at the square behind Baby must not prematurely
            # resolve residents still awaiting the parent's later child batch.
            if baby.racer_id not in positions and not any(
                rid != baby.racer_id and self.racers[rid].position == baby.position
                for rid in positions
            ):
                continue
            occupants = [r for r in self._active() if r.racer_id != baby.racer_id and r.position == baby.position]
            newcomers = [r for r in occupants if r.racer_id in positions]
            residents = [r for r in occupants if r.racer_id not in positions]
            for group in (newcomers, residents):
                # A previous group's complete causal child batch can change
                # source/targets. Only the currently shared, active targets
                # can receive this later Baby placement; never put below Start.
                if baby.status != RacerStatus.ACTIVE or baby.position == 0:
                    break
                group = [target for target in group
                         if target.status == RacerStatus.ACTIVE and target.position == baby.position]
                if not group:
                    continue
                effect = self._emit("ability_effect", baby.racer_id, source_id=baby.racer_id,
                                    ability=Ability.HUGE_BABY.value, parent_sequence=parent_sequence,
                                    arrival_batch_id=batch_id)
                self._queue_scoochers(effect)
                for target in group:
                    displaced.add(target.racer_id)
                    if self._ability(target) == Ability.BANANA and baby.racer_id in positions:
                        self._trip(baby, source_id=target.racer_id, ability=Ability.BANANA.value,
                                   parent_sequence=effect.sequence, notify_ability=True)
                self._arrive({r.racer_id: baby.position - 1 for r in group}, movement={}, parent_sequence=effect.sequence)
                if self.termination_reason is not None:
                    return
        members = [rid for rid in positions if rid not in displaced and self.racers[rid].status == RacerStatus.ACTIVE]
        self._resolve_stops(members, before, passed, causes, batch_id)

    def _capture_powers(self, members, stop_events, passed, batch_id):
        specs = []
        member_set = set(members)
        groups = defaultdict(list)
        for rid in members:
            groups[self.racers[rid].position].append(rid)
        for source in self._active():
            ability = self._ability(source)
            for rid in members:
                racer = self.racers[rid]
                cause = stop_events[rid].sequence
                coarriver = source.racer_id in member_set and source.racer_id != rid
                if ability == Ability.BANANA and source.racer_id in passed.get(rid, ()):
                    specs.append((source, cause, "trip", 1, rid, ability))
                elif ability == Ability.BABA_YAGA:
                    if source.racer_id == rid:
                        for target in self._active():
                            if target.racer_id != rid and target.racer_id not in member_set and target.position == racer.position:
                                specs.append((source, cause, "trip", 1, target.racer_id, ability))
                    elif source.position == racer.position and not coarriver:
                        specs.append((source, cause, "trip", 1, rid, ability))
                elif ability == Ability.MOUTH and source.racer_id == rid:
                    others = [r for r in self._active() if r.racer_id != rid and r.position == racer.position]
                    if len(others) == 1 and others[0].racer_id not in member_set:
                        specs.append((source, cause, "eliminate", 0, others[0].racer_id, ability))
                if ability == Ability.CENTAUR and source.racer_id == rid:
                    for target_id in passed.get(rid, ()):
                        specs.append((source, cause, "move", -2, target_id, ability))
            if ability == Ability.ROMANTIC:
                for position, arriving in groups.items():
                    occupants = [r for r in self._active() if r.position == position]
                    if len(occupants) == 2 and len(arriving) == 1:
                        specs.append((source, stop_events[arriving[0]].sequence, "move", 2, source.racer_id, ability))
            if ability == Ability.DUELIST and hasattr(self, "_offer_duel"):
                for position, arriving in groups.items():
                    others = [r for r in self._active() if r.racer_id != source.racer_id and r.position == source.position]
                    if position == source.position and others and (source.racer_id not in member_set or len(arriving) < len(others) + 1):
                        specs.append((source, stop_events[arriving[0]].sequence, "duel", 0, source.racer_id, ability))
        return specs

    def _resolve_stops(self, members, before, passed, causes, batch_id):
        stops = {rid: self._emit("stop", rid, before=before[rid], after=self.racers[rid].position,
                                 parent_sequence=causes[rid], arrival_batch_id=batch_id) for rid in members}
        # Capture qualifications now, not after an arrow or another power leaves.
        specs = self._capture_powers(members, stops, passed, batch_id)
        # Register now, execute only after track actions complete. Keeping these
        # qualifications in the queue also makes them part of the cycle state.
        for source, cause, kind, steps, target_id, ability in specs:
            self._schedule(source, cause, kind=kind, steps=steps, target_id=target_id, ability=ability)
        arrow_groups = defaultdict(list)
        for rid in members:
            racer = self.racers[rid]
            if self.termination_reason is not None or racer.status != RacerStatus.ACTIVE:
                return
            stop = stops[rid]
            position = racer.position
            if position in self.track.trip_positions:
                self._trip(racer, parent_sequence=stop.sequence)
            if position in self.track.star_positions:
                self._score(racer, 1, None, "star", parent_sequence=stop.sequence)
            arrow = dict(self.track.arrows).get(position)
            if arrow is not None:
                arrow_groups[position].append((rid, arrow))
        for position, group in arrow_groups.items():
            rid, arrow = group[0]
            stop = stops[rid]
            racer = self.racers[rid]
            if self.termination_reason is not None:
                return
            if all(self.racers[member].status != RacerStatus.ACTIVE for member, _ in group):
                continue
            if arrow is not None:
                visits = getattr(self, "_arrow_visits", set())
                top = not hasattr(self, "_arrow_visits")
                self._arrow_visits = visits
                if top:
                    self._arrow_queue_visits = {}
                state = (tuple(member for member, _ in group), position, self._state_key())
                control = (tuple(member for member, _ in group), position,
                           self._state_key(include_pending=False))
                snapshot = self._arrow_queue_visits.get(control)
                try:
                    native_arrow_state = fish_baby_arrow11_scoocher_cycle.arrow_key(self, group, position, stop)
                    native_arrow_snapshot = self._arrow_queue_visits.get((fish_baby_arrow11_scoocher_cycle.CACHE_TAG, native_arrow_state)) if native_arrow_state is not None else None
                    native_arrow_certificate = fish_baby_arrow11_scoocher_cycle.arrow_completed(self, native_arrow_snapshot, stop) if native_arrow_snapshot is not None else False
                    choice_state = fish_baby_arrow11_cycle.key(self, group, position, stop)
                    choice_snapshot = self._arrow_queue_visits.get(
                        (fish_baby_arrow11_cycle.CACHE_TAG, choice_state)) if choice_state is not None else None
                    choice_certificate = (fish_baby_arrow11_cycle.completed(self, choice_snapshot, stop)
                                          if choice_snapshot is not None else False)
                    if native_arrow_certificate:
                        self._emit("loop_stopped", rid, parent_sequence=stop.sequence,
                                   reason=fish_baby_arrow11_scoocher_cycle.ARROW_REASON,
                                   operation=json.dumps(native_arrow_certificate, sort_keys=True, separators=(",", ":")))
                    elif choice_certificate:
                        # Only this repeating synchronous Arrow step stops.
                        # All original and newly qualified FIFO jobs stay owed
                        # and execute after this track call returns.
                        self._emit("loop_stopped", rid, parent_sequence=stop.sequence,
                                   reason=fish_baby_arrow11_cycle.REASON,
                                   operation=json.dumps(choice_certificate, sort_keys=True, separators=(",", ":")))
                    elif state in visits:
                        self._emit("loop_stopped", rid, parent_sequence=stop.sequence,
                                   reason="complete_arrow_state_repeated")
                    elif snapshot is not None and self._queued_arrow_cycle_completed(snapshot):
                        self._emit("loop_stopped", rid, parent_sequence=stop.sequence,
                                   reason="forced_arrow_cycle_with_deferred_queue_proved")
                    else:
                        if native_arrow_state is not None and native_arrow_snapshot is None:
                            self._arrow_queue_visits[(fish_baby_arrow11_scoocher_cycle.CACHE_TAG, native_arrow_state)] = fish_baby_arrow11_scoocher_cycle.arrow_capture(self, native_arrow_state, stop)
                        if choice_state is not None and choice_snapshot is None:
                            self._arrow_queue_visits[(fish_baby_arrow11_cycle.CACHE_TAG, choice_state)] = (
                                fish_baby_arrow11_cycle.capture(self, choice_state, stop))
                        visits.add(state)
                        if snapshot is None:
                            self._arrow_queue_visits[control] = (
                                len(self.events), len(self.score_ledger.entries), tuple(self.pending_effects),
                            )
                        if len(group) == 1:
                            self._move(racer, arrow, movement_kind="track", parent_sequence=stop.sequence)
                        else:
                            self._move_batch(
                                {member: self.racers[member].position + distance for member, distance in group},
                                source_id=None, ability=None, parent_sequence=stop.sequence,
                                movement_kind="track",
                            )
                finally:
                    if top:
                        del self._arrow_visits
                        del self._arrow_queue_visits
        if self.termination_reason is not None:
            return

    def _landing(self, racer, *, before, parent_sequence, passed_ids=()):
        # Compatibility for host actions that have already updated one position.
        batch_id = self._new_batch()
        self._resolve_stops([racer.racer_id], {racer.racer_id: before},
                            {racer.racer_id: tuple(passed_ids)},
                            {racer.racer_id: parent_sequence}, batch_id)
