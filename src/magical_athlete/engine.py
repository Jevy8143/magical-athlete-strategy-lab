"""Race rules and decision windows, shared by fixtures and complete games."""

import hashlib
import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .decisions import (
    Action, ActionKind, DecisionRecord, DecisionRequest, DecisionWindow,
    KeepPolicy, Observation, Policy, PublicRacer, RandomPolicy,
)
from .events import PendingEffect, RaceEvent
from .racer import Ability, Racer, ROLE_DATA_VERSION
from .race_actions import RaceActions, bounded_resolution
from .scoring import ScoreLedger
from .state import RacerStatus
from .track import Track
from . import scoocher_baby_duel22_cycle
from . import duel_romantic_scoocher22_cycle
from . import copy_baby_duel_scoocher_cycle

RULE_VERSION = "2026-10-29-rule-audit-reconciled-U01-H11-decided-H12-rolloff-reroll-H13-last-place-rolloff"
SUPPORTED_ABILITIES = frozenset({
    Ability.BLIMP, Ability.COACH, Ability.GUNK, Ability.SCOOCHER, Ability.STICKLER,
    Ability.ALCHEMIST, Ability.ROCKET_SCIENTIST, Ability.MAGICIAN,
    Ability.DICEMONGER, Ability.SKIPPER, Ability.INCHWORM,
    Ability.BANANA, Ability.BABA_YAGA, Ability.LACKEY, Ability.LEGS,
    Ability.HARE, Ability.HECKLER, Ability.GENIUS,
    Ability.LOVEABLE_LOSER, Ability.SISYPHUS,
    Ability.PARTY_ANIMAL, Ability.ROMANTIC, Ability.HUGE_BABY, Ability.MOUTH,
    Ability.SUCKERFISH, Ability.LEAPTOAD, Ability.CENTAUR,
    Ability.CHEERLEADER, Ability.HYPNOTIST, Ability.THIRD_WHEEL,
    Ability.FLIP_FLOP, Ability.DUELIST,
    Ability.COPY_CAT, Ability.MASTERMIND, Ability.EGG, Ability.TWIN,
})


class Dice(Protocol):
    def roll(self) -> int: ...


class D6:
    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.random = random.Random(seed)

    def roll(self) -> int:
        return self.random.randint(1, 6)


@dataclass(frozen=True)
class ModifierRecord:
    source_id: int
    ability: str
    before: int
    after: int
    operation: str = "add"


@dataclass(frozen=True)
class TurnRecord:
    racer_id: int
    roll: int | None
    start_position: int
    end_position: int
    raw_move: int
    final_move: int
    main_move_end_position: int
    modifiers: tuple[ModifierRecord, ...] = ()
    rolls: tuple[int, ...] = ()
    skipped_reason: str | None = None


@dataclass
class RaceResult:
    finish_order: list[int]
    final_positions: dict[int, int]
    turns: list[TurnRecord]
    events: list[RaceEvent]
    termination_reason: str
    score_deltas: dict[int, int]
    score_totals: dict[int, int]


class DiagnosticLimitExceeded(RuntimeError):
    """An incomplete diagnostic run, never a legal race result."""

    def __init__(self, trace: dict):
        super().__init__(f"Diagnostic budget exceeded: {trace.get('diagnostic_failure', 'incomplete_race')}")
        self.trace = trace


class RaceEngine(RaceActions):
    def __init__(
        self,
        track: Track,
        racers: list[Racer],
        turn_order: list[int],
        dice: Dice,
        policies: Mapping[int, Policy] | None = None,
        score_ledger: ScoreLedger | None = None,
        race_index: int = 1,
        max_events: int = 100_000,
        remaining_deck: tuple[Ability, ...] | None = None,
        previous_winners: tuple[Ability, ...] = (),
    ):
        ids = [r.racer_id for r in racers]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("at least two racers with unique IDs are required")
        if len(turn_order) != len(ids) or set(turn_order) != set(ids):
            raise ValueError("turn_order must contain each racer exactly once")
        if policies is not None and set(policies) - set(ids):
            raise ValueError("policies contain an unknown player ID")
        for racer in racers:
            if racer.ability is not None and racer.ability not in SUPPORTED_ABILITIES:
                raise NotImplementedError(f"Ability is not implemented: {racer.ability}")
            if not 0 <= racer.position < track.finish_position:
                raise ValueError("initial racers must be on the track")
            if racer.status != RacerStatus.ACTIVE or racer.finish_place is not None:
                raise ValueError("initial racers must be active and unplaced")
            if racer.ability == Ability.BLIMP and track.second_corner_position is None:
                raise ValueError("Blimp requires a verified or explicit fixture second corner")
        self.track = track
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive diagnostic budget")
        self.max_events = max_events
        self._turn_in_progress = False
        self._preparing = False
        self.initial_remaining_abilities = tuple(Ability(a) for a in (remaining_deck or ()))
        self.remaining_abilities = list(self.initial_remaining_abilities)
        self.previous_winners = tuple(Ability(a) for a in previous_winners)
        physical = {r.ability for r in racers if r.ability is not None}
        if len(physical) != sum(r.ability is not None for r in racers):
            raise ValueError("the same physical character card cannot race twice")
        if len(set(self.remaining_abilities)) != len(self.remaining_abilities) or physical.intersection(self.remaining_abilities):
            raise ValueError("remaining deck must contain unique cards not already racing")
        if physical.intersection(self.previous_winners):
            raise ValueError("an earlier winning physical card cannot also be racing")
        if set(self.previous_winners).intersection(self.remaining_abilities):
            raise ValueError("a previous winner card cannot still be in the remaining deck")
        if len(set(self.previous_winners)) != len(self.previous_winners):
            raise ValueError("an earlier winning physical card cannot be used twice")
        self.egg_drawn: list[Ability] = []
        self._copy_locks: set[int] = set()
        self._refreshing_copies = False
        if race_index not in (1, 2, 3, 4):
            raise ValueError("race_index must be from 1 to 4")
        self.race_index = race_index
        self.score_ledger = score_ledger if score_ledger is not None else ScoreLedger(ids)
        if set(self.score_ledger.totals()) != set(ids):
            raise ValueError("score ledger must contain exactly the participating players")
        self.initial_score_ledger = self.score_ledger.trace()
        self._score_start = len(self.score_ledger.entries)
        self.racers = {r.racer_id: r for r in racers}
        self.turn_order = turn_order.copy()
        self.dice = dice
        self.seed = dice.seed if isinstance(dice, D6) else None
        self.policies = {rid: (policies or {}).get(rid, KeepPolicy()) for rid in ids}
        self.policy_metadata = {
            str(rid): {
                "policy_id": getattr(policy, "policy_id", type(policy).__qualname__),
                "seed": policy.seed if isinstance(policy, RandomPolicy) else None,
            }
            for rid, policy in self.policies.items()
        }
        self.decisions: list[DecisionRecord] = []
        self.finish_order: list[int] = []
        self.turns: list[TurnRecord] = []
        self.events: list[RaceEvent] = []
        self.pending_effects: deque[PendingEffect] = deque()
        self.turn_index = 0
        self.next_turn_racer_id: int | None = None
        self.termination_reason: str | None = None
        self.initial_racers = [
            {**asdict(r), "status": r.status.value} for r in racers
        ]
        self.initial_code_fingerprint = self._code_fingerprint()
        self._genius_prediction: int | None = None
        self._preparing = True
        self._refresh_copies()
        for racer in self._active():
            self._prepare_racer(racer)
        self._drain_effects()
        self._preparing = False

    @staticmethod
    def _code_fingerprint() -> str:
        digest = hashlib.sha256()
        for name in ("engine.py", "events.py", "racer.py", "state.py", "track.py", "decisions.py", "scoring.py", "race_actions.py",
                     "copy_banana_cycle.py", "fish_baby_cycle.py", "star_baby_cycle.py", "romantic_queue_cycle.py",
                     "preparing_copy_romantic_cycle.py", "romantic_anchor_cycle.py", "romantic_baby_star_cycle.py",
                     "static_inheritance.py", "fish_baby_star_cycle.py", "romantic_baby_arrow_cycle.py",
                     "romantic_duel_copy_cycle.py", "egg_copy_scoocher_cycle.py", "romantic_scoocher_anchor_cycle.py",
                     "romantic_baby_baba_cycle.py", "copy_fish_baby_cycle.py", "centaur_copy_baby_star_cycle.py",
                     "fish_baby_arrow11_cycle.py", "romantic_baby_shared12_cycle.py", "scoocher_baby_duel22_cycle.py", "fish_baby_baba4_cycle.py", "fish_baby_arrow11_scoocher_cycle.py", "duel_romantic_scoocher22_cycle.py", "copy_baby_duel_scoocher_cycle.py"):
            digest.update(name.encode())
            digest.update(Path(__file__).with_name(name).read_bytes())
        return digest.hexdigest()

    def _emit(self, kind: str, racer_id: int, **kwargs) -> RaceEvent:
        self._check_event_budget()
        event = RaceEvent(len(self.events) + 1, kind, racer_id, **kwargs)
        self.events.append(event)
        return event

    def _check_event_budget(self) -> None:
        if (self._turn_in_progress or self._preparing) and len(self.events) >= self.max_events:
            trace = self.trace()
            trace["diagnostic_failure"] = "event_budget_exceeded"
            raise DiagnosticLimitExceeded(trace)

    def _clockwise_order(self) -> list[int]:
        return self.turn_order[self.turn_index:] + self.turn_order[:self.turn_index]

    def _choose(
        self, racer: Racer, window: DecisionWindow, legal_actions: tuple[Action, ...],
        *, roll: int | None, move_value: int | None, parent_sequence: int | None,
        magician_remaining: int = 0, dicemonger_remaining: int = 0,
    ) -> Action:
        if racer.status != RacerStatus.ACTIVE or self.termination_reason is not None:
            raise RuntimeError("no decision is legal for an inactive racer or ended race")
        self._check_event_budget()
        observation = Observation(
            current_player_id=self.turn_order[self.turn_index],
            turn_number=0 if window == DecisionWindow.BEFORE_RACE else len(self.turns) + 1,
            track_name=self.track.name,
            finish_position=self.track.finish_position,
            second_corner_position=self.track.second_corner_position,
            trip_positions=tuple(sorted(self.track.trip_positions)),
            racers=tuple(PublicRacer(
                r.racer_id, r.name, r.position, r.status.value,
                self._ability(r).value if self._ability(r) is not None else None, r.tripped, r.finish_place,
                r.copied_target,
            ) for r in (self.racers[rid] for rid in self.turn_order)),
            finish_order=tuple(self.finish_order), latest_roll=roll,
            current_move_value=move_value,
            magician_rerolls_remaining=magician_remaining,
            dicemonger_rerolls_remaining=dicemonger_remaining,
            score_totals=tuple(self.score_ledger.totals().items()),
        )
        request = DecisionRequest(len(self.decisions) + 1, racer.racer_id, window, observation, legal_actions)
        choice = self.policies[racer.racer_id].choose(request)
        if not isinstance(choice, Action) or choice not in legal_actions:
            raise ValueError("policy returned an illegal action")
        event = self._emit(
            "decision", racer.racer_id, source_id=choice.source_id,
            target_id=choice.target_id, parent_sequence=parent_sequence,
            action_kind=choice.kind.value, decision_id=request.decision_id,
        )
        self.decisions.append(DecisionRecord(request, choice, event.sequence))
        return choice

    def _score(
        self, racer: Racer, delta: int, ability: Ability | None, reason: str,
        *, parent_sequence: int | None = None, phase: str = "during_race",
        clamp_deduction: bool = False,
        notify_ability: bool = True,
    ) -> RaceEvent:
        self._check_event_budget()
        before = self.score_ledger.total(racer.racer_id)
        kwargs = dict(
            source=ability.value if ability is not None else "track",
            reason=reason, event_sequence=len(self.events) + 1, phase=phase,
        )
        if clamp_deduction:
            self.score_ledger.deduct_up_to_balance(racer.racer_id, self.race_index, -delta, **kwargs)
        else:
            self.score_ledger.record(racer.racer_id, self.race_index, delta, **kwargs)
        after = self.score_ledger.total(racer.racer_id)
        event = self._emit(
            "score", racer.racer_id, source_id=racer.racer_id if ability else None,
            ability=ability.value if ability else None, before=before, after=after,
            amount=after - before, parent_sequence=parent_sequence, reason=reason,
        )
        if notify_ability and ability is not None and before != after:
            self._queue_scoochers(event)
        return event

    def _active(self) -> list[Racer]:
        return [self.racers[rid] for rid in self._clockwise_order()
                if self.racers[rid].status == RacerStatus.ACTIVE]

    def _ability(self, racer: Racer) -> Ability | None:
        power = racer.inherited_ability or racer.ability
        return racer.copied_ability if power == Ability.COPY_CAT else power

    def _is_copy(self, racer: Racer) -> bool:
        return (racer.inherited_ability or racer.ability) == Ability.COPY_CAT

    def _leaders(self) -> tuple[int, ...]:
        active = self._active()
        if not active:
            return ()
        position = max(r.position for r in active)
        return tuple(r.racer_id for r in active if r.position == position)

    def _refresh_copies(self, *, allow_choice: bool = False) -> None:
        if self._refreshing_copies or self.termination_reason is not None:
            return
        self._refreshing_copies = True
        try:
            leaders = self._leaders()
            for racer in self._active():
                if not self._is_copy(racer) or racer.racer_id in self._copy_locks:
                    continue
                targets = [rid for rid in leaders if rid != racer.racer_id]
                old = (racer.copied_target, racer.copied_ability)
                if racer.copied_target in targets:
                    if allow_choice and len(targets) > 1:
                        choice = self._choose(
                            racer, DecisionWindow.COPY_CHOICE,
                            (Action(ActionKind.KEEP), *(Action(ActionKind.COPY_TARGET, racer.racer_id, rid) for rid in targets)),
                            roll=None, move_value=None, parent_sequence=None,
                        )
                        if choice.kind != ActionKind.KEEP:
                            racer.copied_target = choice.target_id
                elif len(targets) > 1:
                    choice = self._choose(
                        racer, DecisionWindow.COPY_CHOICE,
                        tuple(Action(ActionKind.COPY_TARGET, racer.racer_id, rid) for rid in targets),
                        roll=None, move_value=None, parent_sequence=None,
                    )
                    racer.copied_target = choice.target_id
                else:
                    racer.copied_target = targets[0] if targets else None
                if racer.copied_target is None:
                    racer.copied_ability = None
                else:
                    # H02: inspect the leader's original text, never its borrowed
                    # before-race power. Copying another Copy Cat adds no power.
                    raw = self.racers[racer.copied_target].ability
                    racer.copied_ability = None if raw in (Ability.EGG, Ability.TWIN, Ability.COPY_CAT) else raw
                if old != (racer.copied_target, racer.copied_ability):
                    self._emit("copy_changed", racer.racer_id, source_id=racer.racer_id,
                               target_id=racer.copied_target,
                               ability=racer.copied_ability.value if racer.copied_ability else None)
        finally:
            self._refreshing_copies = False

    def _prepare_racer(self, racer: Racer) -> None:
        if racer.pre_race_resolved:
            return
        power = racer.ability
        visited = set()
        while power in (Ability.EGG, Ability.TWIN):
            if power in visited:
                raise ValueError("cyclic before-race inheritance is not a legal physical-card context")
            visited.add(power)
            if power == Ability.EGG:
                if len(self.remaining_abilities) < 3:
                    raise ValueError("Egg requires three actual cards from the remaining deck")
                cards = tuple(self.remaining_abilities[:3])
                del self.remaining_abilities[:3]
                self.egg_drawn.extend(cards)
                legal = tuple(Action(ActionKind.EGG_CHOOSE, racer.racer_id, racer.racer_id, value=a.value) for a in cards)
            else:
                legal = (Action(ActionKind.KEEP), *(Action(ActionKind.TWIN_CHOOSE, racer.racer_id, racer.racer_id, value=a.value)
                                                   for a in dict.fromkeys(self.previous_winners)))
                if len(legal) == 1:
                    break
            choice = self._choose(racer, DecisionWindow.BEFORE_RACE, legal,
                                  roll=None, move_value=None, parent_sequence=None)
            if choice.kind == ActionKind.KEEP:
                break
            power = Ability(choice.value)
            racer.inherited_ability = power
            effect = self._emit("inherit", racer.racer_id, source_id=racer.racer_id,
                                ability=power.value, parent_sequence=self.decisions[-1].event_sequence)
            self._queue_scoochers(effect)
        racer.pre_race_resolved = True
        self._refresh_copies()
        if self._ability(racer) == Ability.SISYPHUS:
            self._score(racer, 4, Ability.SISYPHUS, "before_race", phase="before_race")
        self._drain_effects()

    def _alone_at_extreme(self, racer: Racer, *, leading: bool) -> bool:
        active = self._active()
        extreme = (max if leading else min)(r.position for r in active)
        return racer.position == extreme and sum(r.position == extreme for r in active) == 1

    def _start_turn(self, racer: Racer) -> bool:
        self._refresh_copies(allow_choice=True)
        copy = self._is_copy(racer)
        if copy:
            self._copy_locks.add(racer.racer_id)
        try:
            if self._ability(racer) == Ability.MASTERMIND and racer.turns_taken == 1:
                choice = self._choose(
                    racer, DecisionWindow.TURN_START,
                    tuple(Action(ActionKind.MASTERMIND_PREDICT, racer.racer_id, r.racer_id) for r in self._active()),
                    roll=None, move_value=None, parent_sequence=None,
                )
                racer.prediction = choice.target_id
                self._emit("prediction", racer.racer_id, source_id=racer.racer_id,
                           target_id=choice.target_id, ability=Ability.MASTERMIND.value,
                           parent_sequence=self.decisions[-1].event_sequence)
            result = self._start_turn_impl(racer)
        finally:
            self._copy_locks.discard(racer.racer_id)
        self._refresh_copies(allow_choice=True)
        return result

    def _start_turn_impl(self, racer: Racer) -> bool:
        """Return a locked Hare skip flag from the actual turn-start snapshot."""
        hare_skip = self._ability(racer) == Ability.HARE and self._alone_at_extreme(racer, leading=True)
        if self._ability(racer) == Ability.LOVEABLE_LOSER and self._alone_at_extreme(racer, leading=False):
            self._score(racer, 1, Ability.LOVEABLE_LOSER, "alone_last_at_turn_start")
            self._drain_effects()
        elif self._ability(racer) == Ability.PARTY_ANIMAL:
            self._party_pull(racer)
            self._drain_effects()
        elif self._ability(racer) == Ability.CHEERLEADER:
            choice = self._choose(
                racer, DecisionWindow.TURN_START,
                (Action(ActionKind.KEEP), Action(ActionKind.CHEER, racer.racer_id, racer.racer_id)),
                roll=None, move_value=None, parent_sequence=None,
            )
            if choice.kind == ActionKind.CHEER:
                effect = self._emit("ability_effect", racer.racer_id, source_id=racer.racer_id,
                                    ability=Ability.CHEERLEADER.value,
                                    parent_sequence=self.decisions[-1].event_sequence)
                first_event = len(self.events)
                last = min(r.position for r in self._active())
                positions = {r.racer_id: r.position + 2 for r in self._active() if r.position == last}
                self._move_batch(positions, source_id=racer.racer_id,
                                 ability=Ability.CHEERLEADER.value, parent_sequence=effect.sequence)
                self._drain_effects()
                if self.termination_reason is None and racer.status == RacerStatus.ACTIVE:
                    self._move(racer, 1, movement_kind="ability", source_id=racer.racer_id,
                               ability=Ability.CHEERLEADER.value, parent_sequence=effect.sequence)
                if self.termination_reason is None and any(
                    e.kind == "move" and e.source_id == racer.racer_id and e.ability == Ability.CHEERLEADER.value
                    for e in self.events[first_event:]
                ):
                    self._queue_scoochers(effect)
                self._drain_effects()
        elif self._ability(racer) in (Ability.HYPNOTIST, Ability.THIRD_WHEEL):
            legal = [Action(ActionKind.KEEP)]
            if self._ability(racer) == Ability.HYPNOTIST:
                legal.extend(Action(ActionKind.HYPNOTIZE, racer.racer_id, r.racer_id)
                             for r in self._active() if r.racer_id != racer.racer_id)
            else:
                positions = [p for p in sorted({r.position for r in self._active()})
                             if sum(r.position == p for r in self._active()) == 2]
                legal.extend(Action(ActionKind.THIRD_WHEEL_WARP, racer.racer_id,
                                    racer.racer_id, value=p) for p in positions)
            if len(legal) > 1:
                choice = self._choose(racer, DecisionWindow.TURN_START, tuple(legal),
                                      roll=None, move_value=None, parent_sequence=None)
                if choice.kind != ActionKind.KEEP:
                    target = self.racers[choice.target_id]
                    position = racer.position if choice.kind == ActionKind.HYPNOTIZE else int(choice.value)
                    if target.position != position:
                        effect = self._emit("ability_effect", racer.racer_id, source_id=racer.racer_id,
                                            target_id=target.racer_id, ability=self._ability(racer).value,
                                            parent_sequence=self.decisions[-1].event_sequence)
                        self._queue_scoochers(effect)
                        self._warp(target, position, source_id=racer.racer_id,
                                   ability=self._ability(racer).value, parent_sequence=effect.sequence)
                        self._drain_effects()
        if self.termination_reason is None:
            for source in self._active():
                if self._ability(source) == Ability.DUELIST:
                    self._offer_duel(source, None)
                    if self.termination_reason is not None:
                        break
        return hare_skip

    @bounded_resolution
    def _offer_duel(self, source: Racer, trigger_sequence: int | None) -> None:
        self._run_resolution_frames(self._offer_duel_frame(source, trigger_sequence))

    @copy_baby_duel_scoocher_cycle.duel_scope
    @duel_romantic_scoocher22_cycle.duel_scope
    @scoocher_baby_duel22_cycle.duel_scope
    def _offer_duel_frame(self, source: Racer, trigger_sequence: int | None):
        actual_kinds = {"move", "warp", "score", "recover", "roll", "reroll", "duel_roll",
                        "main_move_skipped", "ability_effect", "trip", "finish", "eliminate"}
        def epoch():
            return (len(self.turns), next((e.sequence for e in reversed(self.events)
                                         if e.kind in actual_kinds), 0))
        offered = getattr(self, "_duel_offered_epochs", {})
        self._duel_offered_epochs = offered
        if offered.get(source.racer_id) == epoch():
            return
        while self.termination_reason is None and source.status == RacerStatus.ACTIVE:
            # A nested arrival can offer this opportunity and receive KEEP.
            # On returning, the older loop must not ask the same unchanged
            # window again. A new actual event advances epoch and permits it.
            if offered.get(source.racer_id) == epoch():
                return
            targets = [r for r in self._active()
                       if r.racer_id != source.racer_id and r.position == source.position]
            if not targets:
                return
            proof_event_start = len(self.events)
            choice = self._choose(
                source, DecisionWindow.DUEL,
                (Action(ActionKind.KEEP), *(Action(ActionKind.DUEL, source.racer_id, r.racer_id) for r in targets)),
                roll=None, move_value=None, parent_sequence=trigger_sequence,
            )
            if choice.kind == ActionKind.KEEP:
                offered[source.racer_id] = epoch()
                self._inert_duel_choice_prefix(source, trigger_sequence, proof_event_start, choice)
                return
            target = self.racers[choice.target_id]
            effect = self._emit("ability_effect", source.racer_id, source_id=source.racer_id,
                                target_id=target.racer_id, ability=Ability.DUELIST.value,
                                parent_sequence=self.decisions[-1].event_sequence)
            self._queue_scoochers(effect)
            rolls = []
            for participant in (source, target):
                value = self._roll_die()
                rolls.append(value)
                self._emit("duel_roll", participant.racer_id, source_id=source.racer_id,
                           ability=Ability.DUELIST.value, amount=value, parent_sequence=effect.sequence)
            winner = source if rolls[0] >= rolls[1] else target
            self._move(winner, 2, movement_kind="ability", source_id=source.racer_id,
                       ability=Ability.DUELIST.value, parent_sequence=effect.sequence)
            self._inert_duel_choice_prefix(source, trigger_sequence, proof_event_start, choice, rolls, winner.racer_id)
            yield self._drain_effects_frame()

    def _before_main_move(self, racer: Racer) -> bool | str:
        """Optional no-roll move or prediction. True means Legs chose five."""
        self._genius_prediction = None
        legal = [Action(ActionKind.KEEP)]
        if self._ability(racer) == Ability.LEGS:
            legal.append(Action(ActionKind.LEGS_FIVE, racer.racer_id, racer.racer_id))
        elif self._ability(racer) == Ability.GENIUS:
            legal.extend(Action(ActionKind.GENIUS_PREDICT, racer.racer_id, racer.racer_id, value=n)
                         for n in range(1, 7))
        elif self._ability(racer) == Ability.FLIP_FLOP:
            legal.extend(Action(ActionKind.FLIP_FLOP_SWAP, racer.racer_id, r.racer_id)
                         for r in self._active() if r.racer_id != racer.racer_id)
        if len(legal) == 1:
            return False
        choice = self._choose(
            racer, DecisionWindow.BEFORE_MAIN_MOVE, tuple(legal),
            roll=None, move_value=None, parent_sequence=None,
        )
        if choice.kind == ActionKind.GENIUS_PREDICT:
            self._genius_prediction = int(choice.value)
        elif choice.kind == ActionKind.FLIP_FLOP_SWAP:
            other = self.racers[choice.target_id]
            if racer.position != other.position:
                effect = self._emit("ability_effect", racer.racer_id, source_id=racer.racer_id,
                                    target_id=other.racer_id, ability=Ability.FLIP_FLOP.value,
                                    parent_sequence=self.decisions[-1].event_sequence)
                self._queue_scoochers(effect)
                self._move_batch({racer.racer_id: other.position, other.racer_id: racer.position},
                                 source_id=racer.racer_id, ability=Ability.FLIP_FLOP.value,
                                 parent_sequence=effect.sequence, mode="warp")
                self._drain_effects()
            return "flip_flop"
        return choice.kind == ActionKind.LEGS_FIVE

    def _roll_die(self) -> int:
        self._check_event_budget()
        roll = self.dice.roll()
        if type(roll) is not int or not 1 <= roll <= 6:
            raise ValueError("dice must return an integer from 1 to 6")
        return roll

    def _roll_main_move(self, racer: Racer) -> tuple[int | None, tuple[int, ...], RaceEvent]:
        roll = self._roll_die()
        rolls = [roll]
        last = self._emit("roll", racer.racer_id, amount=roll)
        magician_remaining = 2 if self._ability(racer) == Ability.MAGICIAN else 0
        magician_used = 0
        dicemonger_used = False
        while racer.status == RacerStatus.ACTIVE and self.termination_reason is None:
            if self._ability(racer) == Ability.MAGICIAN:
                magician_remaining = max(magician_remaining, 2 - magician_used)
            grantors = [
                self.racers[rid] for rid in self._clockwise_order()
                if self.racers[rid].ability == Ability.DICEMONGER
                and self.racers[rid].status == RacerStatus.ACTIVE
            ]
            legal = [Action(ActionKind.KEEP)]
            if magician_remaining:
                legal.append(Action(ActionKind.MAGICIAN_REROLL, racer.racer_id, racer.racer_id))
            if grantors and not dicemonger_used:
                legal.extend(Action(ActionKind.DICEMONGER_REROLL, r.racer_id, racer.racer_id) for r in grantors)
            if len(legal) == 1:
                break
            choice = self._choose(
                racer, DecisionWindow.REROLL, tuple(legal), roll=roll,
                move_value=None, parent_sequence=last.sequence,
                magician_remaining=magician_remaining,
                dicemonger_remaining=int(bool(grantors) and not dicemonger_used),
            )
            if choice.kind == ActionKind.KEEP:
                break
            if choice.kind == ActionKind.MAGICIAN_REROLL:
                magician_remaining -= 1
                magician_used += 1
                source = racer
            else:
                dicemonger_used = True
                source = self.racers[choice.source_id]
            previous, roll = roll, self._roll_die()
            rolls.append(roll)
            last = self._emit(
                "reroll", racer.racer_id, source_id=source.racer_id,
                ability=(Ability.MAGICIAN.value if choice.kind == ActionKind.MAGICIAN_REROLL else Ability.DICEMONGER.value),
                before=previous, after=roll, amount=roll,
                parent_sequence=self.decisions[-1].event_sequence,
            )
            # One source instance per actual reroll, including Dicemonger's bonus.
            self._queue_scoochers(last)
            if choice.kind == ActionKind.DICEMONGER_REROLL and source.racer_id != racer.racer_id:
                self._move(
                    source, 1, movement_kind="ability", source_id=source.racer_id,
                    ability=Ability.DICEMONGER.value, parent_sequence=last.sequence,
                )
            self._drain_effects()
        if self.termination_reason is not None or racer.status != RacerStatus.ACTIVE:
            return None, tuple(rolls), last
        locked = self._emit("roll_locked", racer.racer_id, amount=roll, parent_sequence=last.sequence)
        return roll, tuple(rolls), locked

    def _roll_responses(self, racer: Racer, roll: int, parent_sequence: int) -> bool:
        skip_main = False
        sources = [self.racers[rid] for rid in self._clockwise_order()]
        for source in sources:
            if self.termination_reason is not None:
                break
            if source.status != RacerStatus.ACTIVE:
                continue
            if source.racer_id == racer.racer_id and self._genius_prediction == roll:
                self.next_turn_racer_id = source.racer_id
                effect = self._emit(
                    "next_turn_requested", source.racer_id, source_id=source.racer_id,
                    ability=Ability.GENIUS.value, parent_sequence=parent_sequence,
                )
                self._queue_scoochers(effect)
                self._drain_effects()
            if roll == 1 and self._ability(source) == Ability.INCHWORM and source.racer_id != racer.racer_id:
                skip_main = True
                effect = self._emit(
                    "ability_effect", source.racer_id, source_id=source.racer_id,
                    ability=self._ability(source).value, amount=1, parent_sequence=parent_sequence,
                    reason="cancel_main_move",
                )
                self._queue_scoochers(effect)
                self._move(
                    source, 1, movement_kind="ability", source_id=source.racer_id,
                    ability=self._ability(source).value, parent_sequence=effect.sequence,
                )
                self._drain_effects()
            elif roll == 1 and self._ability(source) == Ability.SKIPPER:
                self.next_turn_racer_id = source.racer_id
                effect = self._emit(
                    "next_turn_requested", source.racer_id, source_id=source.racer_id,
                    ability=self._ability(source).value, parent_sequence=parent_sequence,
                )
                self._queue_scoochers(effect)
                self._drain_effects()
            elif roll == 6 and self._ability(source) == Ability.LACKEY and source.racer_id != racer.racer_id:
                self._move(
                    source, 2, movement_kind="ability", source_id=source.racer_id,
                    ability=self._ability(source).value, parent_sequence=parent_sequence,
                    notify_ability=True,
                )
                self._drain_effects()
        return skip_main

    def _queue_scoochers(self, trigger: RaceEvent) -> None:
        """Register each effective power separately; never infer from net movement."""
        for racer_id in self._clockwise_order():
            watcher = self.racers[racer_id]
            if (
                watcher.status == RacerStatus.ACTIVE
                and self._ability(watcher) == Ability.SCOOCHER
                and watcher.racer_id != trigger.source_id
            ):
                queued = self._emit(
                    "effect_queued", racer_id, source_id=trigger.source_id,
                    ability=Ability.SCOOCHER.value, parent_sequence=trigger.sequence,
                )
                self.pending_effects.append(PendingEffect(
                    racer_id, trigger.source_id, trigger.sequence, queued.sequence,
                ))

    def _cancel_effect(self, effect: PendingEffect, reason: str) -> None:
        self._emit(
            "effect_cancelled", effect.racer_id, source_id=effect.source_id,
            ability=effect.ability, parent_sequence=effect.queued_sequence,
            reason=reason,
        )


    def _check_end(self, terminal_reason: str | None = None) -> None:
        if self.termination_reason is not None:
            return
        if len(self.finish_order) >= 2:
            self.termination_reason = terminal_reason or "second_place"
        else:
            active = [r for r in self.racers.values() if r.status == RacerStatus.ACTIVE]
            if len(active) <= 1:
                if active:
                    self._award_place(active[0], terminal_reason="one_racer_remaining")
                    if self.termination_reason is not None:
                        return
                self.termination_reason = "one_racer_remaining" if active else "no_racers_remaining"
        if self.termination_reason is not None:
            while self.pending_effects:
                self._cancel_effect(self.pending_effects.popleft(), "race_ended")
            self._emit("race_end", self.turn_order[self.turn_index])

    def _finish_context(self) -> dict:
        return {
            "leaders": self._leaders(),
            "powers": {r.racer_id: self._ability(r) for r in self._active()},
            "copy_targets": {r.racer_id: r.copied_target for r in self._active() if self._is_copy(r)},
        }

    def _award_place(self, racer: Racer, parent_sequence: int | None = None,
                     finish_context: dict | None = None, terminal_reason: str | None = None) -> None:
        if self.termination_reason is not None or len(self.finish_order) >= 2:
            return
        context = finish_context if finish_context is not None else self._finish_context()
        racer.status = RacerStatus.FINISHED
        self.finish_order.append(racer.racer_id)
        place = len(self.finish_order)
        racer.finish_places += (place,)
        if racer.finish_place is None:
            racer.finish_place = place
        event = self._emit("finish", racer.racer_id, amount=place, parent_sequence=parent_sequence)
        if place == 1:
            for rid in self._clockwise_order():
                predictor = self.racers[rid]
                if context["powers"].get(rid) != Ability.MASTERMIND or predictor.prediction != racer.racer_id:
                    continue
                if predictor.status != RacerStatus.ACTIVE and rid != racer.racer_id:
                    continue
                if self._is_copy(predictor):
                    target_id = context["copy_targets"].get(rid)
                    if (target_id not in context["leaders"] or target_id is None
                            or self.racers[target_id].ability != Ability.MASTERMIND):
                        continue
                effect = self._emit("ability_effect", rid, source_id=rid, target_id=racer.racer_id,
                                    ability=Ability.MASTERMIND.value, parent_sequence=event.sequence)
                self._queue_scoochers(effect)
                self._award_place(predictor, effect.sequence, finish_context=context, terminal_reason=terminal_reason)
                return
        if place == 2:
            self._check_end(terminal_reason)

    def _main_move(
        self, racer: Racer, roll: int, parent_sequence: int,
    ) -> tuple[int, int, tuple[ModifierRecord, ...], RaceEvent | None]:
        raw_move = roll
        modifiers = []
        rocket_cost = None
        # Current player's power, then the other seats clockwise (O02/O07).
        for source_id in self._clockwise_order():
            source = self.racers[source_id]
            if source.status != RacerStatus.ACTIVE:
                continue
            before = raw_move
            operation = "add"
            modifier_parent = parent_sequence
            if self._ability(source) == Ability.BLIMP and source_id == racer.racer_id:
                raw_move += 3 if racer.position < self.track.second_corner_position else -1
            elif self._ability(source) == Ability.COACH and source.position == racer.position:
                raw_move += 1
            elif self._ability(source) == Ability.GUNK and source_id != racer.racer_id:
                raw_move -= 1
            elif self._ability(source) == Ability.HARE and source_id == racer.racer_id:
                raw_move += 2
            elif self._ability(source) == Ability.PARTY_ANIMAL and source_id == racer.racer_id:
                raw_move += sum(r.racer_id != racer.racer_id and r.position == racer.position
                                for r in self._active())
            elif source_id == racer.racer_id and (
                self._ability(source) == Ability.ROCKET_SCIENTIST
                or (self._ability(source) == Ability.ALCHEMIST and roll in (1, 2))
            ):
                kind = (ActionKind.ALCHEMIST_FOUR if self._ability(source) == Ability.ALCHEMIST
                        else ActionKind.ROCKET_DOUBLE)
                choice = self._choose(
                    racer, DecisionWindow.MAIN_MOVE_MODIFIER,
                    (Action(ActionKind.KEEP), Action(kind, source_id, racer.racer_id)),
                    roll=roll, move_value=raw_move, parent_sequence=parent_sequence,
                )
                if choice.kind != ActionKind.KEEP:
                    modifier_parent = self.decisions[-1].event_sequence
                    if kind == ActionKind.ALCHEMIST_FOUR:
                        raw_move, operation = 4, "replace"
                    else:
                        raw_move, operation = raw_move * 2, "multiply"
            if raw_move != before:
                modifiers.append(ModifierRecord(source_id, self._ability(source).value, before, raw_move, operation))
                effect = self._emit(
                    "modifier", racer.racer_id, source_id=source_id,
                    ability=self._ability(source).value, before=before, after=raw_move,
                    amount=raw_move - before, operation=operation,
                    parent_sequence=modifier_parent,
                )
                self._queue_scoochers(effect)
                if self._ability(source) == Ability.ROCKET_SCIENTIST:
                    rocket_cost = effect
        # HOUSE H11: clamp exactly once, after every applicable modification.
        final_move = max(0, raw_move)
        self._emit(
            "main_move_resolved", racer.racer_id, before=raw_move, after=final_move,
            parent_sequence=parent_sequence,
        )
        return raw_move, final_move, tuple(modifiers), rocket_cost

    def _trip(
        self, racer: Racer, *, parent_sequence: int,
        source_id: int | None = None, ability: str | None = None,
        notify_ability: bool = False,
    ) -> bool:
        if self.termination_reason is None and racer.status == RacerStatus.ACTIVE and not racer.tripped:
            racer.tripped = True
            event = self._emit("trip", racer.racer_id, source_id=source_id, ability=ability, parent_sequence=parent_sequence)
            if notify_ability:
                self._queue_scoochers(event)
            return True
        return False



    def _sisyphus_six(self, racer: Racer, parent_sequence: int) -> None:
        effect = self._emit("ability_effect", racer.racer_id, source_id=racer.racer_id,
                            ability=Ability.SISYPHUS.value, parent_sequence=parent_sequence,
                            reason="replace_main_with_warp")
        self._queue_scoochers(effect)
        self._warp(racer, 0, source_id=racer.racer_id, ability=Ability.SISYPHUS.value,
                   parent_sequence=effect.sequence)
        if self.termination_reason is None:
            self._score(racer, -1, Ability.SISYPHUS, "rolled_six", parent_sequence=effect.sequence,
                        clamp_deduction=True, notify_ability=False)

    def _end_turn(self, racer: Racer, start_position: int) -> None:
        if self.termination_reason is not None or abs(racer.position - start_position) > 1:
            return
        # One turn-end snapshot; never recursively reopen it after Heckler moves.
        hecklers = [r for r in self._active() if self._ability(r) == Ability.HECKLER]
        for source in hecklers:
            if self.termination_reason is not None:
                break
            self._move(source, 2, movement_kind="ability", source_id=source.racer_id,
                       ability=Ability.HECKLER.value, notify_ability=True)
            self._drain_effects()


    def take_turn(self, racer: Racer) -> None:
        if self._turn_in_progress or self._preparing:
            raise RuntimeError("incomplete diagnostic state cannot resume as a new turn")
        self._check_end()
        if self.termination_reason is not None:
            raise RuntimeError("race has already ended")
        if self.racers.get(racer.racer_id) is not racer:
            raise ValueError("racer is not part of this engine")
        if racer.racer_id != self.turn_order[self.turn_index] or racer.status != RacerStatus.ACTIVE:
            raise ValueError("only the current active racer may take a turn")
        start_position = racer.position
        self._turn_in_progress = True
        racer.turns_taken += 1
        self.next_turn_racer_id = None
        self._genius_prediction = None
        roll, rolls, raw_move, final_move, modifiers = None, (), 0, 0, ()
        skipped_reason = None
        main_attempted = False
        hare_skip = self._start_turn(racer)
        if self.termination_reason is not None or racer.status != RacerStatus.ACTIVE:
            skipped_reason = "race_ended" if self.termination_reason else "racer_inactive"
        elif racer.tripped:
            racer.tripped = False
            skipped_reason = "tripped"
            self._emit("recover", racer.racer_id)
        elif hare_skip:
            skipped_reason = "hare_alone_lead"
            effect = self._emit("main_move_skipped", racer.racer_id, source_id=racer.racer_id,
                                ability=Ability.HARE.value, reason=skipped_reason)
            self._queue_scoochers(effect)
        else:
            fixed_five = self._before_main_move(racer)
            if fixed_five == "flip_flop":
                skipped_reason = "flip_flop"
                move_value = None
            elif fixed_five:
                locked = self._emit("main_move_fixed", racer.racer_id, source_id=racer.racer_id,
                                    ability=Ability.LEGS.value, amount=5,
                                    parent_sequence=self.decisions[-1].event_sequence)
                self._queue_scoochers(locked)
                move_value = 5
            else:
                roll, rolls, locked = self._roll_main_move(racer)
                move_value = roll
            if move_value is not None:
                skip_main = self._roll_responses(racer, roll, locked.sequence) if roll is not None else False
                if self.termination_reason is None and racer.status == RacerStatus.ACTIVE:
                    if skip_main:
                        skipped_reason = "inchworm"
                        self._emit("main_move_skipped", racer.racer_id, parent_sequence=locked.sequence, reason=skipped_reason)
                    elif self._ability(racer) == Ability.SISYPHUS and roll == 6:
                        skipped_reason = "sisyphus_six"
                        self._sisyphus_six(racer, locked.sequence)
                    else:
                        main_attempted = True
                        if self._is_copy(racer):
                            self._copy_locks.add(racer.racer_id)
                        try:
                            raw_move, final_move, modifiers, rocket_cost = self._main_move(racer, move_value, locked.sequence)
                            self._move(racer, final_move, parent_sequence=self.events[-1].sequence)
                            if rocket_cost is not None:
                                self._trip(
                                    racer, source_id=racer.racer_id,
                                    ability=Ability.ROCKET_SCIENTIST.value,
                                    parent_sequence=rocket_cost.sequence,
                                )
                        finally:
                            self._copy_locks.discard(racer.racer_id)
                        self._refresh_copies(allow_choice=True)
            if self.termination_reason is not None and not main_attempted:
                skipped_reason = skipped_reason or "race_ended"
            elif racer.status != RacerStatus.ACTIVE and not main_attempted:
                skipped_reason = "racer_inactive"
        main_move_end_position = racer.position
        self._drain_effects()
        self._end_turn(racer, start_position)
        self.turns.append(TurnRecord(
            racer.racer_id, roll, start_position, racer.position,
            raw_move, final_move, main_move_end_position, modifiers,
            rolls, skipped_reason,
        ))
        if self.termination_reason is None:
            if (self.next_turn_racer_id is not None
                    and self.racers[self.next_turn_racer_id].status == RacerStatus.ACTIVE):
                self.turn_index = self.turn_order.index(self.next_turn_racer_id)
            else:
                while True:
                    self.turn_index = (self.turn_index + 1) % len(self.turn_order)
                    if self.racers[self.turn_order[self.turn_index]].status == RacerStatus.ACTIVE:
                        break
        self.next_turn_racer_id = None
        self._turn_in_progress = False

    def trace(self) -> dict:
        """Serializable initial state and actual dice/events, with no RNG internals."""
        track = asdict(self.track)
        track["trip_positions"] = sorted(self.track.trip_positions)
        track["star_positions"] = sorted(self.track.star_positions)
        return {
            "schema_version": 5,
            "rule_version": RULE_VERSION,
            "role_data_version": ROLE_DATA_VERSION,
            "code_fingerprint": self.initial_code_fingerprint,
            "run_kind": "incremental_race",
            "seed": self.seed,
            "race_index": self.race_index,
            "initial_remaining_abilities": list(self.initial_remaining_abilities),
            "remaining_abilities": self.remaining_abilities.copy(),
            "previous_winners": list(self.previous_winners),
            "egg_drawn": self.egg_drawn.copy(),
            "max_events": self.max_events,
            "turn_in_progress": self._turn_in_progress,
            "preparing": self._preparing,
            "initial_score_ledger": self.initial_score_ledger,
            "score_ledger": self.score_ledger.trace(),
            "track": track,
            "initial_racers": self.initial_racers,
            "turn_order": self.turn_order.copy(),
            "policies": self.policy_metadata,
            "decisions": [asdict(d) for d in self.decisions],
            "turns": [asdict(t) for t in self.turns],
            "events": [asdict(e) for e in self.events],
            "pending_effects": [asdict(e) for e in self.pending_effects],
            "racers": [{**asdict(r), "status": r.status.value} for r in self.racers.values()],
            "finish_order": self.finish_order.copy(),
            "termination_reason": self.termination_reason,
            "next_player_id": self.turn_order[self.turn_index] if self.termination_reason is None else None,
            "completed": (self.termination_reason is not None
                          and not self._turn_in_progress and not self._preparing),
        }

    def run(self, max_turns: int = 10_000) -> RaceResult:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self._check_end()
        while self.termination_reason is None:
            if len(self.turns) >= max_turns:
                trace = self.trace()
                trace["diagnostic_failure"] = "turn_budget_exceeded"
                raise DiagnosticLimitExceeded(trace)
            self.take_turn(self.racers[self.turn_order[self.turn_index]])
        return RaceResult(
            finish_order=self.finish_order.copy(),
            final_positions={rid: r.position for rid, r in self.racers.items()},
            turns=self.turns.copy(), events=self.events.copy(),
            termination_reason=self.termination_reason,
            score_deltas={rid: sum(e.delta for e in self.score_ledger.entries[self._score_start:]
                                    if e.player_id == rid) for rid in self.racers},
            score_totals=self.score_ledger.totals(),
        )


def replay_trace(trace: dict) -> RaceEngine:
    """Re-execute actual recorded rolls; reject incompatible or altered traces."""
    trace = json.loads(json.dumps(trace))
    if trace["schema_version"] != 5 or trace["rule_version"] != RULE_VERSION:
        raise ValueError("incompatible replay schema or rule version")
    if trace["code_fingerprint"] != RaceEngine._code_fingerprint():
        raise ValueError("replay requires the same engine code version")

    class RecordedDice:
        def __init__(self):
            self.rolls = iter(e["amount"] for e in trace["events"] if e["kind"] in {"roll", "reroll", "duel_roll"})

        def roll(self) -> int:
            return next(self.rolls)

    recorded_decisions = iter(trace["decisions"])

    class RecordedPolicy:
        def choose(self, request: DecisionRequest) -> Action:
            try:
                record = next(recorded_decisions)
            except StopIteration as exc:
                raise ValueError("replay has insufficient decision records") from exc
            if json.loads(json.dumps(asdict(request))) != record["request"]:
                raise ValueError("replay decision observation or legal actions diverged")
            return Action(**record["action"])

    track_data = {
        **trace["track"], "trip_positions": frozenset(trace["track"]["trip_positions"]),
        "star_positions": frozenset(trace["track"].get("star_positions", ())),
        "arrows": tuple(tuple(a) for a in trace["track"].get("arrows", ())),
    }
    racers = [Racer(**{**r, "status": RacerStatus(r["status"])}) for r in trace["initial_racers"]]
    dice = RecordedDice()
    engine = RaceEngine.__new__(RaceEngine)
    diagnostic = None
    try:
        engine.__init__(
            Track(**track_data), racers, trace["turn_order"], dice,
            policies={r.racer_id: RecordedPolicy() for r in racers},
            score_ledger=ScoreLedger.from_trace(trace["initial_score_ledger"]),
            race_index=trace["race_index"], max_events=trace["max_events"],
            remaining_deck=tuple(Ability(a) for a in trace["initial_remaining_abilities"]),
            previous_winners=tuple(Ability(a) for a in trace["previous_winners"]),
        )
        engine.seed = trace["seed"]
        engine.policy_metadata = trace["policies"]
        for _ in range(len(trace["turns"]) + int(trace["turn_in_progress"])):
            engine.take_turn(engine.racers[engine.turn_order[engine.turn_index]])
    except DiagnosticLimitExceeded as exc:
        diagnostic = exc.trace["diagnostic_failure"]
    except StopIteration as exc:
        raise ValueError("replay has insufficient dice records") from exc
    engine.seed = trace["seed"]
    engine.policy_metadata = trace["policies"]
    expected_diagnostic = trace.get("diagnostic_failure")
    if expected_diagnostic == "turn_budget_exceeded":
        expected_diagnostic = None
    if diagnostic != expected_diagnostic:
        raise ValueError("replay diagnostic outcome diverged")
    expected = json.loads(json.dumps({k: v for k, v in trace.items() if k != "diagnostic_failure"}))
    # Normalize tuples in turn records to the JSON representation.
    actual = json.loads(json.dumps(engine.trace()))
    if actual != expected:
        raise ValueError("replay diverged from the recorded state or events")
    return engine
