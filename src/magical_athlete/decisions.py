"""Immutable public decisions shared by the engine and legal-action policies.

These records contain only present, public information. They never carry live
engine/racer references, a dice source, or an engine random-number generator.
"""

import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ActionKind(StrEnum):
    KEEP = "keep"
    ALCHEMIST_FOUR = "alchemist_four"
    ROCKET_DOUBLE = "rocket_double"
    MAGICIAN_REROLL = "magician_reroll"
    DICEMONGER_REROLL = "dicemonger_reroll"
    LEGS_FIVE = "legs_five"
    GENIUS_PREDICT = "genius_predict"
    SUCKERFISH_FOLLOW = "suckerfish_follow"
    CHEER = "cheer"
    HYPNOTIZE = "hypnotize"
    THIRD_WHEEL_WARP = "third_wheel_warp"
    FLIP_FLOP_SWAP = "flip_flop_swap"
    DUEL = "duel"
    COPY_TARGET = "copy_target"
    MASTERMIND_PREDICT = "mastermind_predict"
    EGG_CHOOSE = "egg_choose"
    TWIN_CHOOSE = "twin_choose"


class DecisionWindow(StrEnum):
    REROLL = "reroll"
    MAIN_MOVE_MODIFIER = "main_move_modifier"
    BEFORE_MAIN_MOVE = "before_main_move"
    TURN_START = "turn_start"
    MOVE_START = "move_start"
    DUEL = "duel"
    COPY_CHOICE = "copy_choice"
    BEFORE_RACE = "before_race"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    source_id: int | None = None
    target_id: int | None = None
    value: int | str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ActionKind(self.kind))
        for name in ("source_id", "target_id"):
            value = getattr(self, name)
            if value is not None and type(value) is not int:
                raise ValueError(f"{name} must be an integer or None")
        if self.value is not None and type(self.value) is not int and not isinstance(self.value, str):
            raise ValueError("action value must be an integer, string or None")


@dataclass(frozen=True, slots=True)
class PublicRacer:
    racer_id: int
    name: str
    position: int
    status: str
    ability: str | None
    tripped: bool
    finish_place: int | None
    copied_target: int | None = None


@dataclass(frozen=True, slots=True)
class Observation:
    current_player_id: int
    turn_number: int
    track_name: str
    finish_position: int
    second_corner_position: int | None
    trip_positions: tuple[int, ...]
    racers: tuple[PublicRacer, ...]
    finish_order: tuple[int, ...]
    latest_roll: int | None
    current_move_value: int | None
    magician_rerolls_remaining: int = 0
    dicemonger_rerolls_remaining: int = 0
    score_totals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        # Freeze caller-owned collection containers too, rather than retaining
        # a mutable list behind a frozen dataclass annotation.
        object.__setattr__(self, "trip_positions", tuple(self.trip_positions))
        object.__setattr__(self, "racers", tuple(self.racers))
        object.__setattr__(self, "finish_order", tuple(self.finish_order))
        object.__setattr__(self, "score_totals", tuple(tuple(pair) for pair in self.score_totals))
        if not all(isinstance(racer, PublicRacer) for racer in self.racers):
            raise TypeError("observations require immutable PublicRacer records")


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    decision_id: int
    player_id: int
    window: DecisionWindow
    observation: Observation
    legal_actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "window", DecisionWindow(self.window))
        object.__setattr__(self, "legal_actions", tuple(self.legal_actions))
        if not isinstance(self.observation, Observation):
            raise TypeError("decision requests require an immutable Observation")
        if not all(isinstance(action, Action) for action in self.legal_actions):
            raise TypeError("legal actions must be immutable Action records")


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    request: DecisionRequest
    action: Action
    event_sequence: int


class Policy(Protocol):
    def choose(self, request: DecisionRequest) -> Action: ...


class KeepPolicy:
    policy_id = "keep-or-first-v2"

    def choose(self, request: DecisionRequest) -> Action:
        for action in request.legal_actions:
            if action.kind == ActionKind.KEEP:
                return action
        if request.legal_actions:
            return request.legal_actions[0]
        raise ValueError("cannot choose from an empty legal action set")


class RandomPolicy:
    policy_id = "random-legal-v1"

    def __init__(self, seed: int):
        self.seed = seed
        # This stream belongs only to this policy. It is independent from dice
        # and is never included in public observations or decision records.
        self._random = random.Random(seed)

    def choose(self, request: DecisionRequest) -> Action:
        if not request.legal_actions:
            raise ValueError("cannot choose from an empty legal action set")
        return self._random.choice(request.legal_actions)
