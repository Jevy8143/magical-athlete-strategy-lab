"""Four-player drafting and four-race orchestration with explicit data provenance.

Official original-box data and explicitly labelled synthetic fixtures use the
same engine. Official mode requires the verified canonical tracks and awards.
"""

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol

from .decisions import Action, DecisionRequest, KeepPolicy, Policy, RandomPolicy
from .engine import D6, RULE_VERSION, SUPPORTED_ABILITIES, DiagnosticLimitExceeded, RaceEngine
from .racer import Ability, Racer
from .scoring import AwardSchedule, ORIGINAL_BOX_AWARDS, ScoreLedger
from .state import RacerStatus
from .track import MILD_MILE, WILD_WILDS, Track

PLAYERS = (0, 1, 2, 3)


def _plain(value):
    return json.loads(json.dumps(value, default=lambda item: sorted(item) if isinstance(item, (set, frozenset)) else item.value))


@dataclass(frozen=True, slots=True)
class RacerCard:
    card_id: str
    name: str
    ability: Ability | None

    def __post_init__(self):
        if not self.card_id or not self.name:
            raise ValueError("card ID and name must be non-empty")
        if self.ability is not None:
            object.__setattr__(self, "ability", Ability(self.ability))


BASE_DECK = tuple(RacerCard(a.value, a.value, a) for a in Ability)


@dataclass(frozen=True, slots=True)
class GameConfig:
    mild_track: Track
    wild_track: Track
    award_schedule: AwardSchedule
    deck: tuple[RacerCard, ...] = BASE_DECK
    official: bool = False
    source: str = "Explicit synthetic/restricted workflow fixture; not official game data"
    max_turns_per_race: int = 10_000
    max_rolloff_rounds: int = 100
    max_events_per_race: int = 100_000

    def __post_init__(self):
        object.__setattr__(self, "deck", tuple(self.deck))
        ids = [card.card_id for card in self.deck]
        if len(ids) < 16 or len(ids) != len(set(ids)):
            raise ValueError("drafting requires at least 16 unique physical cards")
        if not self.source:
            raise ValueError("game data provenance must be explicit")
        if any(type(budget) is not int or budget < 1 for budget in (
            self.max_turns_per_race, self.max_rolloff_rounds, self.max_events_per_race,
        )):
            raise ValueError("diagnostic budgets must be positive integers")
        if self.official:
            if {c.card_id: c for c in self.deck} != {c.card_id: c for c in BASE_DECK}:
                raise ValueError("an official game requires the original 36-card deck")
            if not self.award_schedule.verified:
                raise ValueError("official awards must be verified, not fixture values")
            if self.mild_track != MILD_MILE:
                raise ValueError("official Mild data must match the canonical verified transcription")
            if self.wild_track != WILD_WILDS:
                raise ValueError("official Wild data must match the canonical verified transcription")
            if self.award_schedule != ORIGINAL_BOX_AWARDS:
                raise ValueError("official awards must match all eight canonical original-box chips")

    def track_for_race(self, race_index: int) -> Track:
        if race_index not in (1, 2, 3, 4):
            raise ValueError("race_index must be 1 through 4")
        return self.mild_track if race_index % 2 else self.wild_track


def official_config() -> GameConfig:
    """Standard four-player original-box game, including project HOUSE rulings."""
    return GameConfig(
        MILD_MILE, WILD_WILDS, ORIGINAL_BOX_AWARDS, official=True,
        source="Verified original-box tracks and chips; docs/06_正式赛道与筹码数据.md",
    )


@dataclass(frozen=True, slots=True)
class DraftPick:
    round_index: int
    player_id: int
    card: RacerCard


@dataclass(frozen=True, slots=True)
class PublicGameObservation:
    phase: str
    race_index: int
    schedule: tuple[str, ...]
    scores: tuple[int, ...]
    hand_sizes: tuple[int, ...]
    used_cards: tuple[tuple[str, ...], ...]
    draft_history: tuple[DraftPick, ...]
    previous_entries: tuple[tuple[str, ...], ...]
    previous_finish_orders: tuple[tuple[int, ...], ...]
    starting_player: int | None = None
    current_awards: tuple[int, int] | None = None
    track: Track | None = None


@dataclass(frozen=True, slots=True)
class DraftRequest:
    decision_id: int
    player_id: int
    round_index: int
    observation: PublicGameObservation
    hand: tuple[RacerCard, ...]
    available: tuple[RacerCard, ...]
    legal_card_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntryRequest:
    decision_id: int
    player_id: int
    observation: PublicGameObservation
    hand: tuple[RacerCard, ...]
    legal_card_ids: tuple[str, ...]


class GamePolicy(Protocol):
    def choose(self, request: DraftRequest | EntryRequest) -> str: ...


class FirstLegalGamePolicy:
    policy_id = "first-legal-game-v1"

    def choose(self, request: DraftRequest | EntryRequest) -> str:
        return request.legal_card_ids[0]


class RandomGamePolicy:
    policy_id = "random-legal-game-v1"

    def __init__(self, seed: int):
        self.seed = seed
        self._random = random.Random(seed)

    def choose(self, request: DraftRequest | EntryRequest) -> str:
        return self._random.choice(request.legal_card_ids)


@dataclass(frozen=True, slots=True)
class GameDecision:
    request: DraftRequest | EntryRequest
    card_id: str


@dataclass(frozen=True, slots=True)
class RollOffRecord:
    purpose: str
    round_index: int
    rolls: tuple[int, ...]
    winner: int | None
    reason: str
    participants: tuple[int, ...] = PLAYERS


@dataclass(frozen=True, slots=True)
class GameRaceRecord:
    race_index: int
    track_kind: str
    starting_player: int
    entries: tuple[RacerCard, ...]
    finish_order: tuple[int, ...]
    final_positions: tuple[int, ...]
    race_scores: tuple[int, ...]
    scores_after: tuple[int, ...]
    termination_reason: str
    elimination_order: tuple[int, ...]
    unplaced: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GameResult:
    scores: tuple[int, ...]
    winners: tuple[int, ...]
    races: tuple[GameRaceRecord, ...]
    official: bool
    champion_counting: str = "Every player tied for highest total is a co-winner"


class GameDiagnosticError(RuntimeError):
    def __init__(self, reason: str, trace: dict):
        super().__init__(reason)
        self.reason = reason
        self.trace = trace


class _LoggedDice:
    def __init__(self, source, records: list[int]):
        self.source = source
        self.records = records

    def roll(self) -> int:
        value = self.source.roll()
        if type(value) is not int or not 1 <= value <= 6:
            raise ValueError("game dice must return integers from 1 to 6")
        self.records.append(value)
        return value


class GameEngine:
    def __init__(
        self, config: GameConfig, seed: int,
        game_policies: Mapping[int, GamePolicy] | None = None,
        race_policies: Mapping[int, Policy] | None = None,
        *, dice=None,
    ):
        if any(set(policies or {}) - set(PLAYERS) for policies in (game_policies, race_policies)):
            raise ValueError("policy maps must contain only player IDs 0 through 3")
        self.config, self.seed = config, seed
        self.game_policies = {p: (game_policies or {}).get(p, FirstLegalGamePolicy()) for p in PLAYERS}
        self.race_policies = {p: (race_policies or {}).get(p, KeepPolicy()) for p in PLAYERS}
        self.game_policy_metadata = {
            str(p): {"policy_id": getattr(policy, "policy_id", type(policy).__qualname__),
                     "seed": policy.seed if isinstance(policy, RandomGamePolicy) else None}
            for p, policy in self.game_policies.items()
        }
        self.dice_rolls: list[int] = []
        self.dice_seed = seed + 1 if dice is None else None
        self.dice = _LoggedDice(dice if dice is not None else D6(self.dice_seed), self.dice_rolls)
        self.remaining_deck = list(config.deck)
        random.Random(seed).shuffle(self.remaining_deck)
        self.hands: dict[int, list[RacerCard]] = {p: [] for p in PLAYERS}
        self.used_cards: dict[int, list[RacerCard]] = {p: [] for p in PLAYERS}
        self.ability_cards_out: list[RacerCard] = []
        self.ledger = ScoreLedger(PLAYERS)
        self.pool: list[RacerCard] = []
        self.active_entries: tuple[RacerCard, ...] = ()
        self.draft_history: list[DraftPick] = []
        self.decisions: list[GameDecision] = []
        self.rolloffs: list[RollOffRecord] = []
        self.races: list[GameRaceRecord] = []
        self.race_traces: list[dict] = []
        self.active_race: RaceEngine | None = None
        self.draft_start: int | None = None
        self.first_race_start: int | None = None
        self.current_starting_player: int | None = None
        self.drafted = False
        self._replay_race_policy_metadata: list[dict] | None = None
        self.initial_code_fingerprint = self._code_fingerprint()

    @staticmethod
    def _code_fingerprint() -> str:
        digest = hashlib.sha256()
        for name in ("game.py", "scoring.py", "engine.py", "events.py", "racer.py", "state.py", "track.py", "decisions.py", "race_actions.py",
                     "copy_banana_cycle.py", "fish_baby_cycle.py", "star_baby_cycle.py", "romantic_queue_cycle.py",
                     "preparing_copy_romantic_cycle.py", "romantic_anchor_cycle.py", "romantic_baby_star_cycle.py",
                     "static_inheritance.py", "fish_baby_star_cycle.py", "romantic_baby_arrow_cycle.py",
                     "romantic_duel_copy_cycle.py", "egg_copy_scoocher_cycle.py", "romantic_scoocher_anchor_cycle.py",
                     "romantic_baby_baba_cycle.py", "copy_fish_baby_cycle.py", "centaur_copy_baby_star_cycle.py",
                     "fish_baby_arrow11_cycle.py", "romantic_baby_shared12_cycle.py", "scoocher_baby_duel22_cycle.py", "fish_baby_baba4_cycle.py", "fish_baby_arrow11_scoocher_cycle.py", "duel_romantic_scoocher22_cycle.py", "copy_baby_duel_scoocher_cycle.py"):
            digest.update(name.encode())
            digest.update(Path(__file__).with_name(name).read_bytes())
        return digest.hexdigest()

    def _diagnostic(self, reason: str):
        trace = self.trace()
        trace["diagnostic_failure"] = reason
        raise GameDiagnosticError(reason, trace)

    def _assert_card_locations(self):
        located = self.remaining_deck + self.pool + list(self.active_entries) + self.ability_cards_out
        located += [c for hand in self.hands.values() for c in hand]
        located += [c for used in self.used_cards.values() for c in used]
        ids = [c.card_id for c in located]
        if len(ids) != len(set(ids)) or set(ids) != {c.card_id for c in self.config.deck}:
            raise RuntimeError("physical card uniqueness/accounting invariant failed")

    def _consume_egg_draws(self, drawn: list[Ability] | tuple[Ability, ...]) -> None:
        """Move actual drawn physical cards out of the remaining deck once."""
        remaining = self.remaining_deck.copy()
        moved = []
        for raw_ability in drawn:
            ability = Ability(raw_ability)
            card = next((card for card in remaining if card.ability == ability), None)
            if card is None:
                raise RuntimeError("Egg drew an ability without a remaining physical card")
            remaining.remove(card)
            moved.append(card)
        self.remaining_deck[:] = remaining
        self.ability_cards_out.extend(moved)
        self._assert_card_locations()

    def _observation(self, phase: str, race_index: int) -> PublicGameObservation:
        is_entry = phase == "entry_selection"
        return PublicGameObservation(
            phase, race_index, ("Mild", "Wild", "Mild", "Wild"),
            tuple(self.ledger.total(p) for p in PLAYERS),
            tuple(len(self.hands[p]) for p in PLAYERS),
            tuple(tuple(c.card_id for c in self.used_cards[p]) for p in PLAYERS),
            tuple(self.draft_history),
            tuple(tuple(c.card_id for c in r.entries) for r in self.races),
            tuple(r.finish_order for r in self.races),
            self.current_starting_player if is_entry else None,
            self.config.award_schedule.for_race(race_index) if is_entry else None,
            self.config.track_for_race(race_index) if is_entry else None,
        )

    def roll_off(self, purpose: str, participants: tuple[int, ...] = PLAYERS) -> int:
        participants = tuple(participants)
        if len(participants) < 2 or len(set(participants)) != len(participants) or set(participants) - set(PLAYERS):
            raise ValueError("roll-off participants must be two to four distinct player IDs")
        for round_index in range(1, self.config.max_rolloff_rounds + 1):
            rolls = tuple(self.dice.roll() for _ in participants)
            counts = Counter(rolls)
            unique = sorted((value for value, count in counts.items() if count == 1), reverse=True)
            if unique:
                winner = participants[rolls.index(unique[0])]
                self.rolloffs.append(RollOffRecord(purpose, round_index, rolls, winner, "highest_unique_roll", participants))
                return winner
            # HOUSE H12: no unique result means everyone rerolls, including
            # several tied groups such as 6,6,5,5 (user ruling 2026-10-29).
            reason = "everyone_same_roll_reroll" if len(counts) == 1 else "no_unique_roll_reroll"
            self.rolloffs.append(RollOffRecord(purpose, round_index, rolls, None, reason, participants))
        self._diagnostic("rolloff_budget_exceeded")

    @staticmethod
    def snake_order(start_player: int) -> tuple[int, ...]:
        clockwise = tuple((start_player + offset) % 4 for offset in range(4))
        return clockwise + tuple(reversed(clockwise))

    def _game_choice(self, request: DraftRequest | EntryRequest) -> str:
        choice = self.game_policies[request.player_id].choose(request)
        if not isinstance(choice, str) or choice not in request.legal_card_ids:
            raise ValueError("game policy returned an illegal card choice")
        self.decisions.append(GameDecision(request, choice))
        return choice

    def draft(self) -> dict[int, tuple[RacerCard, ...]]:
        if self.drafted or self.draft_history:
            raise RuntimeError("draft may only be performed once")
        self.pool = self.remaining_deck[:8]
        del self.remaining_deck[:8]
        self.draft_start = self.roll_off("draft_start")
        for round_index in (1, 2):
            if round_index == 2:
                self.pool = self.remaining_deck[:8]
                del self.remaining_deck[:8]
            start = (self.draft_start + round_index - 1) % 4
            for player_id in self.snake_order(start):
                request = DraftRequest(
                    len(self.decisions) + 1, player_id, round_index,
                    self._observation("draft", 0), tuple(self.hands[player_id]),
                    tuple(self.pool), tuple(c.card_id for c in self.pool),
                )
                choice = self._game_choice(request)
                card = next(c for c in self.pool if c.card_id == choice)
                self.pool.remove(card)
                self.hands[player_id].append(card)
                self.draft_history.append(DraftPick(round_index, player_id, card))
                self._assert_card_locations()
        self.drafted = True
        return {p: tuple(self.hands[p]) for p in PLAYERS}

    def select_entries(self, race_index: int) -> tuple[RacerCard, ...]:
        if not self.drafted or race_index != len(self.races) + 1 or self.active_entries:
            raise RuntimeError("entry selection is not currently legal")
        if race_index not in (1, 2, 3, 4):
            raise ValueError("race_index must be 1 through 4")
        if self.current_starting_player is None:
            if race_index == 1:
                if self.first_race_start is None:
                    self.first_race_start = self.roll_off("first_race_start")
                self.current_starting_player = self.first_race_start
            else:
                self.current_starting_player = self._next_start(self.races[-1])
        observation = self._observation("entry_selection", race_index)
        requests = tuple(EntryRequest(
            len(self.decisions) + p + 1, p, observation,
            tuple(self.hands[p]), tuple(c.card_id for c in self.hands[p]),
        ) for p in PLAYERS)
        # No mutation/reveal while any player has yet to submit a choice.
        pending = tuple(self._game_choice(request) for request in requests)
        entries = tuple(next(c for c in self.hands[p] if c.card_id == pending[p]) for p in PLAYERS)
        for p, card in zip(PLAYERS, entries):
            self.hands[p].remove(card)
        self.active_entries = entries
        self._assert_card_locations()
        return entries

    def _next_start(self, race: GameRaceRecord) -> int:
        if race.elimination_order:
            return race.elimination_order[0]
        if not race.unplaced:
            self._diagnostic("next_start_has_no_unplaced_racer")
        last_position = min(race.final_positions[p] for p in race.unplaced)
        candidates = tuple(p for p in race.unplaced if race.final_positions[p] == last_position)
        if len(candidates) != 1:
            # HOUSE H13: only the tied-last players roll; apply the same
            # highest-unique/H12-reroll procedure to that participant subset.
            return self.roll_off(f"race_{race.race_index + 1}_start_tied_last", candidates)
        return candidates[0]

    def run(self) -> GameResult:
        if not self.drafted:
            self.draft()
        if self.first_race_start is None:
            self.first_race_start = self.roll_off("first_race_start")
        if self.config.official and set(Ability) - SUPPORTED_ABILITIES:
            raise NotImplementedError("official complete-game acceptance requires all 36 abilities")
        while len(self.races) < 4:
            race_index = len(self.races) + 1
            start = self.first_race_start if not self.races else self._next_start(self.races[-1])
            self.current_starting_player = start
            entries = self.select_entries(race_index)
            egg_cards_consumed = False
            try:
                engine = RaceEngine(
                    self.config.track_for_race(race_index),
                    [Racer(p, card.name, ability=card.ability) for p, card in zip(PLAYERS, entries)],
                    [(start + offset) % 4 for offset in range(4)], self.dice,
                    policies=self.race_policies, score_ledger=self.ledger, race_index=race_index,
                    max_events=self.config.max_events_per_race,
                    remaining_deck=tuple(card.ability for card in self.remaining_deck if card.ability is not None),
                    previous_winners=tuple(
                        race.entries[race.finish_order[0]].ability for race in self.races
                        if race.finish_order and race.entries[race.finish_order[0]].ability is not None
                    ),
                )
                self.active_race = engine
                self._consume_egg_draws(engine.egg_drawn)
                egg_cards_consumed = True
                if self._replay_race_policy_metadata is not None:
                    engine.policy_metadata = self._replay_race_policy_metadata[race_index - 1]
                result = engine.run(max_turns=self.config.max_turns_per_race)
            except DiagnosticLimitExceeded as exc:
                if not egg_cards_consumed:
                    self._consume_egg_draws(exc.trace.get("egg_drawn", ()))
                trace = self.trace()
                trace["race_diagnostic"] = exc.trace
                reason = exc.trace.get("diagnostic_failure", "incomplete_race")
                trace["race_diagnostic_reason"] = reason
                trace["diagnostic_failure"] = f"race_{reason}"
                raise GameDiagnosticError(f"race_{reason}", trace) from exc
            # Keep race-only traces before adding game-level placement awards;
            # each race remains independently replayable from its starting ledger.
            race_trace = engine.trace()
            self.config.award_schedule.award_finish_order(
                self.ledger, race_index, result.finish_order,
                event_sequence=result.events[-1].sequence if result.events else None,
            )
            elimination_order = []
            for event in result.events:
                if event.kind in ("eliminate", "eliminated"):
                    eliminated = event.target_id if event.target_id is not None else event.racer_id
                    if eliminated not in elimination_order:
                        elimination_order.append(eliminated)
            unplaced = tuple(p for p in PLAYERS if engine.racers[p].status != RacerStatus.FINISHED)
            record = GameRaceRecord(
                race_index, "Mild" if race_index % 2 else "Wild", start, entries,
                tuple(result.finish_order), tuple(result.final_positions[p] for p in PLAYERS),
                tuple(self.ledger.race_total(p, race_index) for p in PLAYERS),
                tuple(self.ledger.total(p) for p in PLAYERS), result.termination_reason,
                tuple(elimination_order), unplaced,
            )
            self.races.append(record)
            self.race_traces.append(race_trace)
            for p, card in zip(PLAYERS, entries):
                self.used_cards[p].append(card)
            self.active_entries = ()
            self.active_race = None
            self.current_starting_player = None
            self._assert_card_locations()
        return GameResult(
            tuple(self.ledger.total(p) for p in PLAYERS), self.ledger.winners(),
            tuple(self.races), self.config.official,
        )

    def trace(self) -> dict:
        return _plain({
            "schema_version": 1, "rule_version": RULE_VERSION,
            "code_fingerprint": self.initial_code_fingerprint,
            "run_kind": "official_game" if self.config.official else "synthetic_or_restricted_game_fixture",
            "seed": self.seed, "dice_seed": self.dice_seed,
            "config": asdict(self.config), "game_policies": self.game_policy_metadata,
            "dice_rolls": self.dice_rolls.copy(), "decisions": [asdict(d) for d in self.decisions],
            "draft_history": [asdict(p) for p in self.draft_history],
            "rolloffs": [asdict(r) for r in self.rolloffs],
            "draft_start": self.draft_start, "first_race_start": self.first_race_start,
            "current_starting_player": self.current_starting_player,
            "drafted": self.drafted, "hands": {str(p): [asdict(c) for c in self.hands[p]] for p in PLAYERS},
            "remaining_deck": [asdict(c) for c in self.remaining_deck],
            "pool": [asdict(c) for c in self.pool],
            "active_entries": [asdict(c) for c in self.active_entries],
            "used_cards": {str(p): [asdict(c) for c in self.used_cards[p]] for p in PLAYERS},
            "ability_cards_out": [asdict(c) for c in self.ability_cards_out],
            "races": [asdict(r) for r in self.races], "race_traces": self.race_traces,
            "active_race": self.active_race.trace() if self.active_race is not None else None,
            "scores": self.ledger.trace(), "completed": len(self.races) == 4,
            "winners": self.ledger.winners() if len(self.races) == 4 else (),
        })


def replay_game_trace(trace: dict) -> GameEngine:
    """Re-run a completed game from actual dice and choices, not old policies."""
    trace = _plain(trace)
    if trace["schema_version"] != 1 or trace["rule_version"] != RULE_VERSION:
        raise ValueError("incompatible game replay schema or rule version")
    if trace["code_fingerprint"] != GameEngine._code_fingerprint():
        raise ValueError("game replay requires the same code version")
    if not trace["completed"]:
        raise ValueError("complete-game replay requires a completed four-race trace")
    decisions = iter(trace["decisions"])
    race_decisions = iter(d for r in trace["race_traces"] for d in r["decisions"])

    class RecordedGamePolicy:
        def choose(self, request):
            try:
                record = next(decisions)
            except StopIteration as exc:
                raise ValueError("game replay lacks a game decision") from exc
            if _plain(asdict(request)) != record["request"]:
                raise ValueError("game replay public/private request diverged")
            return record["card_id"]

    class RecordedRacePolicy:
        def choose(self, request: DecisionRequest):
            try:
                record = next(race_decisions)
            except StopIteration as exc:
                raise ValueError("game replay lacks a race decision") from exc
            if _plain(asdict(request)) != record["request"]:
                raise ValueError("game replay race request diverged")
            return Action(**record["action"])

    class RecordedDice:
        def __init__(self):
            self.values = iter(trace["dice_rolls"])

        def roll(self):
            return next(self.values)

    data = trace["config"]
    tracks = []
    for name in ("mild_track", "wild_track"):
        track_data = {**data[name], "trip_positions": frozenset(data[name]["trip_positions"])}
        tracks.append(Track(**track_data))
    awards = AwardSchedule(**{**data["award_schedule"], "awards": tuple(tuple(a) for a in data["award_schedule"]["awards"])})
    config = GameConfig(
        *tracks, awards, deck=tuple(RacerCard(**c) for c in data["deck"]),
        official=data["official"], source=data["source"],
        max_turns_per_race=data["max_turns_per_race"], max_rolloff_rounds=data["max_rolloff_rounds"],
        max_events_per_race=data["max_events_per_race"],
    )
    engine = GameEngine(
        config, trace["seed"], {p: RecordedGamePolicy() for p in PLAYERS},
        {p: RecordedRacePolicy() for p in PLAYERS}, dice=RecordedDice(),
    )
    engine.game_policy_metadata = trace["game_policies"]
    engine.dice_seed = trace["dice_seed"]
    engine._replay_race_policy_metadata = [r["policies"] for r in trace["race_traces"]]
    try:
        engine.run()
    except StopIteration as exc:
        raise ValueError("game replay has insufficient dice") from exc
    if engine.trace() != trace:
        raise ValueError("game replay diverged from recorded state/events/scores")
    return engine
