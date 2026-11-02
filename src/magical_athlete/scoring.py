"""Auditable point changes and explicitly sourced four-race placement awards.

The ledger records actual deltas, including penalties, rather than assuming a
fixed supply of points per game. It does not decide when a power is legal.
"""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Iterable, Mapping


class ScorePhase(StrEnum):
    BEFORE_RACE = "before_race"
    DURING_RACE = "during_race"
    AFTER_RACE = "after_race"


def _integer(value: int, name: str) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")


def _race_index(value: int) -> None:
    _integer(value, "race_index")
    if not 1 <= value <= 4:
        raise ValueError("race_index must be from 1 through 4")


def _label(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ScoreEntry:
    sequence: int
    player_id: int
    race_index: int
    delta: int
    source: str
    reason: str
    event_sequence: int | None = None
    phase: ScorePhase = ScorePhase.DURING_RACE

    def __post_init__(self) -> None:
        _integer(self.sequence, "sequence")
        if self.sequence < 1:
            raise ValueError("score sequence must be positive")
        _integer(self.player_id, "player_id")
        _race_index(self.race_index)
        _integer(self.delta, "delta")
        _label(self.source, "source")
        _label(self.reason, "reason")
        if self.event_sequence is not None:
            _integer(self.event_sequence, "event_sequence")
            if self.event_sequence < 1:
                raise ValueError("event_sequence must be positive when provided")
        object.__setattr__(self, "phase", ScorePhase(self.phase))


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    player_ids: tuple[int, ...]
    totals: tuple[tuple[int, int], ...]
    race_totals: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    entries: tuple[ScoreEntry, ...]

    def total(self, player_id: int) -> int:
        try:
            return dict(self.totals)[player_id]
        except KeyError as exc:
            raise ValueError("unknown player ID") from exc

    def race_total(self, player_id: int, race_index: int) -> int:
        _race_index(race_index)
        self.total(player_id)
        return dict(dict(self.race_totals)[race_index])[player_id]


class ScoreLedger:
    """Append-only account of point changes attributed to races 1..4.

Game orchestration enforces four participants. The ledger also accepts smaller
explicit ID collections for independent engine fixtures, without adding another
game mode. Totals cannot be negative; a race subtotal can be negative when that
race spends points earned earlier in the same game.
    """

    def __init__(self, player_ids: Iterable[int] = (0, 1, 2, 3)):
        player_ids = tuple(player_ids)
        if not player_ids:
            raise ValueError("at least one player is required")
        for player_id in player_ids:
            _integer(player_id, "player_id")
        if len(set(player_ids)) != len(player_ids):
            raise ValueError("player IDs must be unique")
        self.player_ids = player_ids
        self._entries: list[ScoreEntry] = []
        self._totals = {player_id: 0 for player_id in player_ids}
        self._race_totals = {
            race: {player_id: 0 for player_id in player_ids} for race in range(1, 5)
        }

    def _player(self, player_id: int) -> None:
        _integer(player_id, "player_id")
        if player_id not in self._totals:
            raise ValueError("unknown player ID")

    @property
    def entries(self) -> tuple[ScoreEntry, ...]:
        return tuple(self._entries)

    def record(
        self, player_id: int, race_index: int, delta: int, *,
        source: str, reason: str, event_sequence: int | None = None,
        phase: ScorePhase | str = ScorePhase.DURING_RACE,
    ) -> ScoreEntry:
        self._player(player_id)
        entry = ScoreEntry(
            len(self._entries) + 1, player_id, race_index, delta,
            source, reason, event_sequence, ScorePhase(phase),
        )
        if self._totals[player_id] + delta < 0:
            raise ValueError("point balance cannot become negative; clamp an actual deduction explicitly")
        self._entries.append(entry)
        self._totals[player_id] += delta
        self._race_totals[race_index][player_id] += delta
        return entry

    def deduct_up_to_balance(
        self, player_id: int, race_index: int, amount: int, *,
        source: str, reason: str, event_sequence: int | None = None,
        phase: ScorePhase | str = ScorePhase.DURING_RACE,
    ) -> ScoreEntry:
        """Record the actual penalty, never a debt or a per-character allowance.

An exhausted balance records delta=0, preserving the audit of the resolved
effect. Whether that effect notifies another power remains an engine decision.
        """
        self._player(player_id)
        _integer(amount, "amount")
        if amount < 0:
            raise ValueError("deduction amount cannot be negative")
        actual = min(amount, self._totals[player_id])
        return self.record(
            player_id, race_index, -actual, source=source, reason=reason,
            event_sequence=event_sequence, phase=phase,
        )

    def total(self, player_id: int) -> int:
        self._player(player_id)
        return self._totals[player_id]

    def race_total(self, player_id: int, race_index: int) -> int:
        self._player(player_id)
        _race_index(race_index)
        return self._race_totals[race_index][player_id]

    def totals(self) -> dict[int, int]:
        return self._totals.copy()

    def race_totals(self, race_index: int) -> dict[int, int]:
        _race_index(race_index)
        return self._race_totals[race_index].copy()

    def winners(self) -> tuple[int, ...]:
        """Return every highest-scoring player, in the declared seat order."""
        high = max(self._totals.values())
        return tuple(player_id for player_id in self.player_ids if self._totals[player_id] == high)

    def snapshot(self) -> ScoreSnapshot:
        return ScoreSnapshot(
            self.player_ids,
            tuple(self._totals.items()),
            tuple((race, tuple(self._race_totals[race].items())) for race in range(1, 5)),
            self.entries,
        )

    def trace(self) -> dict:
        return {
            "schema_version": 1,
            "player_ids": list(self.player_ids),
            "entries": [
                {**asdict(entry), "phase": entry.phase.value} for entry in self._entries
            ],
            "totals": {str(player_id): total for player_id, total in self._totals.items()},
            "race_totals": {
                str(race): {str(player_id): total for player_id, total in totals.items()}
                for race, totals in self._race_totals.items()
            },
        }

    @classmethod
    def from_trace(cls, trace: Mapping) -> "ScoreLedger":
        """Rebuild by applying actual deltas and verify every redundant total."""
        trace = json.loads(json.dumps(trace))
        expected_keys = {"schema_version", "player_ids", "entries", "totals", "race_totals"}
        if set(trace) != expected_keys or type(trace.get("schema_version")) is not int or trace["schema_version"] != 1:
            raise ValueError("incompatible score ledger trace")
        ledger = cls(trace["player_ids"])
        try:
            for raw in trace["entries"]:
                entry = ScoreEntry(**raw)
                if entry.sequence != len(ledger._entries) + 1:
                    raise ValueError("score entry sequence is not contiguous")
                ledger.record(
                    entry.player_id, entry.race_index, entry.delta,
                    source=entry.source, reason=entry.reason,
                    event_sequence=entry.event_sequence, phase=entry.phase,
                )
        except (TypeError, KeyError) as exc:
            raise ValueError("invalid score ledger entry") from exc
        if ledger.trace() != trace:
            raise ValueError("score ledger entries or totals diverged")
        return ledger


@dataclass(frozen=True, slots=True)
class AwardSchedule:
    """Explicit gold/silver values for all four races; no inferred defaults."""

    awards: tuple[tuple[int, int], ...]
    source: str
    verified: bool = False
    data_version: str = "custom-v1"

    def __post_init__(self) -> None:
        awards = tuple(tuple(pair) for pair in self.awards)
        if len(awards) != 4 or any(len(pair) != 2 for pair in awards):
            raise ValueError("provide gold/silver pairs for exactly four races")
        for pair in awards:
            for value in pair:
                _integer(value, "award")
                if value < 0:
                    raise ValueError("placement awards cannot be negative")
        _label(self.source, "source")
        _label(self.data_version, "data_version")
        if type(self.verified) is not bool:
            raise ValueError("verified must be a boolean")
        object.__setattr__(self, "awards", awards)

    def for_race(self, race_index: int) -> tuple[int, int]:
        _race_index(race_index)
        return self.awards[race_index - 1]

    def award_finish_order(
        self, ledger: ScoreLedger, race_index: int, finish_order: Iterable[int], *,
        event_sequence: int | None = None,
    ) -> tuple[ScoreEntry, ...]:
        """Award existing places only, retaining Mastermind's self-prediction.

The game calls this once at race completion. Duplicate player IDs are legal:
the same participant can receive gold and silver through Mastermind's exception.
No points are manufactured for unawarded places in a terminating loop.
        """
        awards = self.for_race(race_index)
        finish_order = tuple(finish_order)
        if len(finish_order) > 2:
            raise ValueError("only first and second places receive placement awards")
        # Validate every recipient and event link before the first mutation.
        for player_id in finish_order:
            ledger._player(player_id)
        if event_sequence is not None:
            _integer(event_sequence, "event_sequence")
            if event_sequence < 1:
                raise ValueError("event_sequence must be positive when provided")
        return tuple(
            ledger.record(
                player_id, race_index, awards[place], source=self.source,
                reason=("placement_gold" if place == 0 else "placement_silver"),
                event_sequence=event_sequence, phase=ScorePhase.AFTER_RACE,
            )
            for place, player_id in enumerate(finish_order)
        )


# All eight cardboard chips are individually visible in the comparison photo;
# repeated 4/2 values are physical chips, not an inferred progression. The
# rulebook p11/p18 specifies stacking by size and awarding the top pair.
ORIGINAL_BOX_AWARDS = AwardSchedule(
    ((3, 1), (4, 2), (4, 2), (5, 3)),
    source=(
        "Original cardboard chips pictured at "
        "https://www.etsy.com/listing/4513805386/magical-athlete-board-game-acrylic; "
        "official CMYK rulebook p11/p18 for ascending-size order; "
        "see docs/06_正式赛道与筹码数据.md"
    ),
    verified=True,
    data_version="2026-10-29-original-box-chips-v1",
)
