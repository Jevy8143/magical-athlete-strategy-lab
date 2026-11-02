from dataclasses import dataclass


@dataclass(frozen=True)
class RaceEvent:
    sequence: int
    kind: str
    racer_id: int
    source_id: int | None = None
    target_id: int | None = None
    ability: str | None = None
    before: int | None = None
    after: int | None = None
    amount: int | None = None
    parent_sequence: int | None = None
    movement_kind: str | None = None
    reason: str | None = None
    operation: str | None = None
    action_kind: str | None = None
    decision_id: int | None = None
    arrival_batch_id: int | None = None


@dataclass(frozen=True)
class PendingEffect:
    racer_id: int
    source_id: int
    trigger_sequence: int
    queued_sequence: int
    kind: str = "move"
    steps: int = 1
    ability: str = "Scoocher"
    target_id: int | None = None
