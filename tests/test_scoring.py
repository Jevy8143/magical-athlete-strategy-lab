"""Actual score accounting with explicitly synthetic placement values."""

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from magical_athlete.scoring import AwardSchedule, ScoreLedger, ScorePhase


def synthetic_awards():
    # Intentionally irregular fixture values; these are not official chips.
    return AwardSchedule(((7, 2), (4, 3), (11, 1), (6, 5)), source="synthetic accounting fixture")


def test_score_total_contains_before_during_after_and_negative_changes():
    ledger = ScoreLedger()
    ledger.record(0, 2, 4, source="Sisyphus", reason="before_race_points", phase=ScorePhase.BEFORE_RACE)
    ledger.record(0, 2, -1, source="Sisyphus", reason="final_six_penalty", event_sequence=17)
    ledger.record(0, 2, 1, source="Star", reason="stopped_on_star", event_sequence=23)
    ledger.record(0, 2, 1, source="Loveable Loser", reason="alone_in_last_place", event_sequence=27)
    synthetic_awards().award_finish_order(ledger, 2, [0, 1], event_sequence=50)
    assert ledger.total(0) == 9
    assert ledger.total(1) == 3
    assert ledger.race_total(0, 2) == 9
    assert [entry.delta for entry in ledger.entries] == [4, -1, 1, 1, 4, 3]
    assert [entry.phase for entry in ledger.entries[:2]] == [ScorePhase.BEFORE_RACE, ScorePhase.DURING_RACE]
    assert ledger.entries[-1].phase == ScorePhase.AFTER_RACE
    assert [entry.sequence for entry in ledger.entries] == list(range(1, 7))


def test_sisyphus_spends_whole_game_balance_not_only_its_four_initial_points():
    ledger = ScoreLedger()
    ledger.record(0, 1, 3, source="synthetic prior-race award", reason="fixture_opening_balance")
    ledger.record(0, 2, 4, source="Sisyphus", reason="before_race_points", phase="before_race")
    penalties = [
        ledger.deduct_up_to_balance(0, 2, 1, source="Sisyphus", reason="final_six_penalty")
        for _ in range(9)
    ]
    assert [entry.delta for entry in penalties] == [-1] * 7 + [0, 0]
    assert ledger.total(0) == 0
    assert ledger.race_total(0, 1) == 3
    assert ledger.race_total(0, 2) == -3
    assert len(ledger.entries) == 11


def test_deduction_records_actual_partial_amount_and_zero_without_debt():
    ledger = ScoreLedger()
    ledger.record(2, 1, 2, source="fixture", reason="opening_points")
    penalty = ledger.deduct_up_to_balance(2, 3, 5, source="penalty fixture", reason="deduction")
    assert penalty.delta == -2
    assert ledger.total(2) == 0
    zero = ledger.deduct_up_to_balance(2, 3, 1, source="penalty fixture", reason="deduction")
    assert zero.delta == 0
    assert zero.sequence == 3
    assert ledger.race_total(2, 3) == -2


def test_raw_overdraft_is_rejected_atomically_instead_of_changing_the_reward():
    ledger = ScoreLedger()
    ledger.record(0, 1, 1, source="fixture", reason="opening_points")
    before = ledger.trace()
    with pytest.raises(ValueError, match="negative"):
        ledger.record(0, 2, -2, source="fixture", reason="unclamped_penalty")
    assert ledger.trace() == before


def test_all_four_race_subtotals_sum_to_real_cumulative_points():
    ledger = ScoreLedger()
    schedule = synthetic_awards()
    for race_index, order in enumerate(([0, 1], [1, 2], [0, 3], [2, 3]), 1):
        schedule.award_finish_order(ledger, race_index, order)
    ledger.record(3, 4, 2, source="Loveable Loser", reason="two_legal_start_of_turn_effects")
    ledger.record(0, 3, -1, source="Sisyphus", reason="penalty_fixture")
    assert ledger.totals() == {0: 17, 1: 6, 2: 9, 3: 8}
    assert {player_id: sum(ledger.race_total(player_id, r) for r in range(1, 5)) for player_id in ledger.player_ids} == ledger.totals()
    assert sum(ledger.totals().values()) == sum(sum(pair) for pair in schedule.awards) + 1


def test_highest_score_ties_are_joint_winners_in_declared_seat_order():
    ledger = ScoreLedger((40, 10, 30, 20))
    assert ledger.winners() == (40, 10, 30, 20)
    ledger.record(30, 1, 4, source="fixture", reason="points")
    ledger.record(40, 4, 4, source="fixture", reason="points")
    assert ledger.winners() == (40, 30)


def test_snapshot_is_deeply_immutable_and_does_not_change_after_later_entries():
    ledger = ScoreLedger()
    ledger.record(0, 1, 2, source="fixture", reason="points")
    snapshot = ledger.snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.totals = ()
    with pytest.raises(FrozenInstanceError):
        snapshot.entries[0].delta = 99
    with pytest.raises(AttributeError):
        snapshot.entries.append(snapshot.entries[0])
    ledger.record(0, 2, 3, source="fixture", reason="points")
    assert snapshot.total(0) == snapshot.race_total(0, 1) == 2
    assert snapshot.race_total(0, 2) == 0
    assert len(snapshot.entries) == 1
    assert ledger.total(0) == 5


def test_returned_totals_are_detached_copies():
    ledger = ScoreLedger()
    totals = ledger.totals()
    race = ledger.race_totals(1)
    totals[0], race[0] = 90, 100
    assert ledger.total(0) == ledger.race_total(0, 1) == 0


def test_score_trace_rebuilds_all_entries_and_redundant_totals():
    ledger = ScoreLedger((3, 2, 1, 0))
    ledger.record(3, 1, 6, source="fixture", reason="points", event_sequence=4, phase="before_race")
    ledger.deduct_up_to_balance(3, 4, 2, source="Sisyphus", reason="penalty", event_sequence=40)
    synthetic_awards().award_finish_order(ledger, 4, [1, 2], event_sequence=80)
    trace = json.loads(json.dumps(ledger.trace()))
    restored = ScoreLedger.from_trace(trace)
    assert restored.trace() == trace
    assert restored.snapshot() == ledger.snapshot()
    trace["totals"]["3"] = 999
    assert restored.total(3) == 4


@pytest.mark.parametrize("change", ["delta", "sequence", "total", "race_total", "player", "phase"])
def test_score_trace_rejects_forged_changes_and_noncontiguous_entries(change):
    ledger = ScoreLedger()
    ledger.record(0, 1, 3, source="fixture", reason="points")
    trace = copy.deepcopy(ledger.trace())
    if change == "delta":
        trace["entries"][0]["delta"] = 8
    elif change == "sequence":
        trace["entries"][0]["sequence"] = 2
    elif change == "total":
        trace["totals"]["0"] = 8
    elif change == "race_total":
        trace["race_totals"]["1"]["0"] = 8
    elif change == "player":
        trace["entries"][0]["player_id"] = 9
    else:
        trace["entries"][0]["phase"] = "future"
    with pytest.raises(ValueError):
        ScoreLedger.from_trace(trace)


def test_explicit_awards_can_give_same_player_gold_and_silver_for_mastermind():
    ledger = ScoreLedger()
    entries = synthetic_awards().award_finish_order(ledger, 1, [2, 2])
    assert [(entry.player_id, entry.delta, entry.reason) for entry in entries] == [
        (2, 7, "placement_gold"), (2, 2, "placement_silver"),
    ]
    assert ledger.total(2) == 9
    assert ledger.winners() == (2,)


@pytest.mark.parametrize("order,total", [([], 0), ([0], 7)])
def test_missing_places_are_not_fabricated_when_a_race_ends_early(order, total):
    ledger = ScoreLedger()
    entries = synthetic_awards().award_finish_order(ledger, 1, order)
    assert len(entries) == len(order)
    assert sum(ledger.totals().values()) == total


def test_all_award_recipients_are_validated_before_any_points_are_added():
    ledger = ScoreLedger()
    with pytest.raises(ValueError, match="unknown"):
        synthetic_awards().award_finish_order(ledger, 1, [0, 99])
    assert ledger.entries == ()
    with pytest.raises(ValueError, match="first and second"):
        synthetic_awards().award_finish_order(ledger, 1, [0, 1, 2])
    assert ledger.entries == ()


def test_award_schedule_has_no_default_values_and_freezes_caller_lists():
    with pytest.raises(TypeError):
        AwardSchedule()
    values = [[7, 2], [4, 3], [11, 1], [6, 5]]
    schedule = AwardSchedule(values, source="synthetic fixture", verified=False)
    values[0][0] = 99
    assert schedule.for_race(1) == (7, 2)
    assert schedule.verified is False
    with pytest.raises(FrozenInstanceError):
        schedule.source = "different"


@pytest.mark.parametrize("awards", [((3, 1),), ((3, 1),) * 5, ((3, 1, 0),) * 4, ((-1, 0),) * 4, ((True, 1),) * 4])
def test_awards_reject_incomplete_invalid_or_noninteger_values(awards):
    with pytest.raises(ValueError):
        AwardSchedule(awards, source="synthetic invalid fixture")


@pytest.mark.parametrize("player,race,delta,source,reason", [
    (99, 1, 1, "fixture", "points"),
    (True, 1, 1, "fixture", "points"),
    (0, 0, 1, "fixture", "points"),
    (0, 5, 1, "fixture", "points"),
    (0, True, 1, "fixture", "points"),
    (0, 1, 1.5, "fixture", "points"),
    (0, 1, True, "fixture", "points"),
    (0, 1, 1, "", "points"),
    (0, 1, 1, "fixture", " "),
])
def test_invalid_score_entries_are_rejected_before_mutation(player, race, delta, source, reason):
    ledger = ScoreLedger()
    with pytest.raises(ValueError):
        ledger.record(player, race, delta, source=source, reason=reason)
    assert ledger.entries == ()
    assert ledger.totals() == {0: 0, 1: 0, 2: 0, 3: 0}
