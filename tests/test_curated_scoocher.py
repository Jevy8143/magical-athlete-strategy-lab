"""Real-engine H10/H11 notifications on explicitly synthetic tracks.

Every scenario declares four seats and all initial positions. Start=0,
second corner=15, finish=30; these coordinates are fixture choices, not
official Mild/Wild data. Dice supply only fixed legal rolls to the real
engine. No returned movement, notification, or result is mocked.
"""

import pytest

from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Racer
from magical_athlete.state import RacerStatus
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, rolls):
        self.rolls = iter(rolls)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.rolls)


def make_engine(racers, rolls, *, order=(0, 1, 2, 3), trips=()):
    return RaceEngine(
        Track("Scoocher synthetic fixture", 30, 15, frozenset(trips)),
        racers,
        list(order),
        SequenceDice(rolls),
    )


def scoocher_moves(engine, racer_id):
    return [
        event for event in engine.events
        if event.kind == "move" and event.racer_id == racer_id
        and event.movement_kind == "ability" and event.ability == "Scoocher"
    ]


def modifier_ancestors(engine, event):
    """Follow actual causal links rather than infer causes from net distance."""
    by_sequence = {record.sequence: record for record in engine.events}
    ancestors = []
    while event.parent_sequence is not None:
        assert event.parent_sequence < event.sequence
        event = by_sequence[event.parent_sequence]
        if event.kind == "modifier":
            ancestors.append(event)
    return ancestors


def assert_two_independent_scooches(engine, racer_id, start_position, source_ids):
    moves = scoocher_moves(engine, racer_id)
    assert [(event.before, event.after, event.amount) for event in moves] == [
        (start_position, start_position + 1, 1),
        (start_position + 1, start_position + 2, 1),
    ]
    assert [event.source_id for event in moves] == [racer_id, racer_id]
    origins = [modifier_ancestors(engine, event) for event in moves]
    assert [len(ancestors) for ancestors in origins] == [1, 1]
    assert [ancestors[0].source_id for ancestors in origins] == source_ids
    assert len({ancestors[0].sequence for ancestors in origins}) == 2
    return moves


@pytest.mark.rule_vector("T-026")
def test_t026_coach_and_gunk_cancel_but_each_causes_a_separate_scooch():
    # Current P1 shares Coach at 10; Gunk affects P1 from anywhere.
    racers = [Racer(0, "P1", 10), Racer(1, "Coach", 10), Racer(2, "Gunk", 2), Racer(3, "Scoocher", 4)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])

    turn = engine.turns[-1]
    assert (turn.roll, turn.raw_move, turn.final_move, racers[0].position) == (3, 3, 3, 13)
    assert [(m.source_id, m.before, m.after) for m in turn.modifiers] == [(1, 3, 4), (2, 4, 3)]
    assert_two_independent_scooches(engine, 3, 4, [1, 2])
    assert racers[3].position == 6
    assert engine.dice.calls == 1
    assert engine.finish_order == []


@pytest.mark.rule_vector("T-032")
def test_t032_zero_blimp_main_move_still_notifies_for_blimp_and_gunk():
    racers = [Racer(0, "Blimp", 16), Racer(1, "Gunk", 2), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [1])
    engine.take_turn(racers[0])

    turn = engine.turns[-1]
    assert (turn.roll, turn.raw_move, turn.final_move, racers[0].position) == (1, -1, 0, 16)
    assert [(m.source_id, m.before, m.after) for m in turn.modifiers] == [(0, 1, 0), (1, 0, -1)]
    assert not any(event.racer_id == 0 and event.kind in {"move", "passing", "stop", "trip"} for event in engine.events)
    assert_two_independent_scooches(engine, 2, 5, [0, 1])
    assert racers[2].position == 7
    assert engine.dice.calls == 1


def test_scooches_land_separately_and_trip_does_not_create_an_extra_notification():
    # Trip at the intermediate landing proves these are two +1 moves, not +2.
    # Tripping skips a future main move, so the second ability move still occurs.
    racers = [Racer(0, "Blimp", 16), Racer(1, "Gunk", 2), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [1], trips=(6,))
    engine.take_turn(racers[0])

    moves = assert_two_independent_scooches(engine, 2, 5, [0, 1])
    trips = [event for event in engine.events if event.kind == "trip" and event.racer_id == 2]
    assert len(trips) == 1
    assert moves[0].sequence < trips[0].sequence < moves[1].sequence
    assert racers[2].position == 7
    assert racers[2].tripped is True


def test_current_scoocher_responds_to_other_powers_without_self_triggering():
    # The current player is Scoocher, sharing Coach before its main move.
    racers = [Racer(0, "Scoocher", 10), Racer(1, "Coach", 10), Racer(2, "Gunk", 2), Racer(3, "P4", 0)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])

    main_moves = [event for event in engine.events if event.kind == "move" and event.movement_kind == "main"]
    assert [(event.racer_id, event.before, event.after, event.amount) for event in main_moves] == [(0, 10, 13, 3)]
    assert_two_independent_scooches(engine, 0, 13, [1, 2])
    assert racers[0].position == 15
    assert engine.finish_order == []
    assert engine.dice.calls == 1


def test_scoocher_finishing_first_deactivates_its_remaining_notification():
    racers = [Racer(0, "P1", 10), Racer(1, "Coach", 10), Racer(2, "Gunk", 2), Racer(3, "Scoocher", 29)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])

    assert len(engine.turns[-1].modifiers) == 2
    moves = scoocher_moves(engine, 3)
    assert [(event.before, event.after, event.amount) for event in moves] == [(29, 30, 1)]
    assert racers[3].status == RacerStatus.FINISHED
    assert racers[3].finish_place == 1
    assert engine.finish_order == [3]
    assert engine.termination_reason is None
    assert not any(event.kind == "race_end" for event in engine.events)


def test_finished_source_does_not_erase_already_applied_modifier_notifications():
    # Coach finishes first with its own +1 and Gunk's -1 already applied.
    racers = [Racer(0, "Coach", 29), Racer(1, "Gunk", 2), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [1])
    engine.take_turn(racers[0])

    assert racers[0].status == RacerStatus.FINISHED
    assert engine.finish_order == [0]
    assert_two_independent_scooches(engine, 2, 5, [0, 1])
    assert racers[2].position == 7
    assert engine.termination_reason is None


def test_gunk_taking_own_main_move_does_not_notify_from_mere_power_presence():
    # Gunk does not slow itself; Coach is elsewhere and has no actual effect.
    racers = [Racer(0, "Gunk", 10), Racer(1, "Coach", 8), Racer(2, "Scoocher", 3), Racer(3, "P4", 0)]
    engine = make_engine(racers, [2])
    engine.take_turn(racers[0])

    assert engine.turns[-1].modifiers == ()
    assert racers[0].position == 12
    assert racers[2].position == 3
    assert scoocher_moves(engine, 2) == []


def test_second_place_main_move_cancels_pending_scooches_immediately():
    # P2 first finishes from 29; its Gunk notification moves Scoocher 27->28.
    # Coach then finishes second, with two new modifier notifications pending.
    # No notification may move Scoocher after the race has ended.
    racers = [Racer(0, "Coach", 28), Racer(1, "P2", 29), Racer(2, "Gunk", 0), Racer(3, "Scoocher", 27)]
    engine = make_engine(racers, [2, 2], order=(1, 0, 2, 3))
    engine.take_turn(racers[1])
    assert engine.finish_order == [1]
    assert racers[3].position == 28
    boundary = len(engine.events)

    engine.take_turn(racers[0])
    assert len(engine.turns[-1].modifiers) == 2
    assert engine.finish_order == [1, 0]
    assert engine.termination_reason == "second_place"
    assert racers[3].position == 28
    assert not any(event.kind == "move" and event.racer_id == 3 for event in engine.events[boundary:])
    assert engine.events[-1].kind == "race_end"
    assert engine.dice.calls == 2
