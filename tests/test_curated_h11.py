"""Real-engine T-027..031 and T-033 scenarios.

All tests use a synthetic track: Start=0, corner=15, finish=30. Blimp starts
at 16 (past the corner), and acts first clockwise in seats [0,1,2,3].
Unspecified seats are synthetic no-power racers so no unrelated ability fires.
These fixture coordinates are not an official Mild/Wild data claim.
"""

import pytest

from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Racer
from magical_athlete.track import Track


class FixedDice:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def roll(self):
        self.calls += 1
        return self.value


def blimp_engine(roll, *, gunk=True, coach=False, position=16, trips=(), shared=False):
    track = Track("H11 synthetic fixture", 30, 15, frozenset(trips))
    racers = [
        Racer(0, "Blimp", position),
        Racer(1, "Gunk" if gunk else "P2", position if shared else 2),
        Racer(2, "Coach" if coach else "P3", position if coach else 4),
        Racer(3, "P4", 5),
    ]
    return RaceEngine(track, racers, [0, 1, 2, 3], FixedDice(roll))


def take_blimp_turn(engine):
    engine.take_turn(engine.racers[0])
    return engine.turns[-1]


def assert_no_movement_events(engine):
    assert not {"move", "passing", "stop", "trip", "finish"}.intersection(e.kind for e in engine.events)


@pytest.mark.rule_vector("T-027")
def test_t027_negative_final_main_move_is_zero_with_both_effects_preserved():
    engine = blimp_engine(1)
    turn = take_blimp_turn(engine)
    assert (turn.roll, turn.raw_move, turn.final_move, turn.end_position) == (1, -1, 0, 16)
    assert [(m.source_id, m.ability, m.before, m.after) for m in turn.modifiers] == [
        (0, "Blimp", 1, 0), (1, "Gunk", 0, -1),
    ]
    assert [e.kind for e in engine.events] == ["roll", "roll_locked", "modifier", "modifier", "main_move_resolved"]
    assert_no_movement_events(engine)


@pytest.mark.rule_vector("T-028")
def test_t028_blimp_zero_is_not_a_move():
    engine = blimp_engine(1, gunk=False)
    turn = take_blimp_turn(engine)
    assert (turn.raw_move, turn.final_move, engine.racers[0].position) == (0, 0, 16)
    assert [m.ability for m in turn.modifiers] == ["Blimp"]
    assert_no_movement_events(engine)


@pytest.mark.rule_vector("T-029")
def test_t029_two_decrements_reach_zero_and_remain_recorded():
    engine = blimp_engine(2)
    turn = take_blimp_turn(engine)
    assert (turn.raw_move, turn.final_move, turn.end_position) == (0, 0, 16)
    assert [(m.before, m.after) for m in turn.modifiers] == [(2, 1), (1, 0)]
    assert_no_movement_events(engine)


@pytest.mark.rule_vector("T-030")
@pytest.mark.parametrize("position,expected_tail", [(16, ["move", "stop", "trip"]), (29, ["move", "finish"])])
def test_t030_positive_one_moves_and_resolves_landing_or_finish(position, expected_tail):
    # Positive control for zero-move suppression: real Trip/finish work normally.
    engine = blimp_engine(3, position=position, trips=(17,))
    turn = take_blimp_turn(engine)
    assert (turn.roll, turn.raw_move, turn.final_move, turn.end_position) == (3, 1, 1, position + 1)
    assert [e.kind for e in engine.events][-len(expected_tail):] == expected_tail
    assert engine.racers[0].tripped == (position == 16)
    assert engine.finish_order == ([0] if position == 29 else [])


@pytest.mark.rule_vector("T-031")
def test_t031_clamp_once_after_gunk_then_coach_not_between_modifiers():
    engine = blimp_engine(1, coach=True)
    turn = take_blimp_turn(engine)
    assert [(m.ability, m.before, m.after) for m in turn.modifiers] == [
        ("Blimp", 1, 0), ("Gunk", 0, -1), ("Coach", -1, 0),
    ]
    assert (turn.raw_move, turn.final_move, turn.end_position) == (0, 0, 16)
    assert_no_movement_events(engine)


@pytest.mark.rule_vector("T-033")
@pytest.mark.parametrize("shared,trips", [(True, ()), (False, (16,))])
def test_t033_zero_does_not_stop_again_on_shared_or_trip_space(shared, trips):
    # On the Trip fixture Blimp previously recovered here; it is upright now.
    engine = blimp_engine(1, shared=shared, trips=trips)
    turn = take_blimp_turn(engine)
    assert (turn.raw_move, turn.final_move, turn.end_position) == (-1, 0, 16)
    assert not engine.racers[0].tripped
    assert_no_movement_events(engine)
