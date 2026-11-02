"""Real optional-main-move decisions on explicitly synthetic four-seat tracks.

Start=0, second corner=15, finish=30. Positions and rotated clockwise seats
are specified in each scenario; none of these fixtures claims to be a game
from drafting. Scripted policies select only engine-provided legal actions,
and fixed dice provide inputs rather than replacing rule transitions.
"""

import pytest

from magical_athlete.decisions import Action, ActionKind, DecisionWindow
from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Racer
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, rolls):
        self.rolls = iter(rolls)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.rolls)


class SelectMainMove:
    def __init__(self, kind):
        self.kind = kind
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        assert request.window == DecisionWindow.MAIN_MOVE_MODIFIER
        assert request.legal_actions[0].kind == ActionKind.KEEP
        return next(action for action in request.legal_actions if action.kind == self.kind)


def make_engine(racers, rolls, *, order=(0, 1, 2, 3), policies=None):
    return RaceEngine(
        Track("Optional-move synthetic fixture", 30, 15),
        racers,
        list(order),
        SequenceDice(rolls),
        policies=policies,
    )


@pytest.mark.rule_vector("T-021")
@pytest.mark.parametrize("offset", [0, 2])
def test_t021_alchemist_replacement_remains_main_move_and_receives_coach_and_gunk(offset):
    actor, coach, gunk, ordinary = [(offset + index) % 4 for index in range(4)]
    racers = sorted([
        Racer(actor, "Alchemist", 10), Racer(coach, "Coach", 10),
        Racer(gunk, "Gunk", 2), Racer(ordinary, "P4", 0),
    ], key=lambda racer: racer.racer_id)
    policy = SelectMainMove(ActionKind.ALCHEMIST_FOUR)
    engine = make_engine(racers, [1], order=(actor, coach, gunk, ordinary), policies={actor: policy})
    engine.take_turn(engine.racers[actor])

    turn = engine.turns[-1]
    assert (turn.roll, turn.raw_move, turn.final_move, turn.main_move_end_position) == (1, 4, 4, 14)
    assert [(modifier.source_id, modifier.before, modifier.after) for modifier in turn.modifiers] == [
        (actor, 1, 4), (coach, 4, 5), (gunk, 5, 4),
    ]
    assert [(event.racer_id, event.amount, event.movement_kind) for event in engine.events if event.kind == "move"] == [
        (actor, 4, "main"),
    ]
    assert len(policy.requests) == 1
    request = policy.requests[0]
    assert request.player_id == actor
    assert request.observation.latest_roll == 1
    assert Action(ActionKind.ALCHEMIST_FOUR, source_id=actor, target_id=actor) in request.legal_actions


@pytest.mark.rule_vector("T-022")
@pytest.mark.parametrize("offset", [0, 2])
def test_t022_rocket_double_precedes_other_gunk_and_keeps_main_move_trip_cost(offset):
    actor, gunk, ordinary1, ordinary2 = [(offset + index) % 4 for index in range(4)]
    racers = sorted([
        Racer(actor, "Rocket Scientist", 10), Racer(gunk, "Gunk", 2),
        Racer(ordinary1, "P3", 0), Racer(ordinary2, "P4", 0),
    ], key=lambda racer: racer.racer_id)
    policy = SelectMainMove(ActionKind.ROCKET_DOUBLE)
    engine = make_engine(racers, [3], order=(actor, gunk, ordinary1, ordinary2), policies={actor: policy})
    engine.take_turn(engine.racers[actor])

    turn = engine.turns[-1]
    assert (turn.roll, turn.raw_move, turn.final_move, turn.main_move_end_position) == (3, 5, 5, 15)
    assert [(modifier.source_id, modifier.before, modifier.after) for modifier in turn.modifiers] == [
        (actor, 3, 6), (gunk, 6, 5),
    ]
    assert engine.racers[actor].tripped is True
    assert len([event for event in engine.events if event.kind == "trip" and event.racer_id == actor]) == 1
    assert len(policy.requests) == 1
    assert policy.requests[0].observation.latest_roll == 3
    assert Action(ActionKind.ROCKET_DOUBLE, source_id=actor, target_id=actor) in policy.requests[0].legal_actions


@pytest.mark.parametrize("name,roll", [("Alchemist", 2), ("Rocket Scientist", 3)])
def test_optional_main_move_ability_can_be_declined(name, roll):
    racers = [Racer(0, name, 10), Racer(1, "P2", 2), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    policy = SelectMainMove(ActionKind.KEEP)
    engine = make_engine(racers, [roll], policies={0: policy})
    engine.take_turn(racers[0])
    assert racers[0].position == 10 + roll
    assert racers[0].tripped is False
    assert engine.turns[-1].modifiers == ()
    assert len(policy.requests) == 1
    assert not any(event.kind == "trip" for event in engine.events)


@pytest.mark.parametrize("roll", [3, 4, 5, 6])
def test_alchemist_has_no_replacement_window_for_ineligible_final_roll(roll):
    racers = [Racer(0, "Alchemist", 10), Racer(1, "P2", 2), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    policy = SelectMainMove(ActionKind.KEEP)
    engine = make_engine(racers, [roll], policies={0: policy})
    engine.take_turn(racers[0])
    assert policy.requests == []
    assert racers[0].position == 10 + roll
    assert engine.turns[-1].modifiers == ()


def test_rocket_still_trips_when_stickler_blocks_the_doubled_main_move():
    racers = [Racer(0, "Rocket Scientist", 29), Racer(1, "Stickler", 5), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    policy = SelectMainMove(ActionKind.ROCKET_DOUBLE)
    engine = make_engine(racers, [2], policies={0: policy})
    engine.take_turn(racers[0])
    assert (engine.turns[-1].final_move, engine.turns[-1].main_move_end_position) == (4, 29)
    assert racers[0].tripped is True
    assert engine.finish_order == []
    assert not any(event.kind == "move" and event.racer_id == 0 for event in engine.events)
    blocked = next(event for event in engine.events if event.kind == "move_blocked")
    trip = next(event for event in engine.events if event.kind == "trip" and event.racer_id == 0)
    assert blocked.sequence < trip.sequence


def test_rocket_double_and_trip_are_one_effect_for_scoocher():
    racers = [Racer(0, "Rocket Scientist", 10), Racer(1, "Scoocher", 3), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    policy = SelectMainMove(ActionKind.ROCKET_DOUBLE)
    engine = make_engine(racers, [2], policies={0: policy})
    engine.take_turn(racers[0])
    assert racers[0].position == 14
    assert racers[0].tripped is True
    assert racers[1].position == 4
    moves = [event for event in engine.events if event.kind == "move" and event.racer_id == 1]
    assert [(event.amount, event.before, event.after, event.ability) for event in moves] == [(1, 3, 4, "Scoocher")]
    assert len([event for event in engine.events if event.kind == "effect_queued" and event.racer_id == 1]) == 1


def test_rocket_recovers_next_main_move_without_a_roll_or_new_choice():
    racers = [Racer(0, "Rocket Scientist", 10), Racer(1, "P2", 0), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    policy = SelectMainMove(ActionKind.ROCKET_DOUBLE)
    engine = make_engine(racers, [2, 1, 1, 1], policies={0: policy})
    engine.take_turn(racers[0])
    assert racers[0].tripped is True
    for racer in racers[1:]:
        engine.take_turn(racer)
    boundary = len(engine.events)
    engine.take_turn(racers[0])
    assert (racers[0].position, racers[0].tripped, engine.dice.calls) == (14, False, 4)
    assert engine.turns[-1].roll is None
    assert engine.turns[-1].modifiers == ()
    assert [event.kind for event in engine.events[boundary:]] == ["recover"]
    assert len(policy.requests) == 1
