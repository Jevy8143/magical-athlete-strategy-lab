"""Real-engine H06/O06 rerolls and final-die responses on a synthetic track.

Start=0, second corner=15, finish=30 are explicit fixture coordinates. Every
test supplies four seats, initial positions and actual legal die values. The
policies select only from engine-provided legal actions; no effect is mocked.
"""

import pytest

from magical_athlete.decisions import ActionKind, DecisionWindow
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


class RerollPolicy:
    """Choose a declared source, then keep at any remaining legal window."""

    def __init__(self, *kinds):
        self.kinds = list(kinds)
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        assert request.window == DecisionWindow.REROLL
        kind = self.kinds.pop(0) if self.kinds else ActionKind.KEEP
        actions = [action for action in request.legal_actions if action.kind == kind]
        assert len(actions) == 1, f"Expected one legal {kind}, got {request.legal_actions}"
        return actions[0]


def make_engine(racers, rolls, *, order=(0, 1, 2, 3), policies=None):
    return RaceEngine(
        Track("Reroll synthetic fixture", 30, 15),
        racers,
        list(order),
        SequenceDice(rolls),
        policies=policies,
    )


def power_moves(engine, ability, racer_id):
    return [
        event for event in engine.events
        if event.kind == "move" and event.racer_id == racer_id
        and event.movement_kind == "ability" and event.ability == ability
    ]


def current_racer_id(engine):
    return engine.turn_order[engine.turn_index]


@pytest.mark.rule_vector("T-016")
def test_t016_discarded_one_does_not_trigger_skipper_or_inchworm():
    # Skipper is deliberately not the seat after Magician.
    racers = [Racer(0, "Magician", 0), Racer(1, "P2", 3), Racer(2, "Skipper", 12), Racer(3, "Inchworm", 7)]
    policy = RerollPolicy(ActionKind.MAGICIAN_REROLL)
    engine = make_engine(racers, [1, 6], policies={0: policy})
    engine.take_turn(racers[0])

    assert engine.turns[-1].roll == 6
    assert racers[0].position == 6
    assert racers[3].position == 7
    assert power_moves(engine, "Inchworm", 3) == []
    assert current_racer_id(engine) == 1
    assert [event.amount for event in engine.events if event.kind == "roll_locked"] == [6]
    assert [event.amount for event in engine.events if event.kind in {"roll", "reroll"}] == [1, 6]
    assert engine.dice.calls == 2


@pytest.mark.rule_vector("T-017")
def test_t017_magician_own_reroll_does_not_move_dicemonger():
    racers = [Racer(0, "Magician", 0), Racer(1, "Dicemonger", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    policy = RerollPolicy(ActionKind.MAGICIAN_REROLL)
    engine = make_engine(racers, [2, 4], policies={0: policy})
    engine.take_turn(racers[0])

    assert racers[0].position == 4
    assert racers[1].position == 10
    assert power_moves(engine, "Dicemonger", 1) == []
    assert [(event.before, event.after, event.amount) for event in power_moves(engine, "Scoocher", 2)] == [(5, 6, 1)]
    rerolls = [event for event in engine.events if event.kind == "reroll"]
    assert [(event.source_id, event.ability, event.amount) for event in rerolls] == [(0, "Magician", 4)]
    assert engine.turns[-1].roll == 4
    assert engine.dice.calls == 2


def test_magician_and_dicemonger_sources_have_separate_two_plus_one_limits():
    racers = [Racer(0, "Magician", 0), Racer(1, "Dicemonger", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    policy = RerollPolicy(
        ActionKind.DICEMONGER_REROLL,
        ActionKind.MAGICIAN_REROLL,
        ActionKind.MAGICIAN_REROLL,
    )
    engine = make_engine(racers, [2, 3, 4, 5], policies={0: policy})
    engine.take_turn(racers[0])

    assert [request.observation.latest_roll for request in policy.requests] == [2, 3, 4]
    assert [request.observation.magician_rerolls_remaining for request in policy.requests] == [2, 2, 1]
    assert [request.observation.dicemonger_rerolls_remaining for request in policy.requests] == [1, 0, 0]
    assert ActionKind.DICEMONGER_REROLL not in {action.kind for action in policy.requests[1].legal_actions}
    assert all(
        action.source_id == (1 if action.kind == ActionKind.DICEMONGER_REROLL else 0)
        and action.target_id == 0
        for request in policy.requests for action in request.legal_actions
        if action.kind != ActionKind.KEEP
    )
    assert [(event.ability, event.source_id, event.amount) for event in engine.events if event.kind == "reroll"] == [
        ("Dicemonger", 1, 3), ("Magician", 0, 4), ("Magician", 0, 5),
    ]
    assert engine.dice.calls == 4
    assert engine.turns[-1].roll == 5
    assert racers[0].position == 5
    assert racers[1].position == 11
    assert [(event.before, event.after) for event in power_moves(engine, "Scoocher", 2)] == [(5, 6), (6, 7), (7, 8)]


def test_dicemonger_using_own_grant_does_not_get_the_other_racer_bonus():
    racers = [Racer(0, "Dicemonger", 0), Racer(1, "P2", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 8)]
    policy = RerollPolicy(ActionKind.DICEMONGER_REROLL)
    engine = make_engine(racers, [2, 4], policies={0: policy})
    engine.take_turn(racers[0])

    assert racers[0].position == 4
    assert power_moves(engine, "Dicemonger", 0) == []
    assert [(event.before, event.after) for event in power_moves(engine, "Scoocher", 2)] == [(5, 6)]
    assert engine.dice.calls == 2


def test_dicemonger_grant_and_bonus_are_one_reroll_for_scoocher():
    racers = [Racer(0, "P1", 0), Racer(1, "Dicemonger", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    policy = RerollPolicy(ActionKind.DICEMONGER_REROLL)
    engine = make_engine(racers, [2, 4], policies={0: policy})
    engine.take_turn(racers[0])

    dicemonger_moves = power_moves(engine, "Dicemonger", 1)
    scooches = power_moves(engine, "Scoocher", 2)
    assert [(event.before, event.after, event.amount) for event in dicemonger_moves] == [(10, 11, 1)]
    assert [(event.before, event.after, event.amount) for event in scooches] == [(5, 6, 1)]
    main_move = next(event for event in engine.events if event.kind == "move" and event.movement_kind == "main")
    assert dicemonger_moves[0].sequence < main_move.sequence
    assert scooches[0].sequence < main_move.sequence
    assert racers[0].position == 4
    assert engine.dice.calls == 2


def test_final_one_cancels_main_move_moves_inchworm_and_gives_skipper_next_turn():
    racers = [Racer(0, "Magician", 0), Racer(1, "P2", 20), Racer(2, "Inchworm", 5), Racer(3, "Skipper", 10)]
    policy = RerollPolicy(ActionKind.MAGICIAN_REROLL)
    engine = make_engine(racers, [4, 1], policies={0: policy})
    engine.take_turn(racers[0])

    assert engine.turns[-1].roll == 1
    assert engine.turns[-1].final_move == 0
    assert racers[0].position == 0
    assert [(event.before, event.after, event.amount) for event in power_moves(engine, "Inchworm", 2)] == [(5, 6, 1)]
    assert not any(event.kind == "move" and event.racer_id == 0 for event in engine.events)
    assert current_racer_id(engine) == 3
    assert engine.dice.calls == 2


def test_inchworm_does_not_cancel_its_own_final_one_main_move():
    racers = [Racer(0, "Inchworm", 10), Racer(1, "P2", 0), Racer(2, "Skipper", 20), Racer(3, "P4", 5)]
    engine = make_engine(racers, [1])
    engine.take_turn(racers[0])

    assert racers[0].position == 11
    assert engine.turns[-1].final_move == 1
    assert power_moves(engine, "Inchworm", 0) == []
    assert current_racer_id(engine) == 2


def test_skipper_rolling_one_can_take_next_turn_then_order_continues_left():
    racers = [Racer(0, "Skipper", 10), Racer(1, "P2", 0), Racer(2, "P3", 2), Racer(3, "P4", 4)]
    engine = make_engine(racers, [1, 2])
    engine.take_turn(racers[0])
    assert racers[0].position == 11
    assert current_racer_id(engine) == 0

    engine.take_turn(racers[0])
    assert racers[0].position == 13
    assert current_racer_id(engine) == 1
    assert engine.dice.calls == 2


def test_skipper_takes_over_next_turn_and_then_continues_from_its_left():
    racers = [Racer(0, "P1", 0), Racer(1, "P2", 10), Racer(2, "Skipper", 20), Racer(3, "P4", 15)]
    engine = make_engine(racers, [1, 2])
    engine.take_turn(racers[0])
    assert current_racer_id(engine) == 2

    engine.take_turn(racers[2])
    assert racers[2].position == 22
    assert current_racer_id(engine) == 3


def test_reroll_response_finishing_second_prevents_more_windows_and_main_move():
    # P4 finishes first without a reroll; Magician's real reroll then scooches
    # the observer across for second, before any locked-roll/main-move action.
    racers = [Racer(0, "Magician", 0), Racer(1, "Dicemonger", 5), Racer(2, "Scoocher", 29), Racer(3, "P4", 29)]
    policy = RerollPolicy(ActionKind.MAGICIAN_REROLL)
    engine = make_engine(racers, [1, 2, 4], order=(3, 0, 1, 2), policies={0: policy})
    engine.take_turn(racers[3])
    assert engine.finish_order == [3]
    assert racers[2].position == 29
    boundary = len(engine.events)

    engine.take_turn(racers[0])
    assert engine.finish_order == [3, 2]
    assert engine.termination_reason == "second_place"
    assert racers[0].position == 0
    assert len(policy.requests) == 1
    assert not any(event.racer_id == 0 and event.kind in {"roll_locked", "main_move_resolved", "move"} for event in engine.events[boundary:])
    assert engine.events[-1].kind == "race_end"
    assert engine.dice.calls == 3


def test_inchworm_response_finishing_second_prevents_original_main_move():
    racers = [Racer(0, "P1", 0), Racer(1, "Inchworm", 29), Racer(2, "Skipper", 10), Racer(3, "P4", 29)]
    engine = make_engine(racers, [2, 1], order=(3, 0, 1, 2))
    engine.take_turn(racers[3])
    assert engine.finish_order == [3]
    engine.take_turn(racers[0])

    assert engine.finish_order == [3, 1]
    assert racers[0].position == 0
    assert engine.termination_reason == "second_place"
    assert engine.events[-1].kind == "race_end"
    assert not any(event.kind == "move" and event.racer_id == 0 for event in engine.events)
    assert engine.dice.calls == 2


def test_tripped_magician_does_not_roll_or_open_any_reroll_window():
    racers = [Racer(0, "Magician", 5, tripped=True), Racer(1, "Dicemonger", 10), Racer(2, "Scoocher", 3), Racer(3, "Skipper", 20)]
    policy = RerollPolicy(ActionKind.MAGICIAN_REROLL)
    engine = make_engine(racers, [], policies={0: policy})
    engine.take_turn(racers[0])

    assert engine.dice.calls == 0
    assert policy.requests == []
    assert racers[0].position == 5
    assert racers[0].tripped is False
    assert engine.turns[-1].roll is None
    assert not any(event.kind in {"roll", "reroll", "roll_locked", "move"} for event in engine.events)
