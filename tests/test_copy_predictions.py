"""Real Copy Cat/inheritance/prediction acceptance cases on synthetic tracks.

Every fixture declares four physical racers, clockwise order, positions, dice
and choices. Egg cards are actual distinct remaining cards, never a card also
held on the track. Controlled-Move lifecycle tests start from a prediction
registered through a real first turn and use explicit completed action windows.
"""

import pytest

from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Ability, Racer
from magical_athlete.state import RacerStatus
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.values)


class Choices:
    def __init__(self, *, copy_target=1, prediction=2, egg=None, twin=None,
                 cheer=False, rocket=False):
        self.copy_target = copy_target
        self.prediction = prediction
        self.egg = egg
        self.twin = twin
        self.cheer = cheer
        self.rocket = rocket
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        actions = request.legal_actions
        for kind, target, value in (
            ("copy_target", self.copy_target, None),
            ("mastermind_predict", self.prediction, None),
            ("egg_choose", None, self.egg),
            ("twin_choose", None, self.twin),
        ):
            selected = [a for a in actions if a.kind == kind
                        and (target is None or a.target_id == target)
                        and (value is None or a.value == value)]
            if selected:
                return selected[0]
        if self.cheer:
            selected = [a for a in actions if a.kind == "cheer"]
            if selected:
                return selected[0]
        if self.rocket:
            selected = [a for a in actions if a.kind == "rocket_double"]
            if selected:
                return selected[0]
        return next((a for a in actions if a.kind == "keep"), actions[0])


def make_engine(racers, rolls, *, order=(0, 1, 2, 3), policies=None, **context):
    return RaceEngine(Track("copy-prediction synthetic fixture", 30, 15),
                      racers, list(order), SequenceDice(rolls), policies=policies, **context)


def advance_at_action_boundaries(engine, racer, destination):
    """Actual D6-sized Move primitives, with copy refresh after each action.

This deliberately isolates a prediction lifecycle; it does not assert that the
intermediate fixture was produced by an entire drafted random game.
"""
    while racer.position < destination:
        assert racer.status == RacerStatus.ACTIVE
        engine._move(racer, min(6, destination - racer.position))
        engine._drain_effects()
        engine._refresh_copies()


@pytest.mark.rule_vector("T-005")
def test_t005_copy_cat_can_predict_on_first_turn_while_mastermind_has_not_acted():
    racers = [Racer(0, "Copy Cat"), Racer(1, "Mastermind"), Racer(2, "P3"), Racer(3, "P4")]
    policy = Choices(copy_target=1, prediction=2)
    engine = make_engine(racers, [1], policies={0: policy})
    engine.take_turn(racers[0])
    assert racers[0].prediction == 2
    assert racers[0].turns_taken == 1
    assert racers[1].turns_taken == 0
    predictions = [r for r in policy.requests if any(a.kind == "mastermind_predict" for a in r.legal_actions)]
    assert len(predictions) == 1


@pytest.mark.rule_vector("T-006")
def test_t006_copy_cat_cannot_register_a_late_second_turn_prediction():
    racers = [Racer(0, "Copy Cat", 0), Racer(1, "Mastermind", 4), Racer(2, "Legs", 5), Racer(3, "P4", 0)]
    policy = Choices(copy_target=1, prediction=2)
    engine = make_engine(racers, [1, 2, 1, 1, 1], policies={0: policy})
    for racer in racers:
        engine.take_turn(racer)
    engine.take_turn(racers[0])
    assert racers[0].turns_taken == 2
    assert racers[0].prediction is None
    assert not any(any(a.kind == "mastermind_predict" for a in r.legal_actions) for r in policy.requests)


def prediction_lifecycle_fixture():
    racers = [Racer(0, "Copy Cat"), Racer(1, "Mastermind"), Racer(2, "Legs"), Racer(3, "P4")]
    policy = Choices(copy_target=1, prediction=2)
    engine = make_engine(racers, [1], policies={0: policy})
    engine.take_turn(racers[0])
    assert racers[0].prediction == 2
    return engine, racers


@pytest.mark.rule_vector("T-007")
def test_t007_registered_prediction_sleeps_then_recovers_before_the_winning_move():
    engine, racers = prediction_lifecycle_fixture()
    advance_at_action_boundaries(engine, racers[1], 6)
    assert engine._ability(racers[0]) == Ability.MASTERMIND
    advance_at_action_boundaries(engine, racers[2], 7)
    assert racers[0].prediction == 2
    assert engine._ability(racers[0]) != Ability.MASTERMIND
    advance_at_action_boundaries(engine, racers[1], 29)
    advance_at_action_boundaries(engine, racers[2], 28)
    assert engine._leaders() == (1,)
    assert racers[0].copied_target == 1
    assert engine._ability(racers[0]) == Ability.MASTERMIND
    engine._move(racers[2], 2)
    engine._drain_effects()
    assert engine.finish_order == [2, 0]
    assert racers[0].finish_place == 2
    assert engine.termination_reason == "second_place"


@pytest.mark.rule_vector("T-008")
def test_t008_prediction_is_ineligible_when_mastermind_does_not_lead_before_winner():
    engine, racers = prediction_lifecycle_fixture()
    advance_at_action_boundaries(engine, racers[2], 29)
    assert engine._leaders() == (2,)
    assert engine._ability(racers[0]) != Ability.MASTERMIND
    engine._move(racers[2], 1)
    engine._drain_effects()
    assert racers[0].prediction == 2
    assert engine.finish_order == [2]
    assert racers[0].status == RacerStatus.ACTIVE
    assert racers[0].finish_place is None


@pytest.mark.rule_vector("T-009")
def test_t009_copying_egg_does_not_inherit_its_selected_hare_power():
    racers = [Racer(0, "Copy Cat", 0), Racer(1, "Egg", 5), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    engine = make_engine(
        racers, [1], policies={0: Choices(copy_target=1), 1: Choices(egg="Hare")},
        remaining_deck=(Ability.HARE, Ability.LEGS, Ability.COACH),
    )
    assert racers[1].inherited_ability == Ability.HARE
    assert engine._ability(racers[1]) == Ability.HARE
    assert racers[0].copied_target == 1
    assert engine._ability(racers[0]) is None
    engine.take_turn(racers[0])
    assert racers[0].position == 1
    assert engine.turns[-1].modifiers == ()


@pytest.mark.rule_vector("T-010")
def test_t010_copying_twin_does_not_inherit_its_previous_winner_scoocher_power():
    racers = [Racer(0, "Copy Cat", 0), Racer(1, "Twin", 5), Racer(2, "Gunk", 0), Racer(3, "P4", 0)]
    engine = make_engine(
        racers, [2], policies={0: Choices(copy_target=1), 1: Choices(twin="Scoocher")},
        previous_winners=(Ability.SCOOCHER,), race_index=2,
    )
    assert racers[1].inherited_ability == Ability.SCOOCHER
    assert engine._ability(racers[0]) is None
    engine.take_turn(racers[0])
    assert racers[0].position == 1
    assert racers[1].position == 6
    assert not any(e.kind == "move" and e.racer_id == 0 and e.ability == "Scoocher" for e in engine.events)


@pytest.mark.rule_vector("T-011")
def test_t011_copy_cheerleader_completes_own_second_move_after_losing_the_copy():
    racers = [Racer(0, "Copy Cat", 4), Racer(1, "Cheerleader", 5), Racer(2, "P3", 4), Racer(3, "P4", 4)]
    engine = make_engine(racers, [1], policies={0: Choices(copy_target=1, cheer=True)})
    engine.take_turn(racers[0])
    moves = [e for e in engine.events if e.kind == "move" and e.racer_id == 0 and e.ability == "Cheerleader"]
    assert [(e.before, e.after, e.amount) for e in moves] == [(4, 6, 2), (6, 7, 1)]
    assert racers[0].position == 8
    assert engine._ability(racers[0]) != Ability.CHEERLEADER


@pytest.mark.rule_vector("T-012")
def test_t012_copy_rocket_keeps_trip_cost_after_doubled_move_changes_leader():
    racers = [Racer(0, "Copy Cat", 0), Racer(1, "Rocket Scientist", 2), Racer(2, "P3", 0), Racer(3, "P4", 0)]
    engine = make_engine(racers, [2], policies={0: Choices(copy_target=1, rocket=True)})
    engine.take_turn(racers[0])
    assert racers[0].position == 4 and racers[0].tripped
    assert engine.turns[-1].final_move == 4
    assert engine._ability(racers[0]) != Ability.ROCKET_SCIENTIST
    assert len([e for e in engine.events if e.kind == "trip" and e.racer_id == 0]) == 1


@pytest.mark.rule_vector("T-013")
def test_t013_mouth_eliminates_copy_during_first_cheer_move_and_cancels_second():
    # Copy and Baby are tied last at4; Cheer7 is the unique lead. Their +2 batch
    # brings Baby to Mouth6. Copy is first put at5; Mouth is later put at5 and
    # stops with only Copy, legitimately swallowing it before the second move.
    racers = [Racer(0, "Copy Cat", 4), Racer(1, "Cheerleader", 7), Racer(2, "Huge Baby", 4), Racer(3, "M.O.U.T.H.", 6)]
    engine = make_engine(racers, [], policies={0: Choices(copy_target=1, cheer=True)})
    engine.take_turn(racers[0])
    assert racers[0].status == RacerStatus.ELIMINATED
    assert racers[0].position == 5
    moves = [e for e in engine.events if e.kind == "move" and e.racer_id == 0]
    assert [(e.before, e.after, e.amount) for e in moves] == [(4, 6, 2)]
    assert not any(e.kind in {"roll", "reroll"} and e.racer_id == 0 for e in engine.events)
    assert engine.dice.calls == 0


@pytest.mark.rule_vector("T-014")
@pytest.mark.parametrize("order,expected_second", [((0, 1, 2, 3), 0), ((1, 0, 2, 3), 1)])
def test_t014_two_correct_predictions_take_unique_second_in_current_clockwise_order(order, expected_second):
    racers = [Racer(0, "Copy Cat"), Racer(1, "Mastermind"), Racer(2, "Banana"), Racer(3, "P4")]
    policies = {0: Choices(copy_target=1, prediction=2), 1: Choices(prediction=2)}
    engine = make_engine(racers, [1, 1, 2], order=order, policies=policies)
    engine.take_turn(racers[order[0]])
    engine.take_turn(racers[order[1]])
    assert racers[0].prediction == racers[1].prediction == 2
    advance_at_action_boundaries(engine, racers[1], 29)
    advance_at_action_boundaries(engine, racers[2], 28)
    assert engine._leaders() == (1,)
    assert engine._ability(racers[0]) == Ability.MASTERMIND
    engine.take_turn(racers[2])
    assert engine.finish_order == [2, expected_second]
    assert engine.termination_reason == "second_place"
    other = 1 - expected_second
    assert racers[other].status == RacerStatus.ACTIVE
    assert racers[other].finish_place is None
    assert engine.events[-1].kind == "race_end"


@pytest.mark.rule_vector("T-023")
def test_t023_copy_banana_and_original_banana_do_not_count_repeat_trip_twice():
    racers = [Racer(0, "Copy Cat", 2), Racer(1, "Banana", 3), Racer(2, "Scoocher", 0), Racer(3, "P4", 0)]
    engine = make_engine(racers, [4], order=(3, 0, 1, 2), policies={0: Choices(copy_target=1)})
    assert engine._ability(racers[0]) == Ability.BANANA
    engine.take_turn(racers[3])
    trips = [e for e in engine.events if e.kind == "trip" and e.racer_id == 3]
    assert len(trips) == 1 and trips[0].source_id == 0
    assert racers[3].position == 4 and racers[3].tripped
    assert racers[2].position == 1
    assert len([e for e in engine.events if e.kind == "move" and e.racer_id == 2 and e.ability == "Scoocher"]) == 1
