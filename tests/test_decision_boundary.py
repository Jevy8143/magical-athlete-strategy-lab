"""Public decision boundaries against the real incremental race engine.

All scenarios use four explicit seats on a synthetic track. These tests check
the information passed through the policy interface, not isolation from
malicious Python code sharing the engine's process.
"""

import copy
import json
import random
from dataclasses import FrozenInstanceError, asdict

import pytest

from magical_athlete.decisions import (
    Action,
    ActionKind,
    DecisionWindow,
    KeepPolicy,
    PublicRacer,
    RandomPolicy,
)
from magical_athlete.engine import D6, RaceEngine, replay_trace
from magical_athlete.racer import Racer
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.values)


class RecordingPolicy:
    policy_id = "test-recording-v1"

    def __init__(self, choice=ActionKind.KEEP):
        self.choice = choice
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        return next(action for action in request.legal_actions if action.kind == self.choice)


def make_engine(*, dice=None, policies=None, character="Alchemist", position=10, tripped=False):
    racers = [
        Racer(0, character, position, tripped=tripped),
        Racer(1, "P2", 2),
        Racer(2, "P3", 3),
        Racer(3, "P4", 4),
    ]
    return RaceEngine(
        Track("public-decision synthetic fixture", 30, 15),
        racers,
        [0, 1, 2, 3],
        dice if dice is not None else SequenceDice([1]),
        policies=policies,
    )


def normalized(trace):
    return json.loads(json.dumps(trace))


def test_policy_receives_current_public_state_and_only_legal_actions():
    policy = RecordingPolicy(ActionKind.ALCHEMIST_FOUR)
    engine = make_engine(policies={0: policy})
    engine.take_turn(engine.racers[0])

    assert len(policy.requests) == 1
    request = policy.requests[0]
    assert request.player_id == request.observation.current_player_id == 0
    assert request.window == DecisionWindow.MAIN_MOVE_MODIFIER
    assert request.observation.latest_roll == 1
    assert request.observation.current_move_value == 1
    assert request.legal_actions == (
        Action(ActionKind.KEEP),
        Action(ActionKind.ALCHEMIST_FOUR, 0, 0),
    )
    assert [(r.racer_id, r.position) for r in request.observation.racers] == [(0, 10), (1, 2), (2, 3), (3, 4)]
    assert all(isinstance(r, PublicRacer) for r in request.observation.racers)
    assert engine.racers[0].position == 14
    assert engine.dice.calls == 1


@pytest.mark.parametrize("action", [
    Action(ActionKind.ALCHEMIST_FOUR, 1, 0),
    Action(ActionKind.ALCHEMIST_FOUR, 0, 1),
    Action(ActionKind.ROCKET_DOUBLE, 0, 0),
    Action(ActionKind.MAGICIAN_REROLL, 0, 0),
    "alchemist_four",
])
def test_engine_rejects_actions_outside_this_window_without_applying_them(action):
    class InvalidPolicy:
        policy_id = "test-invalid-v1"

        def choose(self, request):
            return action

    engine = make_engine(policies={0: InvalidPolicy()})
    with pytest.raises(ValueError):
        engine.take_turn(engine.racers[0])
    assert engine.racers[0].position == 10
    assert not any(e.kind in {"modifier", "move", "stop", "trip", "finish"} for e in engine.events)
    assert engine.dice.calls == 1


def test_policy_snapshot_is_frozen_and_cannot_change_engine_positions():
    class MutatingPolicy(RecordingPolicy):
        def choose(self, request):
            with pytest.raises(FrozenInstanceError):
                request.player_id = 3
            with pytest.raises(FrozenInstanceError):
                request.observation.latest_roll = 6
            with pytest.raises(FrozenInstanceError):
                request.observation.racers[0].position = 29
            with pytest.raises(AttributeError):
                request.observation.racers.append(request.observation.racers[0])
            return super().choose(request)

    policy = MutatingPolicy()
    engine = make_engine(policies={0: policy})
    engine.take_turn(engine.racers[0])
    assert engine.racers[0].position == 11
    assert policy.requests[0].observation.latest_roll == 1


def test_saved_snapshot_does_not_follow_later_engine_changes():
    policy = RecordingPolicy()
    engine = make_engine(dice=SequenceDice([1, 2]), policies={0: policy})
    engine.take_turn(engine.racers[0])
    snapshot = policy.requests[0].observation
    before = asdict(snapshot)
    engine.take_turn(engine.racers[1])
    assert (engine.racers[0].position, engine.racers[1].position) == (11, 4)
    assert asdict(snapshot) == before
    assert [(r.racer_id, r.position) for r in snapshot.racers][:2] == [(0, 10), (1, 2)]
    assert all(public is not engine.racers[public.racer_id] for public in snapshot.racers)


def test_request_contains_no_engine_rng_or_unrevealed_dice_state():
    policy = RecordingPolicy()
    engine = make_engine(dice=SequenceDice([1, 6, 5]), policies={0: policy})
    engine.take_turn(engine.racers[0])
    payload = asdict(policy.requests[0])
    forbidden_names = {
        "seed", "random", "rng", "dice", "engine", "future_rolls",
        "_random", "random_state", "deck_order", "unrevealed_choices",
    }

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden_names.intersection(value)
            for child in value.values():
                inspect(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                inspect(child)
        else:
            assert value is None or isinstance(value, (str, int, bool))

    inspect(payload)
    assert engine.dice.calls == 1
    assert payload["observation"]["latest_roll"] == 1
    assert not hasattr(policy.requests[0].observation, "__dict__")


def test_same_public_dice_prefix_produces_same_request_despite_different_futures():
    policies = [RecordingPolicy(), RecordingPolicy()]
    engines = [
        make_engine(dice=SequenceDice([1, 6, 6]), policies={0: policies[0]}),
        make_engine(dice=SequenceDice([1, 2, 3]), policies={0: policies[1]}),
    ]
    for engine in engines:
        engine.take_turn(engine.racers[0])
    assert policies[0].requests == policies[1].requests
    assert [engine.dice.calls for engine in engines] == [1, 1]


def test_random_policy_complete_run_is_reproducible_and_keeps_global_rng_independent():
    global_state = random.getstate()
    first = make_engine(dice=D6(2), policies={0: RandomPolicy(73)})
    first.run()
    second = make_engine(dice=D6(2), policies={0: RandomPolicy(73)})
    second.run()
    assert random.getstate() == global_state
    assert normalized(first.trace()) == normalized(second.trace())
    records = first.trace()["decisions"]
    assert records
    for record in records:
        assert record["action"] in record["request"]["legal_actions"]


def test_policy_randomness_cannot_change_the_independent_dice_stream():
    engines = [
        make_engine(dice=D6(2), policies={0: RandomPolicy(1)}),
        make_engine(dice=D6(2), policies={0: RandomPolicy(999)}),
    ]
    for engine in engines:
        for racer_id in range(4):
            engine.take_turn(engine.racers[racer_id])
    rolls = [[e.amount for e in engine.events if e.kind == "roll"] for engine in engines]
    assert rolls[0] == rolls[1]
    assert len(rolls[0]) == 4


def test_default_keep_policy_preserves_optional_alchemist_roll():
    engine = make_engine()
    engine.take_turn(engine.racers[0])
    assert engine.racers[0].position == 11
    assert engine.turns[-1].modifiers == ()
    record = engine.trace()["decisions"][0]
    assert record["action"] == asdict(Action(ActionKind.KEEP))


@pytest.mark.parametrize("character,roll,tripped", [("Alchemist", 3, False), ("Magician", 1, True)])
def test_policy_is_not_called_without_a_legal_optional_window(character, roll, tripped):
    class MustNotChoose:
        policy_id = "test-no-window-v1"

        def choose(self, request):
            raise AssertionError("policy called outside a legal decision window")

    dice = SequenceDice([roll])
    engine = make_engine(dice=dice, policies={0: MustNotChoose()}, character=character, tripped=tripped)
    engine.take_turn(engine.racers[0])
    assert engine.trace()["decisions"] == []
    assert dice.calls == (0 if tripped else 1)


def test_replay_uses_recorded_decision_without_calling_original_policy_again():
    class OneShotPolicy(RecordingPolicy):
        policy_id = "test-one-shot-v1"

        def choose(self, request):
            assert not self.requests, "original policy must not be used during replay"
            return super().choose(request)

    policy = OneShotPolicy(ActionKind.ALCHEMIST_FOUR)
    engine = make_engine(policies={0: policy})
    engine.take_turn(engine.racers[0])
    trace = normalized(engine.trace())
    assert trace["decisions"]
    assert trace["policies"]
    assert trace["decisions"][0]["action"]["kind"] == "alchemist_four"
    replayed = replay_trace(trace)
    assert normalized(replayed.trace()) == trace
    assert len(policy.requests) == 1


def test_replay_rejects_an_action_with_a_forged_source_or_a_forged_observation():
    policy = RecordingPolicy(ActionKind.ALCHEMIST_FOUR)
    engine = make_engine(policies={0: policy})
    engine.take_turn(engine.racers[0])
    trace = normalized(engine.trace())
    wrong_source = copy.deepcopy(trace)
    wrong_source["decisions"][0]["action"]["source_id"] = 1
    with pytest.raises(ValueError):
        replay_trace(wrong_source)
    wrong_observation = copy.deepcopy(trace)
    wrong_observation["decisions"][0]["request"]["observation"]["latest_roll"] = 6
    with pytest.raises(ValueError):
        replay_trace(wrong_observation)


def test_direct_in_memory_trace_replays_decisions_without_a_json_round_trip():
    policy = RecordingPolicy(ActionKind.ALCHEMIST_FOUR)
    engine = make_engine(policies={0: policy})
    engine.take_turn(engine.racers[0])
    trace = engine.trace()
    assert trace["decisions"][0]["action"]["kind"] == "alchemist_four"
    replayed = replay_trace(trace)
    assert normalized(replayed.trace()) == normalized(trace)
    assert len(policy.requests) == 1
