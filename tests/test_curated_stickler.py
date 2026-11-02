"""Stickler effectiveness on a four-seat synthetic track, not an official map."""

import json

import pytest

from magical_athlete.engine import RaceEngine, replay_trace
from magical_athlete.racer import Racer
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.values)


def make_engine(racers, rolls, order=(0, 1, 2, 3)):
    # Synthetic finish=30, corner=15. All seats and positions are explicit below.
    return RaceEngine(Track("Stickler synthetic fixture", 30, 15), racers, list(order), SequenceDice(rolls))


@pytest.mark.rule_vector("T-024")
def test_t024_exact_finish_does_not_count_as_a_stickler_effect():
    racers = [Racer(0, "P1", 27), Racer(1, "Stickler", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])
    assert engine.finish_order == [0]
    assert racers[0].position == 30
    assert racers[2].position == 5
    assert not any(e.kind in {"move_blocked", "effect_queued", "ability_effect"} for e in engine.events)


@pytest.mark.rule_vector("T-025")
def test_t025_actual_stickler_overshoot_restriction_causes_one_scooch():
    racers = [Racer(0, "P1", 27), Racer(1, "Stickler", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [4])
    engine.take_turn(racers[0])
    assert racers[0].position == 27
    assert engine.turns[-1].final_move == 4  # Computed request, subsequently blocked.
    assert engine.turns[-1].main_move_end_position == 27
    blocked = [e for e in engine.events if e.kind == "move_blocked"]
    assert [(e.racer_id, e.source_id, e.ability, e.before, e.after, e.amount) for e in blocked] == [
        (0, 1, "Stickler", 27, 27, 4),
    ]
    assert not any(e.racer_id == 0 and e.kind in {"move", "stop", "passing", "finish"} for e in engine.events)
    moves = [e for e in engine.events if e.kind == "move"]
    assert [(e.racer_id, e.amount, e.before, e.after, e.ability) for e in moves] == [(2, 1, 5, 6, "Scoocher")]
    queued = next(e for e in engine.events if e.kind == "effect_queued")
    assert queued.parent_sequence == blocked[0].sequence
    assert not engine.pending_effects


def test_stickler_does_not_limit_its_own_overshoot_or_notify_for_it():
    racers = [Racer(0, "Stickler", 27), Racer(1, "P2", 10), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [4])
    engine.take_turn(racers[0])
    assert engine.finish_order == [0]
    assert racers[2].position == 5
    assert not any(e.kind in {"move_blocked", "effect_queued"} for e in engine.events)


def test_scoocher_can_finish_with_exact_plus_one_after_its_main_move_is_blocked():
    racers = [Racer(0, "Scoocher", 29), Racer(1, "Stickler", 10), Racer(2, "P3", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [2])
    engine.take_turn(racers[0])
    assert engine.finish_order == [0]
    assert engine.turns[-1].main_move_end_position == 29
    assert engine.turns[-1].end_position == 30
    assert [(e.movement_kind, e.before, e.after, e.amount) for e in engine.events if e.kind == "move"] == [
        ("ability", 29, 30, 1),
    ]
    assert len([e for e in engine.events if e.kind == "move_blocked"]) == 1
    assert not engine.pending_effects


def test_finished_stickler_no_longer_restricts_other_racers():
    racers = [Racer(0, "Stickler", 29), Racer(1, "P2", 27), Racer(2, "Scoocher", 5), Racer(3, "P4", 0)]
    engine = make_engine(racers, [1, 5])
    engine.take_turn(racers[0])
    engine.take_turn(racers[1])
    assert engine.finish_order == [0, 1]
    assert racers[2].position == 5
    assert not any(e.kind in {"move_blocked", "effect_queued"} for e in engine.events)
    assert engine.termination_reason == "second_place"


def test_scoocher_finishing_second_cancels_its_second_queued_move_and_replays():
    racers = [Racer(0, "Blimp", 16), Racer(1, "Gunk", 2), Racer(2, "Scoocher", 28), Racer(3, "P4", 29)]
    engine = make_engine(racers, [2, 1], order=(3, 0, 1, 2))
    engine.take_turn(racers[3])
    assert engine.finish_order == [3]
    assert racers[2].position == 29
    boundary = len(engine.events)
    engine.take_turn(racers[0])
    events = engine.events[boundary:]
    assert engine.finish_order == [3, 2]
    assert racers[0].position == 16
    assert len([e for e in events if e.kind == "effect_queued"]) == 2
    assert len([e for e in events if e.kind == "ability_effect"]) == 1
    assert [(e.racer_id, e.reason) for e in events if e.kind == "effect_cancelled"] == [(2, "race_ended")]
    assert engine.events[-1].kind == "race_end"
    assert not engine.pending_effects
    trace = json.loads(json.dumps(engine.trace()))
    assert json.loads(json.dumps(replay_trace(trace).trace())) == trace


def test_scoocher_causal_events_and_turn_main_endpoint_replay_exactly():
    racers = [Racer(0, "Scoocher", 10), Racer(1, "Coach", 10), Racer(2, "Gunk", 2), Racer(3, "P4", 0)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])
    assert (engine.turns[-1].main_move_end_position, engine.turns[-1].end_position) == (13, 15)
    assert all(e.parent_sequence is None or 0 < e.parent_sequence < e.sequence for e in engine.events)
    trace = json.loads(json.dumps(engine.trace()))
    assert json.loads(json.dumps(replay_trace(trace).trace())) == trace


def test_coach_gunk_and_stickler_effects_remain_independent_when_watcher_finishes():
    racers = [Racer(0, "Scoocher", 29), Racer(1, "Coach", 29), Racer(2, "Gunk", 5), Racer(3, "Stickler", 10)]
    engine = make_engine(racers, [3])
    engine.take_turn(racers[0])
    assert engine.turns[-1].raw_move == 3
    assert engine.finish_order == [0]
    assert engine.termination_reason is None
    assert len([e for e in engine.events if e.kind == "effect_queued"]) == 3
    assert len([e for e in engine.events if e.kind == "ability_effect"]) == 1
    assert [e.reason for e in engine.events if e.kind == "effect_cancelled"] == ["racer_inactive", "racer_inactive"]
    assert not engine.pending_effects
    trace = json.loads(json.dumps(engine.trace()))
    assert json.loads(json.dumps(replay_trace(trace).trace())) == trace
