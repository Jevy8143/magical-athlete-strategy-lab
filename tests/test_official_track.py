"""Real-engine behavior on the source-verified original-box Wild board.

Neutral racers isolate printed board effects. Coordinates use Start=0; these
are deliberately chosen landings from the transcription, not synthetic maps.
"""

import json

import pytest

from magical_athlete.engine import RaceEngine, replay_trace
from magical_athlete.racer import Racer
from magical_athlete.scoring import ScoreLedger
from magical_athlete.track import WILD_WILDS


class SequenceDice:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.values)


def wild_engine(start, rolls, *, name="P1", race_index=2):
    racers = [Racer(0, name, start), *(Racer(i, f"P{i + 1}") for i in range(1, 4))]
    dice = SequenceDice(rolls)
    ledger = ScoreLedger()
    return RaceEngine(
        WILD_WILDS, racers, [0, 1, 2, 3], dice,
        score_ledger=ledger, race_index=race_index,
    ), racers, dice, ledger


@pytest.mark.parametrize("start,arrow_space,displacement,end", [
    (6, 7, 3, 10),
    (10, 11, 1, 12),
    (15, 16, -4, 12),
    (22, 23, 2, 25),
    (23, 24, -2, 22),
])
def test_each_printed_wild_arrow_moves_its_actual_distance_and_stops_again(
    start, arrow_space, displacement, end,
):
    engine, racers, dice, ledger = wild_engine(start, [1])
    engine.take_turn(racers[0])
    moves = [event for event in engine.events if event.kind == "move"]
    assert [(event.before, event.after, event.movement_kind) for event in moves] == [
        (start, arrow_space, "main"), (arrow_space, end, "track"),
    ]
    assert moves[1].amount == displacement
    stops = [event for event in engine.events if event.kind == "stop"]
    assert [event.after for event in stops] == [arrow_space, end]
    assert stops[0].sequence < moves[1].sequence < stops[1].sequence
    assert racers[0].position == end
    assert not racers[0].tripped
    assert dice.calls == 1
    assert ledger.total(0) == 0
    trace = json.loads(json.dumps(engine.trace()))
    assert json.loads(json.dumps(replay_trace(trace).trace())) == trace


@pytest.mark.parametrize("start,roll,end,points", [
    (0, 1, 1, 1),
    (12, 1, 13, 1),
    (0, 2, 2, 0),
    (12, 2, 14, 0),
])
def test_wild_stars_score_only_actual_stops_in_the_current_race(start, roll, end, points):
    engine, racers, _, ledger = wild_engine(start, [roll])
    engine.take_turn(racers[0])
    assert racers[0].position == end
    assert ledger.total(0) == ledger.race_total(0, 2) == points
    assert ledger.race_total(0, 1) == 0
    assert ledger.race_total(0, 3) == ledger.race_total(0, 4) == 0
    scores = [event for event in engine.events if event.kind == "score"]
    assert len(scores) == points
    if points:
        assert scores[0].amount == 1 and scores[0].reason == "star"
        entry = ledger.entries[0]
        assert entry.event_sequence == scores[0].sequence
        assert entry.race_index == 2 and entry.phase == "during_race"


@pytest.mark.parametrize("trip_space", [5, 17, 26])
def test_each_wild_trip_space_skips_the_next_roll_without_retriggering(trip_space):
    engine, racers, dice, _ = wild_engine(trip_space - 1, [1, 1, 1, 1])
    engine.take_turn(racers[0])
    assert racers[0].position == trip_space and racers[0].tripped
    assert len([event for event in engine.events if event.kind == "trip" and event.racer_id == 0]) == 1
    for racer in racers[1:]:
        engine.take_turn(racer)
    calls_before = dice.calls
    events_before = len(engine.events)
    engine.take_turn(racers[0])
    assert dice.calls == calls_before == 4
    assert racers[0].position == trip_space and not racers[0].tripped
    assert engine.turns[-1].roll is None
    assert [event.kind for event in engine.events[events_before:]] == ["recover"]


@pytest.mark.parametrize("start,main_steps,end,tripped", [
    (14, 5, 19, False),
    (15, 1, 12, False),
    (16, 1, 17, True),
])
def test_blimp_switches_at_wild_second_corner_then_resolves_printed_landing_effects(
    start, main_steps, end, tripped,
):
    engine, racers, dice, _ = wild_engine(start, [2], name="Blimp")
    engine.take_turn(racers[0])
    turn = engine.turns[-1]
    assert turn.final_move == main_steps
    assert [(modifier.ability, modifier.before, modifier.after) for modifier in turn.modifiers] == [
        ("Blimp", 2, main_steps),
    ]
    assert racers[0].position == end and racers[0].tripped is tripped
    assert dice.calls == 1
    if start == 15:
        assert [(event.before, event.after) for event in engine.events
                if event.kind == "move" and event.movement_kind == "track"] == [(16, 12)]


def test_official_wild_finish_follows_space_29_and_second_place_stops_the_race():
    racers = [Racer(0, "P1", 29), Racer(1, "P2", 29), Racer(2, "P3"), Racer(3, "P4")]
    dice = SequenceDice([1, 1])
    engine = RaceEngine(WILD_WILDS, racers, [0, 1, 2, 3], dice)
    result = engine.run()
    assert result.finish_order == [0, 1]
    assert result.final_positions == {0: 30, 1: 30, 2: 0, 3: 0}
    assert result.termination_reason == "second_place"
    assert dice.calls == 2
    assert [event.kind for event in engine.events][-2:] == ["finish", "race_end"]
