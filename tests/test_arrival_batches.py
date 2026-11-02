"""Real-engine arrival batches, forced pull actions and following chains.

Coordinates belong to explicit synthetic fixtures. Pull-only tests execute the
forced start-of-turn skill and its consequences in the current player's legal
start window, rather than inventing a replacement outcome for the engine.
"""

import pytest

from magical_athlete.engine import RaceEngine, replay_trace
from magical_athlete.racer import Racer
from magical_athlete.state import RacerStatus
from magical_athlete.track import Track


class SequenceDice:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = 0

    def roll(self):
        self.calls += 1
        return next(self.values)


class FollowPolicy:
    def choose(self, request):
        return next(a for a in request.legal_actions if a.kind == "suckerfish_follow")


class DuelOncePolicy:
    def __init__(self):
        self.used = False

    def choose(self, request):
        if not self.used:
            action = next(a for a in request.legal_actions if a.kind == "duel")
            self.used = True
            return action
        return next(a for a in request.legal_actions if a.kind == "keep")


def make_engine(racers, rolls=(), *, order=(0, 1, 2, 3), policies=None, finish=30):
    return RaceEngine(Track("arrival-batch synthetic fixture", finish, 15), racers,
                      list(order), SequenceDice(rolls), policies=policies)


def power_moves(engine, ability, racer_id):
    return [e for e in engine.events if e.kind == "move" and e.ability == ability and e.racer_id == racer_id]


@pytest.mark.rule_vector("T-001")
def test_t001_party_pull_coarrivals_on_an_empty_space_do_not_trigger_romantic():
    # Two racers share 2, and both move to empty 3 in the same direct pull batch.
    racers = [Racer(0, "Party Animal", 5), Racer(1, "Romantic", 2), Racer(2, "P3", 2), Racer(3, "P4", 9)]
    engine = make_engine(racers)
    engine._party_pull(racers[0])
    engine._drain_effects()
    assert (racers[1].position, racers[2].position) == (3, 3)
    assert power_moves(engine, "Romantic", 1) == []
    arrivals = [e.arrival_batch_id for e in engine.events if e.kind == "stop" and e.racer_id in (1, 2)]
    assert len(arrivals) == 2 and len(set(arrivals)) == 1


@pytest.mark.rule_vector("T-002")
def test_t002_new_mouth_relocation_precedes_resident_party_and_does_not_swallow_it():
    racers = [Racer(0, "Party Animal", 5), Racer(1, "Huge Baby", 4), Racer(2, "M.O.U.T.H.", 6), Racer(3, "P4", 10)]
    engine = make_engine(racers)
    engine._party_pull(racers[0])
    engine._drain_effects()
    assert {r.racer_id: r.position for r in racers} == {0: 4, 1: 5, 2: 4, 3: 9}
    assert racers[0].status == RacerStatus.ACTIVE
    assert not any(e.kind == "eliminate" for e in engine.events)
    mouth_stop = next(e for e in engine.events if e.kind == "stop" and e.racer_id == 2)
    party_stop = next(e for e in engine.events if e.kind == "stop" and e.racer_id == 0)
    assert mouth_stop.sequence < party_stop.sequence
    assert mouth_stop.arrival_batch_id != party_stop.arrival_batch_id


@pytest.mark.rule_vector("T-003")
def test_t003_romantic_relocation_then_resident_party_arrival_triggers_romantic():
    racers = [Racer(0, "Party Animal", 5), Racer(1, "Romantic", 4), Racer(2, "Huge Baby", 6), Racer(3, "P4", 10)]
    engine = make_engine(racers)
    engine._party_pull(racers[0])
    engine._drain_effects()
    assert (racers[0].position, racers[1].position, racers[2].position) == (4, 6, 5)
    romantic_stop = next(e for e in engine.events if e.kind == "stop" and e.racer_id == 1)
    party_stop = next(e for e in engine.events if e.kind == "stop" and e.racer_id == 0)
    romantic_move = power_moves(engine, "Romantic", 1)[0]
    assert romantic_stop.after == party_stop.after == 4
    assert romantic_stop.sequence < party_stop.sequence < romantic_move.sequence
    assert romantic_stop.arrival_batch_id != party_stop.arrival_batch_id


@pytest.mark.rule_vector("T-004")
def test_t004_huge_baby_displacing_banana_is_explicitly_tripped():
    racers = [Racer(0, "Huge Baby", 4), Racer(1, "Banana", 5), Racer(2, "P3", 0), Racer(3, "P4", 10)]
    engine = make_engine(racers, [1])
    engine.take_turn(racers[0])
    assert racers[0].position == 5 and racers[0].tripped
    assert racers[1].position == 4
    assert not any(e.kind == "move" and e.racer_id == 1 for e in engine.events)
    assert any(e.kind == "trip" and e.racer_id == 0 and e.ability == "Banana" for e in engine.events)


@pytest.mark.rule_vector("T-015")
def test_t015_romantic_second_place_cancels_scoocher_consequence():
    racers = [Racer(0, "P1", 27), Racer(1, "Romantic", 28), Racer(2, "Scoocher", 5), Racer(3, "P4", 29)]
    engine = make_engine(racers, [1, 1], order=(3, 0, 1, 2))
    engine.take_turn(racers[3])
    assert engine.finish_order == [3]
    boundary = len(engine.events)
    engine.take_turn(racers[0])
    assert engine.finish_order == [3, 1]
    assert racers[2].position == 5
    assert not any(e.kind == "move" and e.racer_id == 2 for e in engine.events[boundary:])
    assert engine.events[-1].kind == "race_end"
    assert replay_trace(engine.trace()).finish_order == [3, 1]


@pytest.mark.rule_vector("T-018")
def test_t018_suckerfish_crosses_with_hare_but_hare_finishes_first():
    racers = [Racer(0, "Hare", 27), Racer(1, "Suckerfish", 27), Racer(2, "P3", 0), Racer(3, "P4", 10)]
    engine = make_engine(racers, [1], policies={1: FollowPolicy()})
    engine.take_turn(racers[0])
    assert engine.finish_order == [0, 1]
    moves = [e for e in engine.events if e.kind == "move"]
    assert [(e.racer_id, e.after) for e in moves] == [(0, 30), (1, 30)]
    assert moves[0].arrival_batch_id == moves[1].arrival_batch_id
    assert engine.events[-1].kind == "race_end"
    assert replay_trace(engine.trace()).finish_order == [0, 1]


@pytest.mark.rule_vector("T-019")
def test_t019_scoocher_suckerfish_forward_chain_has_no_arbitrary_repeat_cutoff():
    racers = [Racer(0, "P1", 10), Racer(1, "Gunk", 20), Racer(2, "Scoocher", 0), Racer(3, "Suckerfish", 0)]
    engine = make_engine(racers, [1], policies={3: FollowPolicy()}, finish=300)
    engine.take_turn(racers[0])
    assert engine.finish_order == [2, 3]
    assert racers[0].position == 10
    assert racers[2].position == racers[3].position == 300
    assert len(power_moves(engine, "Scoocher", 2)) == 300
    assert len(power_moves(engine, "Suckerfish", 3)) == 300
    assert not any(e.kind == "loop_stopped" for e in engine.events)


@pytest.mark.rule_vector("T-020")
def test_t020_duelist_departure_does_not_erase_qualified_romantic_move():
    racers = [Racer(0, "Duelist", 5), Racer(1, "Romantic", 0), Racer(2, "Banana", 4), Racer(3, "P4", 0)]
    engine = make_engine(racers, [1, 6, 1], order=(2, 0, 1, 3), policies={0: DuelOncePolicy()})
    engine.take_turn(racers[2])
    assert (racers[0].position, racers[1].position, racers[2].position) == (7, 2, 5)
    duel_move = power_moves(engine, "Duelist", 0)[0]
    romantic_move = power_moves(engine, "Romantic", 1)[0]
    assert duel_move.sequence < romantic_move.sequence
    assert engine.dice.calls == 3


@pytest.mark.parametrize("second_position,expected_end,expected_skips", [(2, 3, 2), (1, 2, 1)])
def test_leaptoad_skips_occupied_spaces_not_number_of_racers(second_position, expected_end, expected_skips):
    racers = [Racer(0, "Leaptoad", 0), Racer(1, "P2", 1), Racer(2, "P3", second_position), Racer(3, "Scoocher", 10)]
    engine = make_engine(racers, [1])
    engine.take_turn(racers[0])
    assert racers[0].position == expected_end
    assert racers[3].position == 10 + expected_skips


@pytest.mark.parametrize("target_position,roll", [(2, 4), (1, 3)])
def test_centaur_pushes_passed_racer_back_two_without_crossing_start(target_position, roll):
    racers = [Racer(0, "Centaur", 0), Racer(1, "P2", target_position), Racer(2, "Scoocher", 10), Racer(3, "P4", 20)]
    engine = make_engine(racers, [roll])
    engine.take_turn(racers[0])
    assert racers[0].position == roll and racers[1].position == 0
    assert racers[2].position == 11
    assert len(power_moves(engine, "Centaur", 1)) == 1


def test_sisyphus_warp_does_not_pass_banana_but_can_stop_with_baba_yaga():
    racers = [Racer(0, "Sisyphus", 5), Racer(1, "Banana", 2), Racer(2, "Baba Yaga", 0), Racer(3, "P4", 20)]
    engine = make_engine(racers, [6])
    engine.take_turn(racers[0])
    assert racers[0].position == 0 and racers[0].tripped
    trips = [e for e in engine.events if e.kind == "trip" and e.racer_id == 0]
    assert [e.ability for e in trips] == ["Baba Yaga"]
    assert not any(e.kind == "passing" and e.racer_id == 0 for e in engine.events)


def test_shared_arrow_preserves_simultaneous_arrival_without_phantom_romantic_trigger():
    racers = [Racer(0, "Party Animal", 5), Racer(1, "Romantic", 2), Racer(2, "P3", 2), Racer(3, "P4", 9)]
    engine = RaceEngine(
        Track("simultaneous arrow synthetic fixture", 30, 15, arrows=((3, 1),)),
        racers, [0, 1, 2, 3], SequenceDice([]),
    )
    engine._party_pull(racers[0])
    engine._drain_effects()
    assert (racers[1].position, racers[2].position) == (4, 4)
    assert power_moves(engine, "Romantic", 1) == []
    arrow_moves = [e for e in engine.events if e.kind == "move" and e.movement_kind == "track"]
    assert len(arrow_moves) == 2
    assert arrow_moves[0].arrival_batch_id == arrow_moves[1].arrival_batch_id
