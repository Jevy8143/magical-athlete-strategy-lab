from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Racer
from magical_athlete.track import SYNTHETIC_TRACK


class AlwaysSixDice:

    def roll(self):
        return 6


def test_four_racer_race():

    racers = [
        Racer(0, "P1"),
        Racer(1, "P2"),
        Racer(2, "P3"),
        Racer(3, "P4"),
    ]

    engine = RaceEngine(
        track=SYNTHETIC_TRACK,
        racers=racers,
        turn_order=[0, 1, 2, 3],
        dice=AlwaysSixDice(),
    )

    result = engine.run()

    assert result.finish_order == [0, 1]

    assert result.final_positions == {
        0: 30,
        1: 30,
        2: 24,
        3: 24,
    }
