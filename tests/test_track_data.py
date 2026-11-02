"""Official Mild data transcribed from the rulebook's complete p10-11 spread.

Top row: spaces 1-12 (first corner); right side: 13-14; second corner: 15;
bottom row: 16-29; finish: 30. Start is 0. All ordinary spaces are unmarked.
The source does not supply an unobscured complete Wild track or all award chips.
"""

import pytest

from magical_athlete.track import MILD_MILE, SYNTHETIC_TRACK, Track


def test_official_mild_visits_all_ordinary_spaces_before_the_finish():
    position = 0
    visited = [position]
    for _ in range(30):
        assert not MILD_MILE.has_finished(position)
        position = MILD_MILE.move_forward(position, 1)
        visited.append(position)
    assert visited == list(range(31))
    assert MILD_MILE.has_finished(position)
    assert not MILD_MILE.has_finished(29)
    assert MILD_MILE.move_forward(29, 6) == 30


def test_official_mild_corners_and_plain_spaces_have_explicit_provenance():
    assert MILD_MILE.first_corner_position == 12
    assert MILD_MILE.second_corner_position == 15
    assert MILD_MILE.move_forward(11, 1) == MILD_MILE.first_corner_position
    assert MILD_MILE.move_forward(14, 1) == MILD_MILE.second_corner_position
    assert MILD_MILE.trip_positions == frozenset()
    assert "docs/CMYK_MA_Rulebook_081426.pdf p10-11" in MILD_MILE.source
    assert MILD_MILE.data_version == "2026-08-rulebook-mild-p10-p11-v1"
    assert MILD_MILE is not SYNTHETIC_TRACK
    assert SYNTHETIC_TRACK.data_version == "synthetic-v1"


@pytest.mark.parametrize("corners", [
    {"first_corner_position": 0},
    {"first_corner_position": 30},
    {"second_corner_position": 0},
    {"second_corner_position": 30},
    {"first_corner_position": 15, "second_corner_position": 15},
    {"first_corner_position": 16, "second_corner_position": 15},
])
def test_track_rejects_corners_outside_the_track_or_in_reverse_order(corners):
    with pytest.raises(ValueError, match="corner"):
        Track("invalid corner fixture", 30, **corners)


def test_existing_positional_fixture_arguments_keep_their_meaning():
    fixture = Track("synthetic positional compatibility", 30, 15, frozenset({17}))
    assert fixture.second_corner_position == 15
    assert fixture.trip_positions == frozenset({17})
    assert fixture.first_corner_position is None
