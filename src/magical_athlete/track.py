from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    name: str
    finish_position: int
    second_corner_position: int | None = None
    trip_positions: frozenset[int] = frozenset()
    source: str = "Synthetic fixture; not an official Mild/Wild transcription"
    first_corner_position: int | None = None
    data_version: str = "synthetic-v1"
    star_positions: frozenset[int] = frozenset()
    arrows: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        trips = tuple(self.trip_positions)
        stars = tuple(self.star_positions)
        arrows = tuple(tuple(arrow) for arrow in self.arrows)
        if any(len(arrow) != 2 for arrow in arrows):
            raise ValueError("each arrow requires a position and displacement")
        # Python compares 3.0 == 3 and True == 1. Equality with canonical data
        # must not admit inputs that later fail in integer movement operations.
        coordinates = [self.finish_position, *trips, *stars,
                       *(value for arrow in arrows for value in arrow)]
        coordinates += [corner for corner in (self.first_corner_position, self.second_corner_position)
                        if corner is not None]
        if any(type(value) is not int for value in coordinates):
            raise ValueError("track coordinates and arrow displacements must be integers")
        object.__setattr__(self, "trip_positions", frozenset(trips))
        object.__setattr__(self, "star_positions", frozenset(stars))
        object.__setattr__(self, "arrows", arrows)
        if self.finish_position <= 0:
            raise ValueError("finish_position must be positive")
        if self.second_corner_position is not None and not (
            0 < self.second_corner_position < self.finish_position
        ):
            raise ValueError("second corner must be before the finish")
        if self.first_corner_position is not None and not (
            0 < self.first_corner_position < self.finish_position
        ):
            raise ValueError("first corner must be before the finish")
        if (
            self.first_corner_position is not None
            and self.second_corner_position is not None
            and self.first_corner_position >= self.second_corner_position
        ):
            raise ValueError("first corner must precede the second corner")
        if any(not 0 <= p < self.finish_position for p in self.trip_positions):
            raise ValueError("trip spaces must be on the track")
        if any(not 0 <= p < self.finish_position for p in self.star_positions):
            raise ValueError("star spaces must be on the track")
        if len({p for p, _ in self.arrows}) != len(self.arrows):
            raise ValueError("each space may have at most one arrow")
        if any(not 0 <= p < self.finish_position or steps == 0 for p, steps in self.arrows):
            raise ValueError("arrows require a track space and nonzero displacement")

    def move_forward(self, position: int, steps: int) -> int:
        if steps < 0:
            raise ValueError("forward movement cannot be negative; apply H11 first")
        new_position = position + steps

        return min(new_position, self.finish_position)

    def has_finished(self, position: int) -> bool:
        return position >= self.finish_position


SYNTHETIC_TRACK = Track(
    name="Synthetic 30-space track",
    finish_position=30,
)


# Counted from the complete two-page official illustration. Start is zero;
# the printed 15 is the second corner, and the finish follows ordinary space 29.
# No special spaces appear on this side of the original-box board.
MILD_MILE = Track(
    name="Mild Mile",
    finish_position=30,
    second_corner_position=15,
    trip_positions=frozenset(),
    source="docs/CMYK_MA_Rulebook_081426.pdf p10-11; complete official Mild Mile illustration",
    first_corner_position=12,
    data_version="2026-08-rulebook-mild-p10-p11-v1",
)


# Complete original-box board image, cross-checked against the official
# rulebook's Wild illustration. Bottom-row forward movement is right to left;
# the arrow glyph's screen direction must not be mistaken for a positive sign.
WILD_WILDS = Track(
    name="Wild Wilds",
    finish_position=30,
    first_corner_position=12,
    second_corner_position=15,
    star_positions=frozenset({1, 13}),
    trip_positions=frozenset({5, 17, 26}),
    arrows=((7, 3), (11, 1), (16, -4), (23, 2), (24, -2)),
    source=(
        "Original-box board image at https://thirstymeeples.co.uk/products/magical-athlete; "
        "cross-check docs/CMYK_MA_Rulebook_081426.pdf p20-21; "
        "see docs/06_正式赛道与筹码数据.md"
    ),
    data_version="2026-10-29-original-box-wild-v1",
)
