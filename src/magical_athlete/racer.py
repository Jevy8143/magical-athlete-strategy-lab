from dataclasses import dataclass
from enum import StrEnum

from .state import RacerStatus


class Ability(StrEnum):
    ALCHEMIST = "Alchemist"
    BABA_YAGA = "Baba Yaga"
    BANANA = "Banana"
    BLIMP = "Blimp"
    CENTAUR = "Centaur"
    CHEERLEADER = "Cheerleader"
    COACH = "Coach"
    COPY_CAT = "Copy Cat"
    DICEMONGER = "Dicemonger"
    DUELIST = "Duelist"
    EGG = "Egg"
    FLIP_FLOP = "Flip Flop"
    GENIUS = "Genius"
    GUNK = "Gunk"
    HARE = "Hare"
    HECKLER = "Heckler"
    HUGE_BABY = "Huge Baby"
    HYPNOTIST = "Hypnotist"
    INCHWORM = "Inchworm"
    LACKEY = "Lackey"
    LEAPTOAD = "Leaptoad"
    LEGS = "Legs"
    LOVEABLE_LOSER = "Loveable Loser"
    MAGICIAN = "Magician"
    MASTERMIND = "Mastermind"
    MOUTH = "M.O.U.T.H."
    PARTY_ANIMAL = "Party Animal"
    ROCKET_SCIENTIST = "Rocket Scientist"
    ROMANTIC = "Romantic"
    SCOOCHER = "Scoocher"
    SISYPHUS = "Sisyphus"
    SKIPPER = "Skipper"
    STICKLER = "Stickler"
    SUCKERFISH = "Suckerfish"
    THIRD_WHEEL = "Third Wheel"
    TWIN = "Twin"


ROLE_DATA_VERSION = "2026-08-rulebook-original-36-v1"
ABILITY_SOURCES = {
    ability: f"docs/CMYK_MA_Rulebook_081426.pdf p{page}"
    for page, abilities in {
        24: (Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
             Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.COPY_CAT, Ability.DICEMONGER),
        25: (Ability.DUELIST, Ability.EGG, Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK,
             Ability.HARE, Ability.HECKLER, Ability.HUGE_BABY, Ability.HYPNOTIST),
        26: (Ability.INCHWORM, Ability.LACKEY, Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER,
             Ability.MAGICIAN, Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL),
        27: (Ability.ROMANTIC, Ability.ROCKET_SCIENTIST, Ability.SCOOCHER, Ability.SISYPHUS,
             Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL, Ability.TWIN),
    }.items() for ability in abilities
}


@dataclass
class Racer:
    racer_id: int
    name: str
    position: int = 0
    status: RacerStatus = RacerStatus.ACTIVE
    finish_place: int | None = None
    ability: Ability | None = None
    tripped: bool = False
    turns_taken: int = 0
    inherited_ability: Ability | None = None
    copied_target: int | None = None
    copied_ability: Ability | None = None
    prediction: int | None = None
    finish_places: tuple[int, ...] = ()
    pre_race_resolved: bool = False

    def __post_init__(self) -> None:
        # The name is also accepted as a character identifier for old callers.
        # Arbitrary labels such as P1 denote explicitly synthetic, no-power racers.
        if self.ability is not None:
            self.ability = Ability(self.ability)
        elif self.name in Ability._value2member_map_:
            self.ability = Ability(self.name)
        for field in ("inherited_ability", "copied_ability"):
            value = getattr(self, field)
            if value is not None:
                setattr(self, field, Ability(value))
        self.finish_places = tuple(self.finish_places)
