"""Original 36-card, source-verified two-board/four-race acceptance tests."""

import json
import random
from dataclasses import asdict, replace

import pytest

from magical_athlete.decisions import RandomPolicy
from magical_athlete.engine import replay_trace
from magical_athlete.game import (
    BASE_DECK, GameConfig, GameEngine, RandomGamePolicy, official_config, replay_game_trace,
)
from magical_athlete.scoring import ORIGINAL_BOX_AWARDS
from magical_athlete.track import MILD_MILE, WILD_WILDS, Track


def make_official_game(seed=1):
    return GameEngine(
        official_config(), seed,
        game_policies={player: RandomGamePolicy(100 + player) for player in range(4)},
        race_policies={player: RandomPolicy(200 + player) for player in range(4)},
    )


@pytest.fixture(scope="module")
def official_game():
    game = make_official_game()
    game.run()
    return game


def test_original_box_runs_from_full_draft_through_four_races_with_real_chip_payments(official_game):
    game = official_game
    result = game.run()
    assert result.official and game.trace()["run_kind"] == "official_game"
    assert len(game.config.deck) == 36 and game.config.deck == BASE_DECK
    assert len(game.draft_history) == 16
    assert [race.track_kind for race in result.races] == ["Mild", "Wild", "Mild", "Wild"]
    assert [trace["track"]["data_version"] for trace in game.race_traces] == [
        MILD_MILE.data_version, WILD_WILDS.data_version,
        MILD_MILE.data_version, WILD_WILDS.data_version,
    ]
    entries = [card.card_id for race in result.races for card in race.entries]
    assert len(entries) == len(set(entries)) == 16
    assert all(not hand for hand in game.hands.values())
    assert [len(cards) for cards in game.used_cards.values()] == [4, 4, 4, 4]
    located = game.remaining_deck + game.pool + list(game.active_entries) + game.ability_cards_out
    located += [card for cards in game.used_cards.values() for card in cards]
    assert len(located) == len({card.card_id for card in located}) == 36
    assert {card.card_id for card in located} == {card.card_id for card in BASE_DECK}
    for race, expected_awards in zip(result.races, ((3, 1), (4, 2), (4, 2), (5, 3))):
        placements = [entry for entry in game.ledger.entries
                      if entry.race_index == race.race_index and entry.reason.startswith("placement_")]
        assert [(entry.player_id, entry.delta, entry.phase) for entry in placements] == [
            (player, expected_awards[place], "after_race")
            for place, player in enumerate(race.finish_order)
        ]
        assert race.race_scores == tuple(game.ledger.race_total(player, race.race_index) for player in range(4))
    assert result.scores == tuple(game.ledger.total(player) for player in range(4))
    assert sum(result.scores) == sum(entry.delta for entry in game.ledger.entries)
    assert result.winners == tuple(player for player, score in enumerate(result.scores) if score == max(result.scores))


def test_official_game_and_all_four_races_replay_actual_recorded_decisions(official_game):
    trace = json.loads(json.dumps(official_game.trace()))
    assert replay_game_trace(trace).trace() == trace
    for race_trace in trace["race_traces"]:
        assert json.loads(json.dumps(replay_trace(race_trace).trace())) == race_trace
    # Each independent race ends before the game pays its placement chips.
    assert not any(entry["reason"].startswith("placement_")
                   for entry in trace["race_traces"][0]["score_ledger"]["entries"])


def test_official_game_repeats_with_fresh_seeded_policies_without_global_random_changes(official_game):
    before = random.getstate()
    second = make_official_game()
    second.run()
    assert random.getstate() == before
    assert second.trace() == official_game.trace()


@pytest.mark.parametrize("field,value", [
    ("finish_position", 31),
    ("first_corner_position", 11),
    ("second_corner_position", 14),
    ("trip_positions", frozenset({5, 17})),
    ("star_positions", frozenset({1, 13, 18})),
    ("arrows", ((7, 3), (11, 1), (16, 4), (23, 2), (24, -2))),
])
def test_modified_wild_cannot_keep_official_status_using_the_same_source_labels(field, value):
    config = official_config()
    altered_track = replace(WILD_WILDS, **{field: value})
    assert altered_track.source == WILD_WILDS.source
    assert altered_track.data_version == WILD_WILDS.data_version
    with pytest.raises(ValueError, match="official Wild"):
        replace(config, wild_track=altered_track)


@pytest.mark.parametrize("chip_index", range(8))
def test_each_changed_chip_value_is_rejected_even_with_verified_and_source_retained(chip_index):
    awards = [list(pair) for pair in ORIGINAL_BOX_AWARDS.awards]
    awards[chip_index // 2][chip_index % 2] += 1
    altered_awards = replace(ORIGINAL_BOX_AWARDS, awards=awards)
    assert altered_awards.verified and altered_awards.source == ORIGINAL_BOX_AWARDS.source
    with pytest.raises(ValueError, match="official award"):
        replace(official_config(), award_schedule=altered_awards)


def test_canonical_official_data_survives_json_reconstruction_without_identity_checks():
    data = json.loads(json.dumps(asdict(official_config()), default=list))
    rebuilt = GameConfig(
        Track(**data["mild_track"]), Track(**data["wild_track"]),
        replace(ORIGINAL_BOX_AWARDS, awards=data["award_schedule"]["awards"]),
        official=True, source=data["source"],
    )
    assert rebuilt.official
    assert rebuilt.track_for_race(2) == rebuilt.track_for_race(4) == WILD_WILDS


@pytest.mark.parametrize("field,value", [
    ("finish_position", 30.0),
    ("first_corner_position", 12.0),
    ("second_corner_position", 15.0),
    ("trip_positions", [5.0, 17, 26]),
    ("star_positions", [True, 13]),
    ("arrows", ((7, 3.0), (11, 1), (16, -4), (23, 2), (24, -2))),
    ("arrows", ((7.0, 3), (11, 1), (16, -4), (23, 2), (24, -2))),
    ("arrows", ((7, 3), (11, True), (16, -4), (23, 2), (24, -2))),
])
def test_numeric_equality_cannot_smuggle_noninteger_track_data_into_official_game(field, value):
    # These compare equal to the printed values in Python. Before validation,
    # a +3.0 arrow passed the official gate but crashed on its first real move.
    with pytest.raises(ValueError, match="integers"):
        replace(official_config(), wild_track=replace(WILD_WILDS, **{field: value}))
