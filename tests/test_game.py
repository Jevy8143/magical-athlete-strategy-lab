"""Four-player game workflow with explicit restricted/synthetic data.

All races below execute the real RaceEngine. The 16-card subset, fixture Wild
and irregular award values are labelled test data, never official game data.
The original 36-card draft is tested independently from unsupported abilities.
"""

import copy
import random
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from magical_athlete.decisions import ActionKind, RandomPolicy
from magical_athlete.engine import D6, replay_trace
from magical_athlete.game import (
    BASE_DECK, DraftRequest, EntryRequest, FirstLegalGamePolicy, GameConfig,
    GameDiagnosticError, GameEngine, RacerCard, RandomGamePolicy, replay_game_trace,
)
from magical_athlete.racer import Ability
from magical_athlete.scoring import AwardSchedule
from magical_athlete.track import MILD_MILE, Track


FIXTURE_ABILITIES = (
    Ability.ALCHEMIST, Ability.ROCKET_SCIENTIST, Ability.MAGICIAN, Ability.DICEMONGER,
    Ability.SCOOCHER, Ability.COACH, Ability.GUNK, Ability.STICKLER,
    Ability.SKIPPER, Ability.INCHWORM, Ability.LEGS, Ability.HARE,
    Ability.LACKEY, Ability.GENIUS, Ability.HECKLER, Ability.BANANA,
)


def fixture_config(*, deck=None, awards=None, **kwargs):
    cards = tuple(RacerCard(a.value, a.value, a) for a in FIXTURE_ABILITIES) if deck is None else deck
    # These deliberately irregular contract values are not official chip values.
    schedule = AwardSchedule(
        awards if awards is not None else ((2, 1), (6, 4), (5, 2), (11, 8)),
        source="Synthetic award contract values; not official chips", verified=False,
    )
    return GameConfig(
        Track("Synthetic Mild workflow fixture", 30, 15, data_version="synthetic-mild-game-v1"),
        Track("Synthetic Wild workflow fixture", 35, 18, data_version="synthetic-wild-game-v1"),
        schedule, deck=cards, official=False, **kwargs,
    )


class PrefixDice:
    def __init__(self, prefix, seed=991):
        self.prefix = list(prefix)
        self.tail = D6(seed)

    def roll(self):
        return self.prefix.pop(0) if self.prefix else self.tail.roll()


class RecordingGamePolicy(FirstLegalGamePolicy):
    def __init__(self):
        self.requests = []

    def choose(self, request):
        self.requests.append(request)
        return super().choose(request)


def make_drafted_game(*, policies=None, config=None):
    game = GameEngine(config or fixture_config(deck=BASE_DECK), 7,
                      game_policies=policies, dice=PrefixDice([6, 5, 4, 3]))
    game.draft()
    return game


def test_exact_two_snake_orders_and_original_36_card_draft():
    game = make_drafted_game()
    assert game.draft_start == 0
    assert [pick.player_id for pick in game.draft_history] == [
        0, 1, 2, 3, 3, 2, 1, 0, 1, 2, 3, 0, 0, 3, 2, 1,
    ]
    assert [len(game.hands[p]) for p in range(4)] == [4, 4, 4, 4]
    assert len(game.remaining_deck) == 20
    assert len({card.card_id for hand in game.hands.values() for card in hand}) == 16
    assert {c.card_id for c in game.remaining_deck}.isdisjoint(c.card_id for h in game.hands.values() for c in h)
    assert [len(d.request.available) for d in game.decisions] == list(range(8, 0, -1)) * 2
    assert not game.pool
    with pytest.raises(RuntimeError, match="once"):
        game.draft()


def test_snake_order_rotates_from_actual_start_player():
    assert GameEngine.snake_order(2) == (2, 3, 0, 1, 1, 0, 3, 2)


def test_rolloff_uses_next_highest_unique_roll_instead_of_rerolling_top_tie():
    game = GameEngine(fixture_config(), 0, dice=PrefixDice([6, 6, 5, 4]))
    assert game.roll_off("test") == 2
    assert game.dice_rolls == [6, 6, 5, 4]
    assert game.rolloffs[0].reason == "highest_unique_roll"


def test_everyone_same_roll_rerolls_and_records_both_rounds():
    game = GameEngine(fixture_config(), 0, dice=PrefixDice([2, 2, 2, 2, 3, 6, 1, 4]))
    assert game.roll_off("test") == 1
    assert [record.winner for record in game.rolloffs] == [None, 1]
    assert game.dice_rolls == [2, 2, 2, 2, 3, 6, 1, 4]


def test_h12_no_unique_roll_in_two_tied_pairs_causes_all_four_to_reroll():
    game = GameEngine(fixture_config(), 0, dice=PrefixDice([6, 6, 5, 5, 4, 3, 2, 1]))
    assert game.roll_off("test") == 0
    assert game.trace()["completed"] is False
    assert game.dice_rolls == [6, 6, 5, 5, 4, 3, 2, 1]
    assert game.rolloffs[0].winner is None
    assert game.rolloffs[0].reason == "no_unique_roll_reroll"
    assert game.rolloffs[1].winner == 0


def test_rolloff_budget_failure_never_manufactures_a_starter():
    game = GameEngine(fixture_config(max_rolloff_rounds=2), 0, dice=PrefixDice([1] * 8))
    with pytest.raises(GameDiagnosticError, match="budget") as failure:
        game.roll_off("test")
    assert len(failure.value.trace["rolloffs"]) == 2
    assert all(record["winner"] is None for record in failure.value.trace["rolloffs"])


def test_h13_subset_rolloff_excludes_other_seats_and_preserves_actual_ids():
    game = GameEngine(fixture_config(), 0, dice=PrefixDice([2, 5, 6]))
    assert game.roll_off("race_2_start_tied_last", participants=(1, 3)) == 3
    assert game.dice_rolls == [2, 5]
    assert game.rolloffs[0].participants == (1, 3)
    assert game.rolloffs[0].rolls == (2, 5)


def test_h13_tied_subset_rerolls_only_that_same_subset():
    game = GameEngine(fixture_config(), 0, dice=PrefixDice([4, 4, 1, 6]))
    assert game.roll_off("race_2_start_tied_last", participants=(1, 3)) == 3
    assert [r.participants for r in game.rolloffs] == [(1, 3), (1, 3)]
    assert [r.rolls for r in game.rolloffs] == [(4, 4), (1, 6)]
    assert game.dice_rolls == [4, 4, 1, 6]


def test_entry_choices_share_one_public_snapshot_and_reveal_only_after_all_submit():
    policies = {p: RecordingGamePolicy() for p in range(4)}
    game = make_drafted_game(policies=policies)
    hands_before = {p: tuple(game.hands[p]) for p in range(4)}
    entries = game.select_entries(1)
    requests = [next(r for r in policies[p].requests if isinstance(r, EntryRequest)) for p in range(4)]
    assert all(r.observation is requests[0].observation for r in requests)
    assert all(r.observation.hand_sizes == (4, 4, 4, 4) for r in requests)
    assert all(r.observation.previous_entries == () for r in requests)
    assert all(r.hand == hands_before[p] for p, r in enumerate(requests))
    assert entries == tuple(hands_before[p][0] for p in range(4))
    assert [len(game.hands[p]) for p in range(4)] == [3, 3, 3, 3]
    assert game.active_entries == entries
    assert {c.card_id for c in entries}.isdisjoint(c.card_id for h in game.hands.values() for c in h)


def test_entry_snapshots_are_frozen_and_have_no_future_or_pending_choice_fields():
    policies = {p: RecordingGamePolicy() for p in range(4)}
    game = make_drafted_game(policies=policies)
    game.select_entries(1)
    request = next(r for r in policies[3].requests if isinstance(r, EntryRequest))
    with pytest.raises(FrozenInstanceError):
        request.observation.scores = (99, 99, 99, 99)
    with pytest.raises(FrozenInstanceError):
        request.hand[0].card_id = "changed"
    forbidden = {"dice", "seed", "rng", "engine", "remaining_deck", "deck_order", "pending", "pending_choices", "active_entries"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(asdict(request))


def test_illegal_entry_choice_is_rejected_without_removing_any_hand_card():
    class InvalidEntryPolicy(FirstLegalGamePolicy):
        def choose(self, request):
            return "not-in-this-hand" if isinstance(request, EntryRequest) else super().choose(request)

    game = make_drafted_game(policies={2: InvalidEntryPolicy()})
    before = {p: tuple(game.hands[p]) for p in range(4)}
    with pytest.raises(ValueError, match="illegal"):
        game.select_entries(1)
    assert {p: tuple(game.hands[p]) for p in range(4)} == before
    assert not game.active_entries


def test_official_flag_rejects_restricted_deck_or_unverified_awards():
    config = fixture_config()
    with pytest.raises(ValueError, match="36-card"):
        GameConfig(config.mild_track, config.wild_track, config.award_schedule, deck=config.deck, official=True)
    with pytest.raises(ValueError, match="awards"):
        GameConfig(config.mild_track, config.wild_track, config.award_schedule, official=True)


def test_free_form_wild_source_or_version_labels_cannot_promote_a_fixture_to_official():
    fake_wild = Track("Wild Wilds", 35, 18, source="claimed official source", data_version="claimed-verified-v1")
    declared_awards = AwardSchedule(((1, 0),) * 4, "declared fixture awards", verified=True)
    with pytest.raises(ValueError, match="official Wild data"):
        GameConfig(MILD_MILE, fake_wild, declared_awards, official=True)


def test_duplicate_physical_card_ids_are_rejected():
    config = fixture_config()
    with pytest.raises(ValueError, match="unique"):
        fixture_config(deck=(*config.deck[:-1], config.deck[0]))


def completed_game(seed=2, *, awards=None, scoring_cards=False, random_policies=False):
    deck = tuple(RacerCard(a.value, a.value, a) for a in FIXTURE_ABILITIES)
    if scoring_cards:
        deck = (*deck[:-2], RacerCard("Loveable Loser", "Loveable Loser", Ability.LOVEABLE_LOSER),
                RacerCard("Sisyphus", "Sisyphus", Ability.SISYPHUS))
    game = GameEngine(
        fixture_config(deck=deck, awards=awards), seed,
        game_policies={p: RandomGamePolicy(100 + p) for p in range(4)} if random_policies else None,
        race_policies={p: RandomPolicy(200 + p) for p in range(4)} if random_policies else None,
    )
    game.run()
    return game


def test_four_real_races_follow_schedule_and_use_each_physical_card_once():
    game = completed_game()
    assert [race.track_kind for race in game.races] == ["Mild", "Wild", "Mild", "Wild"]
    assert [trace["track"]["finish_position"] for trace in game.race_traces] == [30, 35, 30, 35]
    assert [len(game.used_cards[p]) for p in range(4)] == [4, 4, 4, 4]
    assert all(not hand for hand in game.hands.values())
    cards = [card.card_id for race in game.races for card in race.entries]
    assert len(cards) == len(set(cards)) == 16
    assert not game.active_entries and not game.remaining_deck
    assert len([e for e in game.ledger.entries if e.reason.startswith("placement_")]) == 8
    assert game.trace()["completed"] is True
    assert game.trace()["run_kind"] == "synthetic_or_restricted_game_fixture"


def test_first_race_rolloff_is_fresh_and_later_starters_are_previous_last_not_rotation():
    game = completed_game()
    assert game.rolloffs[0].purpose == "draft_start"
    assert any(r.purpose == "first_race_start" for r in game.rolloffs)
    assert game.races[0].starting_player == game.first_race_start
    for previous, current in zip(game.races, game.races[1:]):
        last = min(previous.unplaced, key=lambda p: previous.final_positions[p])
        assert current.starting_player == last
    assert any(current.starting_player != (previous.starting_player + 1) % 4 for previous, current in zip(game.races, game.races[1:]))


def test_final_high_score_tie_has_all_four_co_winners_without_a_tiebreak():
    game = completed_game(awards=((0, 0),) * 4)
    result = game.run()
    assert result.scores == (0, 0, 0, 0)
    assert result.winners == (0, 1, 2, 3)
    assert not result.official


def test_game_and_each_race_replay_recorded_choices_scores_and_dice():
    game = completed_game()
    trace = game.trace()
    replayed = replay_game_trace(trace)
    assert replayed.trace() == trace
    for race_trace in game.race_traces:
        assert replay_trace(race_trace).trace()["finish_order"] == race_trace["finish_order"]


def test_score_events_include_real_pre_race_skill_awards_not_only_placement_chips():
    game = completed_game(scoring_cards=True)
    entries = game.ledger.entries
    assert any(e.source == "Sisyphus" and e.delta == 4 and e.phase == "before_race" for e in entries)
    assert sum(game.ledger.totals().values()) == sum(e.delta for e in entries)
    assert any(not e.reason.startswith("placement_") for e in entries)
    for race in game.races:
        assert race.race_scores == tuple(game.ledger.race_total(p, race.race_index) for p in range(4))
    assert replay_game_trace(game.trace()).trace() == game.trace()


def test_seeded_game_policy_and_race_policy_are_reproducible_without_global_randomness():
    before = random.getstate()
    first = completed_game(1, random_policies=True)
    second = completed_game(1, random_policies=True)
    assert first.trace() == second.trace()
    assert random.getstate() == before


def test_game_replay_rejects_changed_entry_choice_or_score_total():
    game = completed_game()
    trace = game.trace()
    forged = copy.deepcopy(trace)
    first_entry = next(d for d in forged["decisions"] if d["request"]["observation"]["phase"] == "entry_selection")
    first_entry["card_id"] = "not-held"
    with pytest.raises(ValueError):
        replay_game_trace(forged)
    forged = copy.deepcopy(trace)
    forged["scores"]["totals"]["0"] += 1
    with pytest.raises(ValueError, match="diverged"):
        replay_game_trace(forged)


def test_h13_real_four_race_game_rolls_only_actual_previous_tied_last_players():
    game = completed_game(0)
    records = [record for record in game.rolloffs if record.purpose.endswith("_start_tied_last")]
    assert records
    for record in records:
        next_race_index = int(record.purpose.split("_")[1])
        previous = game.races[next_race_index - 2]
        low = min(previous.final_positions[p] for p in previous.unplaced)
        tied = tuple(p for p in previous.unplaced if previous.final_positions[p] == low)
        assert record.participants == tied
        assert len(record.rolls) == len(tied)
        if record.winner is not None:
            assert game.races[next_race_index - 1].starting_player == record.winner
    assert replay_game_trace(game.trace()).trace() == game.trace()


def test_first_eliminated_has_priority_over_other_last_positions_for_next_start():
    # Pure orchestration selection rule; this does not claim M.O.U.T.H. coverage.
    game = completed_game()
    last = replace(game.races[0], elimination_order=(2, 3), unplaced=(2, 3),
                   final_positions=(30, 30, 12, 0))
    dice_before = tuple(game.dice_rolls)
    assert game._next_start(last) == 2
    assert tuple(game.dice_rolls) == dice_before


def test_every_race_entry_uses_one_shared_public_snapshot_and_own_remaining_hand():
    policies = {p: RecordingGamePolicy() for p in range(4)}
    game = GameEngine(fixture_config(), 2, game_policies=policies)
    game.run()
    requests = [[r for r in policies[p].requests if isinstance(r, EntryRequest)] for p in range(4)]
    for index in range(4):
        group = [requests[p][index] for p in range(4)]
        assert all(request.observation is group[0].observation for request in group)
        assert group[0].observation.hand_sizes == (4 - index,) * 4
        assert len(group[0].observation.previous_entries) == index
        assert all(len(request.hand) == 4 - index for request in group)
        assert all(tuple(c.card_id for c in request.hand) == request.legal_card_ids for request in group)
        assert all(request.observation.starting_player == game.races[index].starting_player for request in group)
        assert all(request.observation.current_awards == game.config.award_schedule.for_race(index + 1) for request in group)
        assert all(request.observation.track is game.config.track_for_race(index + 1) for request in group)
        assert group[0].observation.previous_entries == tuple(tuple(c.card_id for c in r.entries) for r in game.races[:index])
        if index:
            assert group[0].observation.scores == game.races[index - 1].scores_after


def test_draft_observation_does_not_invent_unrolled_race_starter_or_awards():
    policy = RecordingGamePolicy()
    game = make_drafted_game(policies={0: policy})
    assert all(request.observation.starting_player is None
               and request.observation.current_awards is None
               and request.observation.track is None for request in policy.requests)
    assert game.first_race_start is None


def test_used_physical_card_cannot_be_selected_again_in_a_later_race():
    class ReusePolicy(FirstLegalGamePolicy):
        def __init__(self):
            self.used = None

        def choose(self, request):
            if isinstance(request, DraftRequest):
                return super().choose(request)
            if self.used is None:
                self.used = request.legal_card_ids[0]
            return self.used

    policy = ReusePolicy()
    game = GameEngine(fixture_config(), 2, game_policies={0: policy})
    with pytest.raises(ValueError, match="illegal"):
        game.run()
    assert len(game.races) == 1
    assert policy.used == game.used_cards[0][0].card_id
    assert policy.used not in {c.card_id for c in game.hands[0]}
    assert tuple(len(game.hands[p]) for p in range(4)) == (3, 3, 3, 3)
    assert not game.active_entries


def test_race_budget_diagnostic_stays_incomplete_and_does_not_award_placements():
    game = GameEngine(fixture_config(max_turns_per_race=1), 2)
    with pytest.raises(GameDiagnosticError, match="budget") as failure:
        game.run()
    trace = failure.value.trace
    assert trace["completed"] is False
    assert trace["races"] == []
    assert trace["active_race"] is not None
    assert trace["race_diagnostic"]["completed"] is False
    assert trace["diagnostic_failure"] == "race_turn_budget_exceeded"
    assert not any(entry["reason"].startswith("placement_") for entry in trace["scores"]["entries"])
    with pytest.raises(ValueError, match="completed"):
        replay_game_trace(trace)


def test_completed_game_run_is_idempotent_and_does_not_award_chips_twice():
    game = completed_game()
    before = game.trace()
    result = game.run()
    assert result.scores == tuple(game.ledger.total(p) for p in range(4))
    assert game.trace() == before


def test_game_dice_seed_metadata_distinguishes_default_and_external_sources():
    default = GameEngine(fixture_config(), 2)
    external = GameEngine(fixture_config(), 2, dice=PrefixDice([6, 5, 4, 3]))
    assert default.dice_seed == default.trace()["dice_seed"] == 3
    assert external.dice_seed is None
    assert external.trace()["dice_seed"] is None


def test_completed_external_dice_game_replay_preserves_unknown_seed_metadata():
    game = GameEngine(fixture_config(), 2, dice=PrefixDice([6, 5, 4, 3, 5, 6, 4, 3], seed=17))
    game.run()
    trace = game.trace()
    assert trace["dice_seed"] is None
    replayed = replay_game_trace(trace)
    assert replayed.dice_seed is None
    assert replayed.trace() == trace


def test_event_budget_diagnostic_is_not_mislabeled_as_a_turn_budget_failure():
    game = GameEngine(fixture_config(max_events_per_race=3), 2)
    with pytest.raises(GameDiagnosticError, match="race_event_budget_exceeded") as failure:
        game.run()
    trace = failure.value.trace
    assert trace["completed"] is False
    assert trace["diagnostic_failure"] == "race_event_budget_exceeded"
    assert trace["race_diagnostic_reason"] == "event_budget_exceeded"
    assert trace["race_diagnostic"]["diagnostic_failure"] == "event_budget_exceeded"
    assert trace["active_race"] is not None
    assert len(trace["race_diagnostic"]["events"]) == 3
    assert trace["race_diagnostic"]["max_events"] == 3
    assert not any(e["reason"].startswith("placement_") for e in trace["scores"]["entries"])


def test_before_race_constructor_budget_failure_preserves_partial_race_trace():
    class PreferSisyphusOrScoocher(FirstLegalGamePolicy):
        def choose(self, request):
            if isinstance(request, EntryRequest):
                for name in ("Sisyphus", "Scoocher"):
                    if name in request.legal_card_ids:
                        return name
            return super().choose(request)

    deck = tuple(RacerCard(a.value, a.value, a) for a in FIXTURE_ABILITIES[:-2]) + (
        RacerCard("Loveable Loser", "Loveable Loser", Ability.LOVEABLE_LOSER),
        RacerCard("Sisyphus", "Sisyphus", Ability.SISYPHUS),
    )
    # Seed 0 drafts Sisyphus to seat 3 and Scoocher to seat 1. Before-race
    # scoring creates one score event, then the notification exhausts the budget.
    game = GameEngine(fixture_config(deck=deck, max_events_per_race=1), 0,
                      game_policies={p: PreferSisyphusOrScoocher() for p in range(4)})
    with pytest.raises(GameDiagnosticError, match="race_event_budget_exceeded") as failure:
        game.run()
    trace = failure.value.trace
    assert trace["completed"] is False
    assert trace["active_race"] is None  # Constructor never returned a RaceEngine.
    assert trace["race_diagnostic_reason"] == "event_budget_exceeded"
    assert trace["race_diagnostic"]["turns"] == []
    assert trace["race_diagnostic"]["events"][0]["kind"] == "score"
    assert trace["race_diagnostic"]["events"][0]["ability"] == "Sisyphus"
    assert any(e["source"] == "Sisyphus" and e["phase"] == "before_race" for e in trace["scores"]["entries"])
    assert not any(e["reason"].startswith("placement_") for e in trace["scores"]["entries"])


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_game_event_budget_requires_a_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integers"):
        fixture_config(max_events_per_race=budget)


class PreferEggGamePolicy(FirstLegalGamePolicy):
    """Legal own-card choices; no hidden deck or other pending choice access."""

    def choose(self, request):
        if "Egg" in request.legal_card_ids:
            return "Egg"
        if isinstance(request, EntryRequest):
            for card_id in request.legal_card_ids:
                if card_id not in {"Sisyphus", "Scoocher", "Twin", "Copy Cat"}:
                    return card_id
        return super().choose(request)


class KeepOptionalChooseMandatory:
    policy_id = "test-keep-optional-first-mandatory-v1"

    def choose(self, request):
        return next((action for action in request.legal_actions if action.kind == ActionKind.KEEP),
                    request.legal_actions[0])


def original_deck_egg_game(*, max_events=100_000):
    # The full 36-card deck leaves 20 undrafted physical cards for Egg. Seed 1
    # drafts Egg to seat 0; the policy declares it for race 1. Track/awards remain
    # explicit fixtures, so this remains a non-official workflow test.
    return GameEngine(
        fixture_config(deck=BASE_DECK, max_events_per_race=max_events), 1,
        game_policies={p: PreferEggGamePolicy() for p in range(4)},
        race_policies={p: KeepOptionalChooseMandatory() for p in range(4)},
    )


def assert_all_original_physical_cards_accounted_for(game):
    cards = game.remaining_deck + game.pool + list(game.active_entries) + game.ability_cards_out
    cards += [card for hand in game.hands.values() for card in hand]
    cards += [card for used in game.used_cards.values() for card in used]
    assert len(cards) == len({card.card_id for card in cards}) == 36
    assert {card.card_id for card in cards} == {card.card_id for card in BASE_DECK}


def test_original_36_card_egg_game_draws_only_undrafted_physical_cards_and_replays():
    game = original_deck_egg_game()
    result = game.run()
    assert len(result.races) == 4
    assert any(card.card_id == "Egg" for card in game.races[0].entries)
    drawn = [Ability(value) for trace in game.race_traces for value in trace["egg_drawn"]]
    assert len(drawn) >= 3
    assert [card.ability for card in game.ability_cards_out] == drawn
    drafted_ids = {pick.card.card_id for pick in game.draft_history}
    assert drafted_ids.isdisjoint(card.card_id for card in game.ability_cards_out)
    assert {card.card_id for card in game.ability_cards_out}.isdisjoint(card.card_id for card in game.remaining_deck)
    assert_all_original_physical_cards_accounted_for(game)
    assert game.trace()["ability_cards_out"]
    assert game.trace()["run_kind"] == "synthetic_or_restricted_game_fixture"
    assert replay_game_trace(game.trace()).trace() == game.trace()
    for trace in game.race_traces:
        assert replay_trace(trace).trace()["egg_drawn"] == trace["egg_drawn"]


def test_game_passes_original_physical_winner_abilities_as_twin_context():
    game = original_deck_egg_game()
    game.run()
    for index, trace in enumerate(game.race_traces):
        expected = [
            race.entries[race.finish_order[0]].ability.value for race in game.races[:index]
            if race.finish_order and race.entries[race.finish_order[0]].ability is not None
        ]
        assert trace["previous_winners"] == expected


def test_egg_winner_contributes_original_egg_not_borrowed_twin_to_later_context():
    # Draft starts at 3, first race at Egg's seat 0. The legal fixed prefix
    # gives Egg five sixes while the others roll ones; Egg's selected Twin
    # power has no previous-race target in race 1 and adds no movement.
    prefix = [1, 1, 1, 6, 6, 5, 4, 3] + [6, 1, 1, 1] * 5
    game = GameEngine(
        fixture_config(deck=BASE_DECK), 1,
        game_policies={p: PreferEggGamePolicy() for p in range(4)},
        race_policies={p: KeepOptionalChooseMandatory() for p in range(4)},
        dice=PrefixDice(prefix, seed=17),
    )
    game.run()
    assert game.races[0].entries[0].ability == Ability.EGG
    assert game.races[0].finish_order[0] == 0
    egg_final = next(r for r in game.race_traces[0]["racers"] if r["racer_id"] == 0)
    assert egg_final["inherited_ability"] == "Twin"
    assert game.race_traces[1]["previous_winners"][0] == "Egg"
    assert game.race_traces[1]["previous_winners"][0] != egg_final["inherited_ability"]
    assert replay_game_trace(game.trace()).trace() == game.trace()


def test_egg_draws_are_consumed_when_before_race_constructor_hits_event_budget():
    game = original_deck_egg_game(max_events=1)
    with pytest.raises(GameDiagnosticError, match="race_event_budget_exceeded") as failure:
        game.run()
    trace = failure.value.trace
    assert trace["completed"] is False
    assert trace["active_race"] is None
    assert len(trace["race_diagnostic"]["egg_drawn"]) == 3
    assert len(game.ability_cards_out) == 3
    assert [card.ability.value for card in game.ability_cards_out] == trace["race_diagnostic"]["egg_drawn"]
    assert len(game.remaining_deck) == 17
    assert_all_original_physical_cards_accounted_for(game)
    assert not any(entry["reason"].startswith("placement_") for entry in trace["scores"]["entries"])


def test_egg_draws_are_not_consumed_twice_when_later_run_hits_event_budget():
    game = original_deck_egg_game(max_events=20)
    with pytest.raises(GameDiagnosticError, match="race_event_budget_exceeded") as failure:
        game.run()
    trace = failure.value.trace
    assert trace["active_race"] is not None
    assert len(trace["race_diagnostic"]["egg_drawn"]) == 3
    assert len(game.ability_cards_out) == 3
    assert len(game.remaining_deck) == 17
    assert_all_original_physical_cards_accounted_for(game)
