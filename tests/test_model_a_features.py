"""Observable state abstraction on actual complete-game decision requests."""

import copy
import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from magical_athlete.decisions import Action, RandomPolicy
from magical_athlete.experiment_records import build_game_records
from magical_athlete.game import EntryRequest, GameEngine, RandomGamePolicy, official_config
from magical_athlete.engine import RaceEngine
from magical_athlete.racer import Racer
from magical_athlete.track import WILD_WILDS
from magical_athlete.model_a_features import (
    FEATURE_VERSION, action_key, encode_request, legal_choices,
    normalize_request, request_scope, state_keys,
)


class CaptureGamePolicy(RandomGamePolicy):
    def __init__(self, seed, capture, context):
        super().__init__(seed)
        self.capture, self.context = capture, context

    def choose(self, request):
        if isinstance(request, EntryRequest):
            self.context["race_index"] = request.observation.race_index
        self.capture[("game", request.decision_id, request.player_id)] = request
        return super().choose(request)


class CaptureRacePolicy(RandomPolicy):
    def __init__(self, seed, capture, context):
        super().__init__(seed)
        self.capture, self.context = capture, context

    def choose(self, request):
        self.capture[("race", self.context["race_index"], request.decision_id, request.player_id)] = request
        return super().choose(request)


@pytest.fixture(scope="module")
def actual_decisions():
    capture, context = {}, {}
    game = GameEngine(
        official_config(), 1,
        game_policies={seat: CaptureGamePolicy(100 + seat, capture, context) for seat in range(4)},
        race_policies={seat: CaptureRacePolicy(200 + seat, capture, context) for seat in range(4)},
    )
    game.run()
    return build_game_records(game.trace())["decisions"], capture


def test_all_actual_typed_choices_and_json_replay_requests_have_identical_features(actual_decisions):
    decisions, capture = actual_decisions
    scopes = set()
    for decision in decisions:
        scope, index = decision["scope"], decision["race_index"]
        scopes.add(scope)
        key = (("race", index, decision["local_decision_id"], decision["player_id"])
               if scope == "race" else ("game", decision["local_decision_id"], decision["player_id"]))
        typed = capture[key]
        raw = decision["request"]
        # Typed EntryRequest.track owns frozensets: the real policy call must
        # normalize just as successfully as the already-JSON replay log.
        assert normalize_request(typed, race_index=index) == raw
        encoded = encode_request(typed, race_index=index)
        assert encoded == encode_request(raw, race_index=index)
        assert encoded.scope == scope and encoded.race_index == index
        assert len(encoded.actions) == len(encoded.action_keys) == len(decision["legal_actions"])
        assert encoded.state_keys == state_keys(raw, race_index=index)
        assert len(set(encoded.state_keys)) == 3
        assert all(key.startswith(FEATURE_VERSION + "|") for key in encoded.state_keys)
        for choice, identity in zip(encoded.actions, encoded.action_keys):
            assert action_key(raw, choice, race_index=index) == identity
        assert request_scope(typed) == scope
    assert scopes == {"draft", "entry", "race"}


def rotate_request(request, amount):
    raw = copy.deepcopy(request)
    rotate = lambda seat: None if seat is None else (seat + amount) % 4

    def seats(values):
        output = [None] * 4
        for seat, value in enumerate(values):
            output[rotate(seat)] = value
        return output

    raw["player_id"] = rotate(raw["player_id"])
    observation = raw["observation"]
    if "window" in raw:
        observation["current_player_id"] = rotate(observation["current_player_id"])
        for racer in observation["racers"]:
            racer["racer_id"] = rotate(racer["racer_id"])
            racer["copied_target"] = rotate(racer["copied_target"])
        observation["finish_order"] = [rotate(seat) for seat in observation["finish_order"]]
        observation["score_totals"] = [[rotate(seat), total] for seat, total in observation["score_totals"]]
        for action in raw["legal_actions"]:
            action["source_id"], action["target_id"] = rotate(action["source_id"]), rotate(action["target_id"])
    else:
        for name in ("scores", "hand_sizes", "used_cards"):
            observation[name] = seats(observation[name])
        for pick in observation["draft_history"]:
            pick["player_id"] = rotate(pick["player_id"])
        observation["starting_player"] = rotate(observation["starting_player"])
        observation["previous_entries"] = [seats(entries) for entries in observation["previous_entries"]]
        observation["previous_finish_orders"] = [[rotate(seat) for seat in order] for order in observation["previous_finish_orders"]]
    return raw


@pytest.mark.parametrize("rotation", [1, 2, 3])
def test_complete_actual_game_groups_and_target_actions_are_invariant_to_seat_rotation(actual_decisions, rotation):
    decisions, _ = actual_decisions
    for decision in decisions:
        request, index = decision["request"], decision["race_index"]
        rotated = rotate_request(request, rotation)
        original = encode_request(request, race_index=index)
        turned = encode_request(rotated, race_index=index)
        assert turned.state_keys == original.state_keys
        assert turned.action_keys == original.action_keys
        assert turned.public_scores == original.public_scores
        assert turned.repetition_key == original.repetition_key


def test_counters_card_entity_ids_and_provenance_are_not_strategy_information(actual_decisions):
    decisions, _ = actual_decisions
    for decision in decisions:
        request = copy.deepcopy(decision["request"])
        index = decision["race_index"]
        original = encode_request(request, race_index=index)
        request["decision_id"] += 5000
        observation = request["observation"]
        if decision["scope"] == "race":
            observation["turn_number"] += 4000
        else:
            rename = lambda card_id: "physical-id-" + card_id
            for card in request["hand"] + request.get("available", []):
                card["card_id"] = rename(card["card_id"])
            for pick in observation["draft_history"]:
                pick["card"]["card_id"] = rename(pick["card"]["card_id"])
            request["legal_card_ids"] = [rename(card_id) for card_id in request["legal_card_ids"]]
            observation["used_cards"] = [[rename(card_id) for card_id in cards] for cards in observation["used_cards"]]
            observation["previous_entries"] = [[rename(card_id) for card_id in cards] for cards in observation["previous_entries"]]
            if observation["track"] is not None:
                observation["track"]["source"] = "different local path to the same public board"
                observation["track"]["data_version"] = "same geometry, other provenance label"
        changed = encode_request(request, race_index=index)
        assert changed.state_keys == original.state_keys
        assert changed.action_keys == original.action_keys


def test_fine_groups_distinguish_visible_changes_while_coarse_groups_share_experience(actual_decisions):
    decisions, _ = actual_decisions
    # v2 intentionally separates shared-space and ahead states even at its
    # coarsest level. Select a genuine ahead opponent so +1 preserves that
    # interaction class while the exact public geometry still changes.
    decision = next(decision for decision in decisions if decision["scope"] == "race" and any(
        racer["racer_id"] != decision["player_id"] and racer["position"] < 29
        and racer["position"] > next(own["position"] for own in decision["request"]["observation"]["racers"]
                                     if own["racer_id"] == decision["player_id"])
        for racer in decision["request"]["observation"]["racers"]
    ))
    request, index = copy.deepcopy(decision["request"]), decision["race_index"]
    before = state_keys(request, race_index=index)
    actor = request["player_id"]
    own = next(racer for racer in request["observation"]["racers"] if racer["racer_id"] == actor)
    other = next(racer for racer in request["observation"]["racers"]
                 if racer["racer_id"] != actor and own["position"] < racer["position"] < 29)
    other["position"] += 1
    after = state_keys(request, race_index=index)
    assert before[0] != after[0]
    assert before[-1] == after[-1]
    assert "|fine|" in before[0] and "|coarse|" in before[-1]


def test_own_visible_hand_and_available_pool_change_fine_draft_context(actual_decisions):
    decisions, _ = actual_decisions
    request = copy.deepcopy(next(decision["request"] for decision in decisions if decision["scope"] == "draft"))
    before = state_keys(request)
    # A different publicly visible pool is different information, even while
    # the transferable coarse round group and shared role actions stay usable.
    request["available"] = request["available"][:-1]
    request["legal_card_ids"] = request["legal_card_ids"][:-1]
    after = state_keys(request)
    assert before[0] != after[0]
    assert before[-1] == after[-1]


@pytest.mark.parametrize("location", ["request", "observation", "racer", "action"])
def test_hidden_or_future_fields_fail_closed_instead_of_entering_features(actual_decisions, location):
    decisions, _ = actual_decisions
    request = copy.deepcopy(next(decision["request"] for decision in decisions if decision["scope"] == "race"))
    containers = {"request": request, "observation": request["observation"],
                  "racer": request["observation"]["racers"][0], "action": request["legal_actions"][0]}
    containers[location]["future_dice_or_hidden_deck"] = [6, 6, 6]
    with pytest.raises(ValueError, match="unknown or missing fields"):
        encode_request(request, race_index=1)


@pytest.mark.parametrize("index", [None, 0, 5, True, 1.0])
def test_race_context_must_be_explicit_and_valid(actual_decisions, index):
    decisions, _ = actual_decisions
    request = next(decision["request"] for decision in decisions if decision["scope"] == "race")
    with pytest.raises(ValueError):
        encode_request(request, race_index=index)


def test_game_race_context_cannot_be_relabelled(actual_decisions):
    decisions, _ = actual_decisions
    request = next(decision["request"] for decision in decisions if decision["scope"] == "entry")
    with pytest.raises(ValueError, match="explicit race index"):
        encode_request(request, race_index=2)


def test_selected_actions_are_checked_for_legality_and_do_not_change_state_groups(actual_decisions):
    decisions, _ = actual_decisions
    for decision in decisions:
        request, index = decision["request"], decision["race_index"]
        before = state_keys(request, race_index=index)
        for choice in legal_choices(request):
            action_key(request, choice, race_index=index)
        assert state_keys(request, race_index=index) == before
    draft = next(decision["request"] for decision in decisions if decision["scope"] == "draft")
    with pytest.raises(ValueError, match="illegal action"):
        action_key(draft, "not-a-visible-card")


def test_encoded_actions_and_groups_are_detached_immutable_values(actual_decisions):
    decisions, _ = actual_decisions
    raw = copy.deepcopy(next(decision["request"] for decision in decisions if decision["scope"] == "race"))
    encoded = encode_request(raw, race_index=1)
    raw["legal_actions"][0]["value"] = "changed after encoding"
    raw["observation"]["racers"][0]["position"] = 29
    assert encoded.actions[0].value != "changed after encoding"
    with pytest.raises(FrozenInstanceError):
        encoded.scope = "different"
    with pytest.raises(FrozenInstanceError):
        encoded.actions[0].value = "different"
    assert all(isinstance(action, Action) for action in encoded.actions)


def actual_hypnotist_request():
    class KeepAndCapture:
        def __init__(self):
            self.requests = []

        def choose(self, request):
            self.requests.append(request)
            return next(action for action in request.legal_actions if action.kind == "keep")

    class One:
        def roll(self):
            return 1

    policy = KeepAndCapture()
    racers = [Racer(0, "Hypnotist", 9), Racer(1, "Baba Yaga", 12), Racer(2, "Hare", 29), Racer(3, "P4", 0)]
    engine = RaceEngine(WILD_WILDS, racers, [0, 1, 2, 3], One(), policies={0: policy})
    engine.take_turn(racers[0])
    request = next(request for request in policy.requests if request.window == "turn_start")
    return normalize_request(request, race_index=2)


def test_v2_coarse_preserves_tripped_and_sharing_while_middle_retains_actor_offset():
    original = actual_hypnotist_request()
    before = encode_request(original, race_index=2)
    mutations = []
    recovering = copy.deepcopy(original)
    recovering["observation"]["racers"][0]["tripped"] = True
    mutations.append(recovering)
    shared = copy.deepcopy(original)
    shared["observation"]["racers"][1]["position"] = 9
    mutations.append(shared)
    for changed in mutations:
        after = encode_request(changed, race_index=2)
        assert before.state_keys[-1] != after.state_keys[-1]
        assert before.repetition_key != after.repetition_key
    coarse = json.loads(before.state_keys[-1].split("|", 2)[2])
    assert coarse["self_tripped"] is False and coarse["sharing"] is False
    assert "score_band" not in coarse and "score_gap" not in coarse
    another_turn = copy.deepcopy(original)
    another_turn["observation"]["current_player_id"] = 1
    changed_actor = encode_request(another_turn, race_index=2)
    assert changed_actor.state_keys[-1] == before.state_keys[-1]
    assert changed_actor.state_keys[1] != before.state_keys[1]
    assert changed_actor.repetition_key != before.repetition_key


def test_opponent_score_farming_does_not_hide_geometry_repetition_or_change_own_balance():
    original = actual_hypnotist_request()
    before = encode_request(original, race_index=2)
    changed = copy.deepcopy(original)
    changed["observation"]["score_totals"][1][1] += 532
    after = encode_request(changed, race_index=2)
    assert before.state_keys[0] != after.state_keys[0]
    assert before.state_keys[-1] == after.state_keys[-1]
    assert before.repetition_key == after.repetition_key
    assert before.public_scores == (0, 0, 0, 0)
    assert after.public_scores == (0, 532, 0, 0)
    assert isinstance(after.public_scores, tuple)


def test_own_score_changes_remain_visible_to_policy_memory_without_changing_geometry_key():
    original = actual_hypnotist_request()
    changed = copy.deepcopy(original)
    changed["observation"]["score_totals"][0][1] = 3
    before, after = encode_request(original, race_index=2), encode_request(changed, race_index=2)
    assert before.public_scores[0] == 0 and after.public_scores[0] == 3
    assert before.repetition_key == after.repetition_key
    assert before.state_keys[0] != after.state_keys[0]


def test_v2_middle_preserves_immediate_public_trip_and_near_finish_distance():
    request = actual_hypnotist_request()
    near_trip = copy.deepcopy(request)
    near_trip["observation"]["racers"][0]["position"] = 26
    away_trip = copy.deepcopy(near_trip)
    away_trip["observation"]["racers"][0]["position"] = 27
    encoded_trip = encode_request(near_trip, race_index=2)
    encoded_away = encode_request(away_trip, race_index=2)
    trip = json.loads(encoded_trip.state_keys[1].split("|", 2)[2])
    away = json.loads(encoded_away.state_keys[1].split("|", 2)[2])
    assert trip["on_trip"] is True and away["on_trip"] is False
    assert trip["finish_distance"] == {"exact": 4}
    assert away["finish_distance"] == {"exact": 3}
    assert encoded_trip.state_keys[-1] == encoded_away.state_keys[-1]


def test_json_track_geometry_container_order_has_the_same_feature_semantics(actual_decisions):
    decisions, _ = actual_decisions
    request = copy.deepcopy(next(decision["request"] for decision in decisions
                                 if decision["scope"] == "entry" and decision["race_index"] == 2))
    before = encode_request(request)
    track = request["observation"]["track"]
    track["trip_positions"] = list(reversed(track["trip_positions"]))
    track["star_positions"] = list(reversed(track["star_positions"]))
    assert encode_request(request) == before


def test_copy_coarse_context_preserves_leading_role_relations_and_keep_legality():
    class CaptureFirst:
        def __init__(self):
            self.requests = []

        def choose(self, request):
            self.requests.append(request)
            return request.legal_actions[0]

    class UnusedDice:
        def roll(self):
            raise AssertionError("before-race copy registration must not roll a die")

    groups = []
    for copy_position in (10, 12):
        policy = CaptureFirst()
        racers = [Racer(0, "Skipper", 12), Racer(1, "Romantic", 12),
                  Racer(2, "Copy Cat", copy_position), Racer(3, "Lackey", 6)]
        engine = RaceEngine(WILD_WILDS, racers, [2, 3, 0, 1], UnusedDice(), policies={2: policy})
        initial = encode_request(policy.requests[0], race_index=2)
        coarse = json.loads(initial.state_keys[-1].split("|", 2)[2])
        middle = json.loads(initial.state_keys[1].split("|", 2)[2])
        assert coarse["can_keep_copy"] is False
        assert "copy_targets" not in coarse
        assert {target["role"] for target in middle["copy_targets"]} == {"Skipper", "Romantic"}
        assert {target["relation"] for target in middle["copy_targets"]} == ({"ahead"} if copy_position == 10 else {"same"})
        # A still-valid existing target makes KEEP a different legal window.
        engine._refresh_copies(allow_choice=True)
        optional = encode_request(policy.requests[-1], race_index=2)
        optional_coarse = json.loads(optional.state_keys[-1].split("|", 2)[2])
        assert optional_coarse["can_keep_copy"] is True
        groups.append(initial.state_keys[-1])
    assert groups[0] != groups[1]
