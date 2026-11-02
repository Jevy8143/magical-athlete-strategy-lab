"""Score-posterior mathematics, actual legal play and immutable MC updates."""

import hashlib
import json
import random
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from magical_athlete.experiment_records import build_game_records
from magical_athlete.experiment_spec import PolicySnapshot, canonical_json
from magical_athlete.game import BASE_DECK, EntryRequest, GameEngine, PublicGameObservation, official_config, replay_game_trace
from magical_athlete.model_a import (
    ALGORITHM, AConfig, AGamePolicy, ARacePolicy, FrozenAModel, PosteriorRow, SharedPublicContext,
)
from magical_athlete.model_a_features import action_key, encode_request
from magical_athlete.track import WILD_WILDS


def request(*, player=0, score=100, decision=1):
    """A possible public race-2 entry window; balances are not a fixed supply."""
    cards = BASE_DECK[:2]
    observation = PublicGameObservation(
        "entry_selection", 2, ("Mild", "Wild", "Mild", "Wild"), (score,) * 4,
        (2,) * 4, ((),) * 4, (), (), (), 0, (4, 2), WILD_WILDS,
    )
    return EntryRequest(decision, player, observation, cards, tuple(card.card_id for card in cards))


def transition(*, player=0, score=100, final=110, selected=0, seq=1, done=True):
    req = request(player=player, score=score, decision=seq)
    return {"decision_seq": seq, "player_id": player, "scope": "entry", "race_index": 2,
            "request": json.loads(canonical_json(req)),
            "action": req.legal_card_ids[selected], "score_before": score,
            "final_total": final, "return_to_go": final - score, "done": done}


def source(game_id, transitions=None, *, context="opponents-a0", focus=0, opponents=None):
    return {"game_id": game_id, "context_id": context, "focus_player_id": focus,
            "purpose": "train", "transitions": transitions if transitions is not None else [transition()],
            "opponent_snapshot_ids": opponents}


def test_prior_is_distribution_over_signed_scores_and_unseen_scores_are_not_clipped():
    model = FrozenAModel.empty()
    encoded = encode_request(request())
    prior = model.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[0])
    assert prior.mean == pytest.approx(6 / 1.75)
    assert prior.source == "prior_only" and prior.observations == 0
    updated = model.update_games([source("negative", [transition(final=97)]),
                                  source("large", [transition(final=1100)])])
    estimate = updated.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[0])
    assert estimate.mean == pytest.approx((6 - 3 + 1000) / (1.75 + 2))
    assert estimate.observations == 2 and estimate.source == "coarse_sparse"
    assert {-3, 1000} == set(updated.summary()["observed_return_values"])
    assert model.training_games == 0 and updated.training_games == 2


def test_hierarchy_uses_coarse_sparse_fallback_then_fine_threshold_and_disjoint_contexts():
    encoded = encode_request(request())
    model = FrozenAModel.empty().update_games([source("one")])
    sparse = model.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[0])
    assert sparse.level == 2 and sparse.source == "coarse_sparse"
    model = model.update_games([source("two"), source("three")])
    fine = model.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[0])
    assert fine.level == 0 and fine.source == "sample_threshold" and fine.observations == 3
    other = model.estimate("different-opponents", encoded.state_keys, encoded.action_keys[0])
    assert other.source == "prior_only" and other.observations == 0


def test_first_visit_counts_one_focus_trajectory_once_at_each_level():
    rows = [transition(seq=1, done=False), transition(seq=2), transition(player=1, seq=3, final=999)]
    model = FrozenAModel.empty().update_games([source("one-game", rows)])
    assert model.training_games == 1
    assert len(model.rows) == 3
    assert all(row.counts == ((10, 1),) and row.observations == 1 for row in model.rows)
    assert model.summary()["hierarchical_first_visit_observations"] == 3


def test_return_is_remaining_actual_score_not_final_total_repeated_as_new_reward():
    model = FrozenAModel.empty().update_games([source("earned-before", [transition(score=100, final=97)])])
    assert all(row.counts == ((-3, 1),) for row in model.rows)
    invalid = transition(score=100, final=110)
    invalid["return_to_go"] = 110
    with pytest.raises(ValueError, match="already earned"):
        model.update_games([source("bad-return", [invalid])])
    invalid = transition()
    invalid["score_before"] = 99
    invalid["return_to_go"] = 11
    with pytest.raises(ValueError, match="actual observed score"):
        model.update_games([source("bad-observation", [invalid])])


def test_model_updates_are_atomic_immutable_and_duplicate_game_ids_are_rejected():
    empty = FrozenAModel.empty()
    model = empty.update_games([source("one")], batch_id="batch-1")
    digest = model.artifact_sha256
    with pytest.raises(ValueError, match="duplicate update"):
        model.update_games([source("one")])
    with pytest.raises(ValueError, match="duplicate update"):
        model.update_games([source("two"), source("one")])
    with pytest.raises(FrozenInstanceError):
        model.rows = ()
    data = model.to_dict()
    data["rows"][0]["counts"][0][1] += 9
    assert model.artifact_sha256 == digest and empty.training_games == 0
    assert model.to_dict()["source_batches"][0]["parent_artifact_sha256"] == empty.artifact_sha256


@pytest.mark.parametrize("purpose", ["evaluation", "crossplay", "diagnostic"])
def test_nontraining_sources_cannot_update_model(purpose):
    game = source("not-training")
    game["purpose"] = purpose
    with pytest.raises(ValueError, match="only train"):
        FrozenAModel.empty().update_games([game])


def test_incomplete_and_mismatched_focus_trajectories_are_refused():
    with pytest.raises(ValueError, match="complete"):
        FrozenAModel.empty().update_games([source("unfinished", [transition(done=False)])])
    bad = transition()
    bad["request"]["player_id"] = 1
    with pytest.raises(ValueError, match="another player"):
        FrozenAModel.empty().update_games([source("wrong-player", [bad])])
    with pytest.raises(ValueError, match="nonempty"):
        FrozenAModel.empty().update_games([])


def test_same_context_cannot_silently_pool_other_frozen_opponents():
    model = FrozenAModel.empty().update_games([source("one", opponents=["a0"] * 3)])
    with pytest.raises(ValueError, match="different opponent"):
        model.update_games([source("two", opponents=["different"] * 3)])
    updated = model.update_games([source("two", context="new-context", opponents=["different"] * 3)])
    assert updated.summary()["contexts"] == 2


def test_thompson_uses_score_posterior_owned_rng_and_does_not_fabricate_action_probability():
    model = FrozenAModel.empty()
    encoded = encode_request(request())
    before = random.getstate()
    first = model.select("opponents-a0", encoded, mode="thompson", rng=random.Random(41))
    second = model.select("opponents-a0", encoded, mode="thompson", rng=random.Random(41))
    assert first == second and first.choice in encoded.actions
    assert random.getstate() == before
    assert first.behavior_probability is None
    assert first.probability_source == "thompson_history_mixture_integral_not_computed"
    assert first.audit["behavior_probability_lower_bound"] == model.config.exploration_floor / len(encoded.actions)
    assert len(first.audit["estimates"]) == len(encoded.actions)
    assert any(estimate["value"] != estimate["mean"] for estimate in first.audit["estimates"])
    detached = first.audit
    detached["selected_index"] = -1
    assert first.audit["selected_index"] >= 0


def test_mean_policy_picks_expected_score_and_records_exact_tie_probability():
    encoded = encode_request(request())
    empty = FrozenAModel.empty()
    prior = empty.select("opponents-a0", encoded, mode="mean", rng=random.Random(3))
    assert prior.behavior_probability == 1 / len(encoded.actions)
    model = empty.update_games([source("bad", [transition(selected=0, final=80)]),
                                source("good", [transition(selected=1, final=120)])])
    selected = model.select("opponents-a0", encoded, mode="mean", rng=random.Random(3))
    assert selected.index == 1
    assert selected.behavior_probability == pytest.approx(1 - model.config.exploration_floor + model.config.exploration_floor / len(encoded.actions))
    assert selected.audit["estimates"][1]["mean"] > selected.audit["estimates"][0]["mean"]


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("algorithm", "beta-bernoulli"),
    ("objective", "championship_probability"), ("feature_version", "unknown"),
    ("visit_rule", "every_visit"), ("return_definition", "repeat_current_points"),
    ("training_games", 99), ("training_game_ids", "one"),
])
def test_artifact_rejects_schema_algorithm_reward_and_count_changes(field, value):
    data = FrozenAModel.empty().update_games([source("one")]).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        FrozenAModel.from_dict(data)


def test_artifact_roundtrip_preserves_predictions_counts_and_source_lineage():
    model = FrozenAModel.empty().update_games([source("one")], batch_id="batch1")
    model = model.update_games([source("two")], batch_id="batch2")
    restored = FrozenAModel.from_bytes(model.to_bytes())
    assert restored == model and restored.to_bytes() == model.to_bytes()
    encoded = encode_request(request())
    assert restored.select("opponents-a0", encoded, mode="thompson", rng=random.Random(9)) == model.select(
        "opponents-a0", encoded, mode="thompson", rng=random.Random(9))
    with pytest.raises(ValueError, match="duplicate model JSON"):
        FrozenAModel.from_bytes('{"schema_version":1,"schema_version":1}')
    data = model.to_dict()
    data["unknown_future_state"] = [6, 6]
    with pytest.raises(ValueError, match="fields"):
        FrozenAModel.from_dict(data)


@pytest.mark.parametrize("config", [
    {"prior_atoms": ()}, {"prior_atoms": ((0, 0),)}, {"prior_atoms": ((True, 1),)},
    {"prior_atoms": ((0, float("nan")),)}, {"prior_atoms": ((0, 1), (0, 2))},
    {"min_samples": True}, {"min_samples": 0},
])
def test_invalid_prior_and_sample_threshold_do_not_enter_a_snapshot(config):
    with pytest.raises(ValueError):
        AConfig(**config)


def test_posterior_rows_reject_noninteger_scores_counts_and_impossible_source_counts():
    with pytest.raises(ValueError):
        PosteriorRow("ctx", "state", "action", ((1.5, 1),))
    with pytest.raises(ValueError):
        PosteriorRow("ctx", "state", "action", ((1, True),))
    model = FrozenAModel.empty().update_games([source("one")])
    data = model.to_dict()
    data["rows"][0]["counts"][0][1] = 2
    with pytest.raises(ValueError, match="source games"):
        FrozenAModel.from_dict(data)


def test_actual_model_a_policies_play_official_four_races_and_keep_every_decision_replayable():
    model = FrozenAModel.empty()
    shared = {seat: SharedPublicContext() for seat in range(4)}
    game_policies = {seat: AGamePolicy(model, "opponents-a0", 100 + seat, shared_context=shared[seat]) for seat in range(4)}
    race_policies = {seat: ARacePolicy(model, "opponents-a0", 200 + seat, shared_context=shared[seat]) for seat in range(4)}
    game = GameEngine(official_config(), 5, game_policies, race_policies)
    result = game.run()
    assert result.official and len(result.races) == 4
    trace = game.trace()
    assert replay_game_trace(trace).trace() == trace
    assert all(shared[seat].race_index == 4 for seat in range(4))
    assert sum(len(policy.record_audits) for policy in game_policies.values()) == len(trace["decisions"]) == 32
    assert sum(len(policy.record_audits) for policy in race_policies.values()) == sum(len(raw["decisions"]) for raw in trace["race_traces"])
    assert all(trace["game_policies"][str(seat)]["seed"] == 100 + seat for seat in range(4))
    records = build_game_records(trace)
    updated = model.update_games([source("actual-official-game", records["transitions"])])
    assert updated.training_games == 1 and updated.summary()["state_action_rows"] > 0
    assert updated.artifact_sha256 != model.artifact_sha256
    assert model.training_games == 0  # The actual behavior snapshot stayed frozen.


def test_race_policy_requires_public_race_context_and_rejects_unsupported_mode():
    policy = ARacePolicy(FrozenAModel.empty(), "ctx", 0)
    with pytest.raises(ValueError, match="public race index"):
        policy.choose(None)
    with pytest.raises(ValueError, match="1..4"):
        policy.set_race_index(5)
    with pytest.raises(ValueError, match="mode"):
        FrozenAModel.empty().select("ctx", encode_request(request()), mode="win_rate", rng=random.Random(0))


@pytest.mark.parametrize("config", [
    {"exploration_floor": True}, {"exploration_floor": -0.01}, {"exploration_floor": 1},
    {"exploration_floor": float("nan")}, {"exploration_floor": float("inf")},
    {"repetition_limit": True}, {"repetition_limit": 0},
    {"history_capacity": True}, {"history_capacity": 2, "repetition_limit": 3},
    {"prior_atoms": ((0, 1e308), (1, 1e308))},
])
def test_policy_parameters_and_total_pseudo_mass_are_strict_finite_contracts(config):
    with pytest.raises(ValueError):
        AConfig(**config)


def test_v1_model_payload_cannot_be_silently_loaded_as_changed_v3_policy():
    data = FrozenAModel.empty().to_dict()
    data["schema_version"] = 1
    data["algorithm"] = "mc-dirichlet-thompson-v1"
    with pytest.raises(ValueError, match="schema"):
        FrozenAModel.from_dict(data)


@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_historical_policy_metadata_keeps_its_identity_but_only_current_behavior_can_execute(version):
    from magical_athlete.policy_runtime import require_implemented

    algorithm = "mc-dirichlet-thompson-" + version
    frozen = FrozenAModel.empty()
    snapshot = PolicySnapshot("A", "model_artifact", "metadata-" + version,
                              canonical_json({"algorithm": algorithm, "feature_version": "model-a-observable-groups-v2", "mode": "mean"}),
                              artifact_sha256=frozen.artifact_sha256)
    restored = PolicySnapshot.from_dict(snapshot.to_dict())
    assert restored.snapshot_id == snapshot.snapshot_id and restored.to_dict() == snapshot.to_dict()
    assert restored.game_policy_id == "mc-thompson-a-game-" + version
    assert restored.race_policy_id == "mc-thompson-a-race-" + version
    if algorithm == ALGORITHM:
        require_implemented(restored)
    else:
        with pytest.raises(NotImplementedError):
            require_implemented(restored)


def test_schema2_v2_artifact_is_rejected_instead_of_reinterpreted_with_v3_history():
    historical = FrozenAModel.empty().to_dict()
    historical["algorithm"] = "mc-dirichlet-thompson-v2"
    assert historical["schema_version"] == 2
    with pytest.raises(ValueError, match="algorithm"):
        FrozenAModel.from_bytes(canonical_json(historical))


def test_every_compared_action_uses_one_common_observable_group_level():
    config = AConfig(exploration_floor=0)
    games = [source(f"fine-a-{i}", [transition(score=100, final=120, selected=0)]) for i in range(3)]
    games += [source(f"neighbor-b-{i}", [transition(score=101, final=131, selected=1)]) for i in range(3)]
    model = FrozenAModel.empty(config).update_games(games)
    encoded = encode_request(request(score=100))
    assert model.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[0]).level == 0
    assert model.estimate("opponents-a0", encoded.state_keys, encoded.action_keys[1]).level == 1
    selected = model.select("opponents-a0", encoded, mode="mean", rng=random.Random(2))
    assert selected.audit["common_level"] == 1
    assert {estimate["level"] for estimate in selected.audit["estimates"]} == {1}
    assert selected.index == 1


def test_cached_artifact_is_the_exact_canonical_model_bytes_and_reuses_immutable_buffer():
    empty = FrozenAModel.empty()
    updated = empty.update_games([source("cached-signed", [transition(final=97)])], batch_id="cached-batch")
    for model in (empty, updated):
        canonical = canonical_json(model.to_dict()).encode("utf-8")
        cached = model.to_bytes()
        assert isinstance(cached, bytes) and cached == canonical
        assert model.to_bytes() is cached
        assert model.artifact_sha256 == hashlib.sha256(canonical).hexdigest()
        assert model.artifact_sha256 is model.artifact_sha256
        assert "_canonical_bytes=" not in repr(model) and "_artifact_sha256=" not in repr(model)


def test_cached_model_detaches_nested_input_and_output_trees():
    original = FrozenAModel.empty().update_games([source("detached", opponents=["a0"] * 3)])
    input_tree = original.to_dict()
    restored = FrozenAModel.from_dict(input_tree)
    expected_tree = restored.to_dict()
    expected_bytes, expected_hash = restored.to_bytes(), restored.artifact_sha256
    input_tree["config"]["prior_atoms"][0][1] += 10
    input_tree["rows"][0]["counts"][0][1] += 10
    input_tree["source_batches"][0]["games"][0]["opponent_snapshot_ids"][0] = "changed-input"
    input_tree["source_batches"][0]["games"].clear()
    input_tree["training_game_ids"].clear()
    output_tree = restored.to_dict()
    output_tree["config"]["prior_atoms"].clear()
    output_tree["rows"][0]["counts"].clear()
    output_tree["source_batches"][0]["games"][0]["opponent_snapshot_ids"].clear()
    output_tree["source_batches"].clear()
    assert restored.to_dict() == expected_tree
    assert restored.to_bytes() is expected_bytes and restored.artifact_sha256 == expected_hash
    assert canonical_json(restored.to_dict()).encode("utf-8") == expected_bytes


def test_direct_mutable_constructor_containers_and_raw_batch_buffer_cannot_stale_cache():
    original = FrozenAModel.empty().update_games([source("raw-buffer", opponents=["a0"] * 3)])
    input_rows = list(original.rows)
    input_games = list(original.training_game_ids)
    input_buffer = bytearray(original.source_batches[0].encode("utf-8"))
    input_batches = [input_buffer]
    copied = FrozenAModel(original.config, input_rows, input_games, input_batches)
    expected = original.to_bytes()
    input_rows.clear()
    input_games.clear()
    input_batches.clear()
    input_buffer[:] = b"invalid caller mutation"
    assert isinstance(copied.source_batches, tuple) and all(type(raw) is str for raw in copied.source_batches)
    assert copied.to_bytes() == expected and copied.to_dict() == original.to_dict()
    assert copied.artifact_sha256 == original.artifact_sha256
    with pytest.raises(FrozenInstanceError):
        copied._canonical_bytes = b"not an artifact"
    with pytest.raises(FrozenInstanceError):
        copied._artifact_sha256 = "0" * 64
    with pytest.raises(TypeError):
        copied._index[("ctx", "state", "action")] = copied.rows[0]
    with pytest.raises(FrozenInstanceError):
        copied.rows[0].counts = ((9, 99),)


def test_each_real_update_and_dataclass_reconstruction_builds_its_own_cache():
    parent = FrozenAModel.empty().update_games([source("cache-parent")], batch_id="cache-parent-batch")
    parent_bytes, parent_hash = parent.to_bytes(), parent.artifact_sha256
    child = parent.update_games([source("cache-child", [transition(final=97)])], batch_id="cache-child-batch")
    assert child.to_bytes() is not parent_bytes
    assert child.to_bytes() != parent_bytes and child.artifact_sha256 != parent_hash
    assert parent.to_bytes() is parent_bytes and parent.artifact_sha256 == parent_hash
    assert child.training_games == 2 and parent.training_games == 1
    assert child.to_dict()["source_batches"][-1]["parent_artifact_sha256"] == parent_hash
    assert child.to_bytes() == canonical_json(child.to_dict()).encode("utf-8")
    reconstructed = replace(child)
    assert reconstructed == child and reconstructed.to_bytes() == child.to_bytes()
    assert reconstructed.to_bytes() is not child.to_bytes()
    assert reconstructed.artifact_sha256 == child.artifact_sha256


def test_noncanonical_input_format_retains_the_original_canonical_serialization_contract():
    model = FrozenAModel.empty().update_games([source("format-independent")])
    pretty_json = json.dumps(model.to_dict(), ensure_ascii=False, sort_keys=False, indent=4)
    assert pretty_json.encode("utf-8") != model.to_bytes()
    restored = FrozenAModel.from_bytes(pretty_json)
    assert restored.to_dict() == model.to_dict()
    assert restored.to_bytes() == model.to_bytes() and restored.artifact_sha256 == model.artifact_sha256
