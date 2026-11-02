"""Versioned policy guards and explicit frozen reference-table deployment."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from . import policy_runtime as _legacy
from .decisions import KeepPolicy, RandomPolicy
from .experiment_spec import (
    A_FIXED_CONTEXT_SCHEME, A_GUARDED_FIXED_CONTEXT_SCHEME, PolicySnapshot, canonical_json,
)
from .game import FirstLegalGamePolicy, RandomGamePolicy
from .model_a import ALGORITHM, AGamePolicy, ARacePolicy, SharedPublicContext
from .model_a_features import FEATURE_VERSION
from .model_a_deployment import snapshot_deployment, validate_lookup_context

opponent_context_id = _legacy.opponent_context_id


def _guard_runtime(scheme):
    """Only explicitly reviewed, content-bound guard implementations execute."""
    if scheme == "all-legal-exhaustion-release-v1":
        from . import model_a_repetition_guard_v1 as module
        policy_class = module.AllLegalExhaustionRacePolicy
    elif scheme == "public-turn-local-duel-keep-sweep-v2":
        from . import model_a_repetition_guard_v2 as module
        module.validate_base_implementation()
        policy_class = module.PublicDuelSweepRacePolicy
    elif scheme == "public-turn-local-duel-fixed-keep-uniform-retry-v3":
        from . import model_a_repetition_guard_v3 as module
        module.validate_base_implementation()
        policy_class = module.PublicDuelRetryRacePolicy
    elif scheme == "public-turn-local-duel-retry-fish-keep-v4":
        from . import model_a_repetition_guard_v4 as module
        module.validate_base_implementation()
        policy_class = module.PublicDuelFishKeepMeanRacePolicy
    else:
        raise NotImplementedError("unsupported repetition guard version")
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if module.GUARD_VERSION != scheme or actual != module.IMPORTED_IMPLEMENTATION_SHA256:
        raise ValueError("frozen repetition guard implementation differs from its actual imported source")
    return {"scheme": scheme, "implementation_sha256": actual}, policy_class


def repetition_guard_binding(scheme):
    """Return detached parameters for an explicit immutable plan/factory."""
    return dict(_guard_runtime(scheme)[0])


def _guard_binding(snapshot):
    parameters = json.loads(snapshot.parameters_json)
    if "repetition_guard" not in parameters:
        return None
    if snapshot.model_type != "A" or snapshot.policy_kind != "model_artifact":
        raise NotImplementedError("guard snapshots require an implemented A model artifact")
    expected = {"algorithm", "feature_version", "mode", "repetition_guard"}
    if "deployment" in parameters:
        expected.add("deployment")
        deployment = snapshot_deployment(snapshot)
        if deployment["scheme"] != A_GUARDED_FIXED_CONTEXT_SCHEME:
            raise NotImplementedError("guard deployment requires its explicit versioned source binding")
    if (snapshot.model_type != "A" or snapshot.policy_kind != "model_artifact"
            or set(parameters) != expected):
        raise NotImplementedError("guard snapshots require their exact versioned A parameters")
    binding = parameters["repetition_guard"]
    if not isinstance(binding, dict) or set(binding) != {"scheme", "implementation_sha256"}:
        raise NotImplementedError("unsupported repetition guard version")
    expected, _ = _guard_runtime(binding["scheme"])
    if binding != expected:
        raise ValueError("frozen repetition guard implementation differs from its actual imported source")
    if binding["scheme"] == "public-turn-local-duel-retry-fish-keep-v4" and parameters["mode"] != "mean":
        raise ValueError("V4 is a mean-only evaluation guard")
    return dict(binding)


def _legacy_model_validation_view(snapshot):
    # A detached schema-only view for the old immutable artifact decoder.
    # Never register it, derive context IDs from it, or use it as behavior ID.
    parameters = json.loads(snapshot.parameters_json)
    parameters.pop("repetition_guard")
    if "deployment" in parameters:
        parameters["deployment"] = {
            "scheme": A_FIXED_CONTEXT_SCHEME,
            "lookup_context_id": parameters["deployment"]["lookup_context_id"],
        }
    return replace(snapshot, parameters_json=canonical_json(parameters))


def require_implemented(snapshot):
    if snapshot.policy_kind == "first_legal_memory":
        from .first_legal_memory_v1 import require_snapshot
        return require_snapshot(snapshot)
    binding = _guard_binding(snapshot)
    return _legacy.require_implemented(snapshot if binding is None else _legacy_model_validation_view(snapshot))


def require_trainable_snapshot(snapshot):
    if snapshot.policy_kind == "first_legal_memory":
        raise ValueError("a frozen public-memory baseline cannot supply learner trajectories")
    binding = _guard_binding(snapshot)
    if binding is not None and binding["scheme"] == "public-turn-local-duel-retry-fish-keep-v4":
        raise ValueError("V4 mean-only evaluation guard cannot supply training trajectories")
    return _legacy.require_trainable_snapshot(snapshot if binding is None else _legacy_model_validation_view(snapshot))


def load_model(snapshot, artifacts):
    if snapshot.policy_kind == "first_legal_memory":
        raise ValueError("a public-memory baseline has no learned model artifact")
    binding = _guard_binding(snapshot)
    _verify_guarded_deployment_source(snapshot, artifacts)
    return _legacy.load_model(snapshot if binding is None else _legacy_model_validation_view(snapshot), artifacts)


def _verify_guarded_deployment_source(snapshot, artifacts):
    """Authenticate worker-visible provenance without a mutable registry token."""
    deployment = snapshot_deployment(snapshot)
    if deployment is None or deployment["scheme"] != A_GUARDED_FIXED_CONTEXT_SCHEME:
        return None
    source_id = deployment["source_snapshot_id"]
    raw = artifacts.get(source_id)
    if not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != source_id:
        raise ValueError("guarded source snapshot artifact is missing or its checksum differs")
    source = PolicySnapshot.from_dict(json.loads(raw))
    parameters = json.loads(source.parameters_json)
    actual = json.loads(snapshot.parameters_json)
    if (canonical_json(source.to_dict()).encode("utf-8") != raw or source.snapshot_id != source_id
            or source.model_type != "A" or source.policy_kind != "model_artifact" or not source.frozen
            or source.training_games != snapshot.training_games or source.training_games < 1
            or source.artifact_sha256 != snapshot.artifact_sha256
            or set(parameters) != {"algorithm", "feature_version", "mode", "repetition_guard"}
            or any(parameters[key] != actual[key] for key in ("algorithm", "feature_version", "repetition_guard"))):
        raise ValueError("guarded deployment source differs from its native frozen model/guard binding")
    require_implemented(source)
    return source


def _has_guard(spec):
    return any("repetition_guard" in json.loads(snapshot.parameters_json) for snapshot in spec.policies)


def _versioned_runtime(spec):
    return _has_guard(spec) or any(snapshot.policy_kind == "first_legal_memory" for snapshot in spec.policies)


def snapshot_artifacts(store, spec):
    if not _versioned_runtime(spec):
        return _legacy.snapshot_artifacts(store, spec)
    artifacts = {}
    for snapshot in spec.policies:
        require_implemented(snapshot)
        if snapshot.artifact_sha256:
            artifacts[snapshot.artifact_sha256] = store.read_artifact(snapshot.artifact_sha256)
            deployment = snapshot_deployment(snapshot)
            if deployment is not None and deployment["scheme"] == A_GUARDED_FIXED_CONTEXT_SCHEME:
                source_id = deployment["source_snapshot_id"]
                registered_source = store.read_snapshot(source_id)
                raw = store.read_artifact(source_id)
                if canonical_json(registered_source.to_dict()).encode("utf-8") != raw:
                    raise ValueError("registered guarded source differs from its content-addressed proof")
                artifacts[source_id] = raw
            load_model(snapshot, artifacts)
    return artifacts


def guard_snapshot(store, model, version, mode="thompson", *, guard_scheme="all-legal-exhaustion-release-v1"):
    binding = repetition_guard_binding(guard_scheme)
    if mode not in {"thompson", "mean"}:
        raise ValueError("guard snapshot mode must be thompson or mean")
    if guard_scheme == "public-turn-local-duel-retry-fish-keep-v4" and mode != "mean":
        raise ValueError("V4 is a mean-only evaluation guard")
    snapshot = PolicySnapshot(
        "A", "model_artifact", version,
        canonical_json({"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": mode,
                        "repetition_guard": binding}),
        frozen=True, artifact_sha256=store.put_artifact(model.to_bytes()), training_games=model.training_games,
    )
    require_implemented(snapshot)
    store.register_snapshot(snapshot)
    return snapshot


def build_policies(spec, seeds, artifacts):
    if not _versioned_runtime(spec):
        return _legacy.build_policies(spec, seeds, artifacts)
    game_policies, race_policies, models = {}, {}, {}
    for seat, snapshot in enumerate(spec.policies):
        require_implemented(snapshot)
        if snapshot.policy_kind == "random":
            game_policies[seat] = RandomGamePolicy(seeds.game_policy_seeds[seat])
            race_policies[seat] = RandomPolicy(seeds.race_policy_seeds[seat])
        elif snapshot.policy_kind == "first_legal":
            game_policies[seat] = FirstLegalGamePolicy()
            race_policies[seat] = KeepPolicy()
        elif snapshot.policy_kind == "first_legal_memory":
            from .first_legal_memory_v1 import build_pair
            game_policies[seat], race_policies[seat] = build_pair()
        else:
            if snapshot.artifact_sha256 not in models:
                models[snapshot.artifact_sha256] = load_model(snapshot, artifacts)
            model = models[snapshot.artifact_sha256]
            # One artifact may serve several distinct deployed snapshots. Each
            # seat still authenticates its own source and real lookup table.
            _verify_guarded_deployment_source(snapshot, artifacts)
            if model.training_games != snapshot.training_games:
                raise ValueError("model checkpoint progress differs from its frozen snapshot")
            shared = SharedPublicContext()
            mode = json.loads(snapshot.parameters_json)["mode"]
            context = opponent_context_id(spec, seat)
            deployment = snapshot_deployment(snapshot)
            if deployment is not None:
                validate_lookup_context(model, deployment["lookup_context_id"])
                context = deployment["lookup_context_id"]
            game_policies[seat] = AGamePolicy(model, context, seeds.game_policy_seeds[seat], mode, shared)
            binding = _guard_binding(snapshot)
            policy_class = _guard_runtime(binding["scheme"])[1] if binding is not None else ARacePolicy
            race_policies[seat] = policy_class(model, context, seeds.race_policy_seeds[seat], mode, shared)
            if deployment is not None:
                game_policies[seat].policy_id = snapshot.game_policy_id
                race_policies[seat].policy_id = snapshot.race_policy_id
    return game_policies, race_policies


def annotate_behavior(records, spec, seeds, artifacts):
    if not _versioned_runtime(spec):
        return _legacy.annotate_behavior(records, spec, seeds, artifacts)
    return _annotate_guarded_behavior(records, spec, seeds, artifacts)


def _annotate_guarded_behavior(records, spec, seeds, artifacts):
    # Reconstruct the actual frozen policy, including its public-history
    # memory. Repeating a stateless model.select call would miss a guarded
    # choice and could incorrectly quarantine a fully legal real game.
    game_policies, race_policies = build_policies(spec, seeds, artifacts)
    decisions = {}
    for decision in records["decisions"]:
        seat, legal = decision["player_id"], decision["legal_actions"]
        snapshot = spec.policies[seat]
        policy = (race_policies if decision["scope"] == "race" else game_policies)[seat]
        audit = None
        baseline_audit = None
        if snapshot.policy_kind == "random":
            expected = policy._random.choice(legal)
            probability, source = 1 / len(legal), "uniform_legal_baseline"
            exploration = "random"
        elif snapshot.policy_kind == "first_legal":
            expected = next((action for action in legal if isinstance(action, dict) and action["kind"] == "keep"), legal[0])
            probability, source = 1.0, "deterministic_baseline"
            exploration = "first_legal"
        elif snapshot.policy_kind == "first_legal_memory":
            if "model_a_audit" in decision:
                raise ValueError("a public-memory baseline cannot declare a model A audit")
            if decision["scope"] == "race" and policy.race_index != decision["race_index"]:
                raise ValueError("public-memory Race request precedes its own public entry context")
            expected = json.loads(canonical_json(policy.choose(decision["request"])))
            baseline_audit = json.loads(canonical_json(policy.last_audit))
            probability, source = 1.0, "deterministic_public_memory_baseline"
            exploration = "first_legal_memory"
        else:
            exploration = json.loads(snapshot.parameters_json)["mode"]
            if decision["scope"] == "race":
                # Race requests intentionally lack race_index; the entry
                # decision already revealed it to that same player.
                if policy.shared_context.race_index != decision["race_index"]:
                    raise ValueError("race request precedes its own public entry context")
            expected = json.loads(canonical_json(policy.choose(decision["request"])))
            audit = policy.record_audits[-1]
            probability = audit["behavior_probability"]
            source = audit["behavior_probability_source"]
        if decision["action"] != expected:
            raise ValueError("actual action differs from its seeded/fixed behavior snapshot")
        decision.update({
            "behavior_probability": probability, "behavior_probability_source": source,
            "snapshot_id": snapshot.snapshot_id, "model_type": snapshot.model_type,
            "model_version": snapshot.model_version,
            "opponent_snapshot_ids": [policy.snapshot_id for other, policy in enumerate(spec.policies) if other != seat],
            "exploration": {"mode": exploration, "learning": spec.learning},
        })
        # Baseline v1 rows keep their existing shape. A rows retain the actual
        # posterior values, evidence level and chosen action, independently
        # reconstructed from their frozen artifact and controlled seed.
        if audit is not None:
            if "model_a_audit" in decision and decision["model_a_audit"] != audit:
                raise ValueError("recorded guarded/mixed audit differs from its frozen behavior")
            decision["model_a_audit"] = audit
        if baseline_audit is not None:
            if "baseline_memory_audit" in decision and canonical_json(decision["baseline_memory_audit"]) != canonical_json(baseline_audit):
                raise ValueError("recorded public-memory baseline audit differs from its frozen behavior")
            decision["baseline_memory_audit"] = baseline_audit
        elif "baseline_memory_audit" in decision:
            raise ValueError("undeclared public-memory baseline audit")
        deployment = snapshot_deployment(snapshot)
        if deployment is not None:
            declared = {**deployment, "actual_context_id": opponent_context_id(spec, seat),
                        "actual_opponent_snapshot_ids": [spec.policies[(seat + offset) % 4].snapshot_id
                                                         for offset in (1, 2, 3)],
                        "lookup_context_observed": True, "training_eligible": False}
            if "model_a_deployment" in decision and decision["model_a_deployment"] != declared:
                raise ValueError("recorded model A deployment differs from its frozen binding")
            decision["model_a_deployment"] = declared
        elif "model_a_deployment" in decision:
            raise ValueError("undeclared model A deployment metadata in an actual-context policy")
        decisions[decision["decision_seq"]] = decision
    for transition in records["transitions"]:
        current = decisions[transition["decision_seq"]]
        if ("baseline_memory_audit" in transition
                and canonical_json(transition["baseline_memory_audit"]) != canonical_json(current.get("baseline_memory_audit"))):
            raise ValueError("transition public-memory baseline audit differs from its audited decision")
        if ("model_a_deployment" in transition
                and transition["model_a_deployment"] != current.get("model_a_deployment")):
            raise ValueError("transition model A deployment differs from its audited decision")
        for key in ("behavior_probability", "behavior_probability_source", "snapshot_id", "model_type",
                    "model_version", "opponent_snapshot_ids", "exploration", "model_a_audit", "model_a_deployment", "baseline_memory_audit"):
            if key in current:
                transition[key] = current[key]
    return records
