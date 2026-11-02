"""Construct and audit real policies from immutable experiment snapshots."""

import json
from functools import lru_cache

from .decisions import KeepPolicy, RandomPolicy
from .experiment_spec import PolicySnapshot, canonical_json, content_hash
from .game import FirstLegalGamePolicy, RandomGamePolicy
from .model_a import ALGORITHM, AGamePolicy, ARacePolicy, FrozenAModel, SharedPublicContext
from .model_a_features import FEATURE_VERSION
from .model_a_deployment import snapshot_deployment, validate_lookup_context


def require_implemented(snapshot: PolicySnapshot) -> None:
    if snapshot.model_type == "baseline":
        return
    if snapshot.model_type != "A":
        raise NotImplementedError("A/B policies require their actual future policy implementations; baselines cannot impersonate them")
    parameters = json.loads(snapshot.parameters_json)
    expected = {"algorithm", "feature_version", "mode"}
    if "deployment" in parameters:
        expected.add("deployment")
        snapshot_deployment(snapshot)
    if (set(parameters) != expected
            or parameters["algorithm"] != ALGORITHM or parameters["feature_version"] != FEATURE_VERSION
            or parameters["mode"] not in {"thompson", "mean"}):
        raise NotImplementedError("this A algorithm/feature/mode is not executable in the current runtime; use its matching source archive")


def require_trainable_snapshot(snapshot: PolicySnapshot) -> None:
    """A reference-context policy may be a frozen opponent, never a learner."""
    require_implemented(snapshot)
    if snapshot_deployment(snapshot) is not None:
        raise ValueError("fixed-context deployed snapshots cannot supply training trajectories")


@lru_cache(maxsize=2)
def _decode_model(payload: bytes) -> FrozenAModel:
    # Models are immutable; random streams and decision history belong to each
    # newly constructed policy wrapper. Bound memory to two decoded artifacts.
    return FrozenAModel.from_bytes(payload)


def load_model(snapshot: PolicySnapshot, artifacts: dict[str, bytes]) -> FrozenAModel:
    require_implemented(snapshot)
    payload = artifacts[snapshot.artifact_sha256]
    if not isinstance(payload, bytes):
        raise ValueError("frozen model artifacts require immutable bytes")
    model = _decode_model(payload)
    if model.artifact_sha256 != snapshot.artifact_sha256 or model.training_games != snapshot.training_games:
        raise ValueError("model checkpoint content/progress differs from its frozen snapshot")
    deployment = snapshot_deployment(snapshot)
    if deployment is not None:
        validate_lookup_context(model, deployment["lookup_context_id"])
    return model


def snapshot_artifacts(store, spec) -> dict[str, bytes]:
    artifacts = {}
    for snapshot in spec.policies:
        require_implemented(snapshot)
        if snapshot.artifact_sha256:
            artifacts[snapshot.artifact_sha256] = store.read_artifact(snapshot.artifact_sha256)
            load_model(snapshot, artifacts)
    return artifacts


def opponent_context_id(spec, seat: int) -> str:
    # Relative clockwise seats are part of the condition; the learner's own
    # changing checkpoint is not an opponent. No actual version is pooled away.
    return content_hash({
        "protocol": "fixed-opponent-4A-v1", "feature_version": FEATURE_VERSION,
        "opponents_clockwise": [spec.policies[(seat + offset) % 4].snapshot_id for offset in (1, 2, 3)],
        "game_config": spec.to_dict()["game_config"], "rule_version": spec.rule_version,
    })


def build_policies(spec, seeds, artifacts):
    game_policies, race_policies = {}, {}
    models = {}
    for seat, snapshot in enumerate(spec.policies):
        require_implemented(snapshot)
        if snapshot.policy_kind == "random":
            game_policies[seat] = RandomGamePolicy(seeds.game_policy_seeds[seat])
            race_policies[seat] = RandomPolicy(seeds.race_policy_seeds[seat])
        elif snapshot.policy_kind == "first_legal":
            game_policies[seat] = FirstLegalGamePolicy()
            race_policies[seat] = KeepPolicy()
        else:
            if snapshot.artifact_sha256 not in models:
                models[snapshot.artifact_sha256] = load_model(snapshot, artifacts)
            model = models[snapshot.artifact_sha256]
            if model.training_games != snapshot.training_games:
                raise ValueError("model checkpoint content/progress differs from its frozen snapshot")
            shared = SharedPublicContext()
            mode = json.loads(snapshot.parameters_json)["mode"]
            context = opponent_context_id(spec, seat)
            deployment = snapshot_deployment(snapshot)
            if deployment is not None:
                validate_lookup_context(model, deployment["lookup_context_id"])
                context = deployment["lookup_context_id"]
            game_policies[seat] = AGamePolicy(model, context, seeds.game_policy_seeds[seat], mode, shared)
            race_policies[seat] = ARacePolicy(model, context, seeds.race_policy_seeds[seat], mode, shared)
            if deployment is not None:
                game_policies[seat].policy_id = snapshot.game_policy_id
                race_policies[seat].policy_id = snapshot.race_policy_id
    return game_policies, race_policies


def annotate_behavior(records, spec, seeds, artifacts):
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
        if snapshot.policy_kind == "random":
            expected = policy._random.choice(legal)
            probability, source = 1 / len(legal), "uniform_legal_baseline"
            exploration = "random"
        elif snapshot.policy_kind == "first_legal":
            expected = next((action for action in legal if isinstance(action, dict) and action["kind"] == "keep"), legal[0])
            probability, source = 1.0, "deterministic_baseline"
            exploration = "first_legal"
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
            decision["model_a_audit"] = audit
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
        if ("model_a_deployment" in transition
                and transition["model_a_deployment"] != current.get("model_a_deployment")):
            raise ValueError("transition model A deployment differs from its audited decision")
        for key in ("behavior_probability", "behavior_probability_source", "snapshot_id", "model_type",
                    "model_version", "opponent_snapshot_ids", "exploration", "model_a_audit", "model_a_deployment"):
            if key in current:
                transition[key] = current[key]
    return records
