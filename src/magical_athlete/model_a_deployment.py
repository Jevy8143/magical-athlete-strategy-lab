"""Explicit read-only deployment of one learned A context into new matchups.

The actual lineup never changes its experiment context. A separate, frozen
lookup binding names the real historical table used by the deployed policy.
It is a deployment experiment, not pooled TRAIN data or proof of generality.
"""

import json
import re

from .experiment_spec import (
    A_FIXED_CONTEXT_SCHEME, A_GUARDED_FIXED_CONTEXT_SCHEME, PolicySnapshot, canonical_json,
)
from .model_a import ALGORITHM, FrozenAModel
from .model_a_features import FEATURE_VERSION


def snapshot_deployment(snapshot):
    """Return the validated binding, or None for legacy actual-context A."""
    if snapshot.model_type != "A":
        return None
    parameters = json.loads(snapshot.parameters_json)
    if "deployment" not in parameters:
        return None
    value = parameters["deployment"]
    if not isinstance(value, dict):
        raise ValueError("invalid frozen A deployment binding")
    guarded = value.get("scheme") == A_GUARDED_FIXED_CONTEXT_SCHEME
    expected = {"scheme", "lookup_context_id", "source_snapshot_id"} if guarded else {"scheme", "lookup_context_id"}
    if (set(value) != expected
            or value.get("scheme") not in {A_FIXED_CONTEXT_SCHEME, A_GUARDED_FIXED_CONTEXT_SCHEME}
            or not isinstance(value["lookup_context_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["lookup_context_id"])
            or not snapshot.frozen or snapshot.training_games < 1):
        raise ValueError("invalid frozen A deployment binding")
    if guarded and (set(parameters) != {"algorithm", "feature_version", "mode", "repetition_guard", "deployment"}
                    or not isinstance(value["source_snapshot_id"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value["source_snapshot_id"])):
        raise ValueError("invalid guarded frozen A deployment source binding")
    return dict(value)


def validate_lookup_context(model, lookup_context_id):
    if (not isinstance(model, FrozenAModel) or model.training_games < 1
            or not isinstance(lookup_context_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", lookup_context_id)
            or not any(row.context_id == lookup_context_id for row in model.rows)):
        raise ValueError("fixed-context deployment requires an actually observed trained lookup context")


def fixed_context_snapshot(store, model, version, lookup_context_id, mode="mean"):
    """Register new deployment identity without modifying frozen model bytes."""
    validate_lookup_context(model, lookup_context_id)
    if mode not in {"mean", "thompson"}:
        raise ValueError("fixed-context mode must be mean or thompson")
    raw = model.to_bytes()
    digest = store.put_artifact(raw)
    snapshot = PolicySnapshot(
        "A", "model_artifact", version,
        canonical_json({"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": mode,
                        "deployment": {"scheme": A_FIXED_CONTEXT_SCHEME,
                                       "lookup_context_id": lookup_context_id}}),
        frozen=True, artifact_sha256=digest, training_games=model.training_games,
    )
    store.register_snapshot(snapshot)
    return snapshot


def guarded_fixed_context_snapshot(store, model, version, lookup_context_id, source_snapshot, mode=None):
    """Bind a registered native guarded policy to one real learned table.

    The source policy is retained verbatim as a content-addressed artifact so
    execution can authenticate its guard/weights independently of this factory.
    An explicit mode override creates a different deployment identity while
    preserving the identity and original mode of the source policy.
    """
    # The router imports this module; defer the validation import until a
    # factory call, after both modules have finished loading.
    from .policy_runtime_router import load_model, require_implemented

    validate_lookup_context(model, lookup_context_id)
    if not isinstance(source_snapshot, PolicySnapshot):
        raise ValueError("guarded deployment requires a registered native guarded source snapshot")
    source_parameters = json.loads(source_snapshot.parameters_json)
    if (source_snapshot.model_type != "A" or source_snapshot.policy_kind != "model_artifact"
            or not source_snapshot.frozen or source_snapshot.training_games < 1
            or set(source_parameters) != {"algorithm", "feature_version", "mode", "repetition_guard"}):
        raise ValueError("guarded deployment requires a registered native guarded source snapshot")
    require_implemented(source_snapshot)
    if store.read_snapshot(source_snapshot.snapshot_id) != source_snapshot:
        raise ValueError("registered guarded source snapshot differs from the requested source")
    raw = model.to_bytes()
    if (source_snapshot.artifact_sha256 != model.artifact_sha256
            or source_snapshot.training_games != model.training_games
            or store.read_artifact(source_snapshot.artifact_sha256) != raw):
        raise ValueError("guarded source snapshot weights or training progress differ from the frozen model")
    source_model = load_model(source_snapshot, {source_snapshot.artifact_sha256: raw})
    if source_model.to_bytes() != raw or source_model.config != model.config:
        raise ValueError("guarded source model bytes or configuration differ")
    mode = source_parameters["mode"] if mode is None else mode
    if mode not in {"mean", "thompson"}:
        raise ValueError("guarded fixed-context mode must be mean or thompson")
    source_bytes = canonical_json(source_snapshot.to_dict()).encode("utf-8")
    if store.put_artifact(source_bytes) != source_snapshot.snapshot_id:
        raise ValueError("guarded source snapshot artifact differs from its immutable identity")
    snapshot = PolicySnapshot(
        "A", "model_artifact", version,
        canonical_json({"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION, "mode": mode,
                        "repetition_guard": source_parameters["repetition_guard"],
                        "deployment": {"scheme": A_GUARDED_FIXED_CONTEXT_SCHEME,
                                       "lookup_context_id": lookup_context_id,
                                       "source_snapshot_id": source_snapshot.snapshot_id}}),
        frozen=True, artifact_sha256=source_snapshot.artifact_sha256, training_games=model.training_games,
    )
    require_implemented(snapshot)
    store.register_snapshot(snapshot)
    return snapshot
