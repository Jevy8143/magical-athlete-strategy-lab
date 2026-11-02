"""Strict current-engine semantic certification of immutable TRAIN evidence.

An archive is authenticated, never executed. Every original source Game is
executed by the current engine using its original frozen policies and seeds.
Only Game/Race fingerprint stamps may differ. Original identities and TRAIN
records remain unchanged; the additional attempts are separate DIAGNOSTICs.
"""

import ast
import hashlib
import json
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .engine import RaceEngine
from .experiment_spec import RunSpec, canonical_json, content_hash
from .experiment_store import ExperimentError, _write_immutable, infrastructure_fingerprint, runtime_identity
from .experiments import _simulate_one
from .game import GameEngine, replay_game_trace
from .model_a import FrozenAModel
from .policy_runtime_router import opponent_context_id, require_trainable_snapshot, snapshot_artifacts


CERTIFICATE_KIND = "model-a-current-source-semantics-v1"
DIAGNOSTIC_KIND = "model-a-source-semantic-revalidation-game-v1"
_PARAMETERS = "all original TRAIN batches reconstructed by current pure MC; frozen checkpoint bytes equal"
_EQUIVALENCE = "all same-seed frozen-policy complete traces equal except Game/Race code_fingerprint stamps"
_REPLAY_CACHE = OrderedDict()
_DERIVED_RECORD_CACHE = OrderedDict()
_RECONSTRUCTION_CACHE = OrderedDict()
_CACHE_LIMITS = {"replay": 4096, "derived": 4096, "reconstruction": 128}
_ATTEMPT_COMPARISON_CACHE_LIMIT = 4096


def _cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache, key, value, limit):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def _loaded_json_bytes(value):
    """Canonicalize JSON already parsed from bytes, without typed conversion.

    Callers use this only for json.loads results and their five permitted
    fingerprint-stamp replacements, never for model/features/dataclasses.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


class _Verification:
    """Fresh physical scans, with transactionally published deterministic proofs.

    No actual artifact bytes or mutable model/policy instances cross public
    verifications. A cache hit requires a fresh complete source descriptor;
    each pass still reads every distinct artifact and every original row.
    Only a successful complete public verification publishes pending proofs.
    """

    def __init__(self, store, origins, current):
        self.store = store
        self.namespace = (str(store.root.resolve()), str(store.database.resolve()),
                          content_hash(origins), content_hash(current))
        self.pending = {"derived": {}, "reconstruction": {}, "replay": {}}
        # Successful exact-input comparisons stay inside this one verification.
        # No trace bytes, mutable JSON, or trust token is retained by this memo.
        self.attempt_comparisons = OrderedDict()
        self.begin_pass()

    def begin_pass(self):
        self.artifacts = {}
        self.snapshot_artifact_sets = {}
        self.decoded_models = {}

    def read_model_artifact(self, digest):
        if digest not in self.artifacts:
            self.artifacts[digest] = self.store.read_artifact(digest)
        return self.artifacts[digest]

    def snapshot_artifacts(self, spec):
        identity = tuple(policy.snapshot_id for policy in spec.policies)
        if identity not in self.snapshot_artifact_sets:
            # The existing runtime performs all policy/schema/model checks.
            # Its read facade deduplicates the same immutable artifact only
            # inside this one fresh physical scan, never across checks.
            facade = SimpleNamespace(read_artifact=self.read_model_artifact)
            if hasattr(self.store, "read_snapshot"):
                facade.read_snapshot = self.store.read_snapshot
            self.snapshot_artifact_sets[identity] = snapshot_artifacts(facade, spec)
        return self.snapshot_artifact_sets[identity]

    def model(self, digest):
        raw = self.read_model_artifact(digest)
        if digest not in self.decoded_models:
            model = FrozenAModel.from_bytes(raw)
            _require(model.artifact_sha256 == digest and model.to_bytes() == raw,
                     "reconstructed checkpoint is not its canonical model artifact")
            self.decoded_models[digest] = model
        return self.decoded_models[digest]

    def get(self, kind, key):
        key = (self.namespace, key)
        if key in self.pending[kind]:
            return self.pending[kind][key]
        cache = {"derived": _DERIVED_RECORD_CACHE, "reconstruction": _RECONSTRUCTION_CACHE,
                 "replay": _REPLAY_CACHE}[kind]
        return _cache_get(cache, key)

    def stage(self, kind, key, value):
        self.pending[kind][(self.namespace, key)] = value

    def commit(self):
        for kind, cache in (("derived", _DERIVED_RECORD_CACHE),
                            ("reconstruction", _RECONSTRUCTION_CACHE), ("replay", _REPLAY_CACHE)):
            for key, value in self.pending[kind].items():
                _cache_put(cache, key, value, _CACHE_LIMITS[kind])


def _require(condition, message):
    if not condition:
        raise ExperimentError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid source-semantics SHA256")
    return value


def _source_bundle():
    package = Path(__file__).resolve().parent
    root = package.parent.parent
    paths = sorted(package.glob("*.py")) + [root / "pyproject.toml"]
    files = {}
    for path in paths:
        raw = path.read_bytes()
        files[path.relative_to(root).as_posix()] = {"sha256": hashlib.sha256(raw).hexdigest(), "utf8": raw.decode("utf-8")}
    return {"kind": "magical-athlete-source-v2", "files": files,
            "engine_fingerprint": GameEngine._code_fingerprint(),
            "infrastructure_fingerprint": infrastructure_fingerprint()}


def _current_identity(store, archive_sha=None):
    bundle = _source_bundle()
    raw = canonical_json(bundle).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    if archive_sha is not None:
        _require(archive_sha == digest and store.read_artifact(archive_sha) == raw,
                 "certificate current source archive differs from actual current source bytes")
    return {"game_fingerprint": bundle["engine_fingerprint"], "race_fingerprint": RaceEngine._code_fingerprint(),
            "infrastructure_fingerprint": bundle["infrastructure_fingerprint"], "source_archive_sha256": digest,
            "runtime": runtime_identity()}


def _archived_fingerprint(bundle, filename, function_name, class_name=None):
    """Interpret only the static file-name list; never import archived code."""
    prefix = "src/magical_athlete/"
    try:
        body = ast.parse(bundle["files"][prefix + filename]["utf8"]).body
        if class_name is not None:
            owner, = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]
            body = owner.body
        function, = [node for node in body if isinstance(node, ast.FunctionDef) and node.name == function_name]
        loop, = [node for node in function.body if isinstance(node, ast.For)]
        names = ast.literal_eval(loop.iter)
        _require(isinstance(names, (tuple, list)) and names
                 and all(isinstance(name, str) and re.fullmatch(r"[a-z_][a-z0-9_]*\.py", name) for name in names),
                 "archive fingerprint list is not a static supported file list")
        result = hashlib.sha256()
        for name in names:
            result.update(name.encode("utf-8"))
            result.update(bundle["files"][prefix + name]["utf8"].encode("utf-8"))
        return result.hexdigest()
    except (KeyError, ValueError, SyntaxError, TypeError) as error:
        raise ExperimentError("unsupported archived fingerprint definition") from error


def _origins(store, identities):
    # Import at call time: evaluation_a imports training_a, whose optional
    # certification branch imports this module after normal initialization.
    from .evaluation_a import REVIEWED_SCHEMA_BRIDGE, _verified_origin_archive

    _require(isinstance(identities, (tuple, list)) and identities and len(set(identities)) == len(identities),
             "source certification requires distinct explicit origin archives")
    result = []
    for identity in identities:
        _sha(identity)
        verified = _verified_origin_archive(store, identity, schema_bridge=REVIEWED_SCHEMA_BRIDGE)
        bundle = json.loads(store.read_artifact(identity))
        infra = _archived_fingerprint(bundle, "experiment_store.py", "infrastructure_fingerprint")
        _require(bundle.get("infrastructure_fingerprint") == infra, "origin infrastructure archive fingerprint differs")
        origin = {"source_archive_sha256": identity, "game_fingerprint": verified["engine_fingerprint"],
                  "race_fingerprint": _archived_fingerprint(bundle, "engine.py", "_code_fingerprint", "RaceEngine"),
                  "infrastructure_fingerprint": infra,
                  "protected_policy_files_sha256": verified["protected_policy_files_sha256"]}
        if "protected_policy_schema_bridge" in verified:
            origin["protected_policy_schema_bridge"] = verified["protected_policy_schema_bridge"]
        result.append(origin)
    return sorted(result, key=lambda item: item["source_archive_sha256"])


def _current_trace_view(trace, current):
    """Replace only the explicitly permitted Game/Race code metadata."""
    value = json.loads(_loaded_json_bytes(trace))
    _require(value["completed"] and len(value["race_traces"]) == len(value["races"]) == 4
             and not value.get("active_race") and not value.get("race_diagnostic"),
             "semantic certification requires an actual complete four-race trace")
    value["code_fingerprint"] = current["game_fingerprint"]
    for race in value["race_traces"]:
        race["code_fingerprint"] = current["race_fingerprint"]
    return value


def _original_identity(spec, manifest):
    raw = manifest["spec"]
    return SimpleNamespace(policies=spec.policies, game_config=spec.game_config, rule_version=raw["rule_version"],
                           engine_fingerprint=raw["engine_fingerprint"], spec_id=manifest["spec_hash"],
                           run_id=spec.run_id, purpose=spec.purpose, matchup=spec.matchup,
                           seat_snapshot_ids=spec.seat_snapshot_ids, seed_plan=spec.seed_plan)


def _read_source(store, source, parent_digest, origins, current, manifests, artifact_pool, verification=None):
    _require(source["purpose"] == "train" and source["run_id"] is not None,
             "source semantics accepts only explicitly committed TRAIN data")
    run_id = source["run_id"]
    if run_id not in manifests:
        manifest = store.manifest(run_id)
        raw = manifest["spec"]
        original_fp = raw["engine_fingerprint"]
        origin = next((item for item in origins if item["game_fingerprint"] == original_fp
                       and item["infrastructure_fingerprint"] == manifest["infrastructure_fingerprint"]), None)
        _require((original_fp == current["game_fingerprint"]
                  and manifest["infrastructure_fingerprint"] == current["infrastructure_fingerprint"])
                 or origin is not None,
                 "TRAIN source has no authenticated origin/current physics")
        spec = RunSpec.from_dict({**raw, "engine_fingerprint": current["game_fingerprint"]})
        _require(spec.purpose.value == "train" and spec.learning
                 and {policy.model_type for policy in spec.policies} == {"A"},
                 "certified source is not an isolated learning 4A TRAIN run")
        manifests[run_id] = manifest, spec, origin
    manifest, spec, origin = manifests[run_id]
    raw = manifest["spec"]
    focus = source["focus_player_id"]
    _require(type(focus) is int and 0 <= focus < 4, "invalid certified source focus")
    behavior = spec.policies[focus]
    require_trainable_snapshot(behavior)
    _require(behavior.frozen and behavior.artifact_sha256 == parent_digest
             and behavior.snapshot_id == source["behavior_snapshot_id"]
             and json.loads(behavior.parameters_json).get("mode") == "thompson"
             and opponent_context_id(spec, focus) == source["context_id"]
             and [spec.policies[(focus + offset) % 4].snapshot_id for offset in (1, 2, 3)] == source["opponent_snapshot_ids"],
             "certified TRAIN behavior/context/opponents differ from retained lineage")
    rows = store._rows("SELECT * FROM games WHERE game_id=? AND run_id=?", (source["game_id"], run_id))
    _require(len(rows) == 1 and rows[0]["status"] == "completed" and rows[0]["replay_verified"] == 1,
             "certified source must be an original completed replay-verified TRAIN game")
    row = rows[0]
    _require(source["game_id"] == f"{manifest['spec_hash']}:{row['game_index']}", "original source GameID differs")
    records = store._load_records(row)
    _require(records["game"]["purpose"] == "train" and records["game"]["completed"] and records["transitions"],
             "certified source has no complete original TRAIN trajectory")
    trace_raw = store.read_artifact(row["trace_hash"])
    trace = json.loads(trace_raw)
    _require(_loaded_json_bytes(trace) == trace_raw, "original source trace is not canonical immutable JSON")
    seeds = store.seeds_for(run_id, row["game_index"])
    _require(seeds == spec.seed_plan.derive(row["game_index"]), "certified source reserved seeds differ")
    identity = _original_identity(spec, manifest)
    store._validate_trace_identity(identity, seeds, trace)
    _current_trace_view(trace, current)
    race_fp = current["race_fingerprint"] if raw["engine_fingerprint"] == current["game_fingerprint"] else origin["race_fingerprint"]
    _require(all(race["code_fingerprint"] == race_fp for race in trace["race_traces"]), "original Race source stamp differs")
    artifacts = (verification.snapshot_artifacts(spec) if verification is not None
                 else snapshot_artifacts(store, spec))
    transitions = [item for item in records["transitions"] if item["player_id"] == focus]
    _require(transitions and all("model_a_deployment" not in item for item in transitions),
             "certified source contains deployed/nontraining behavior")
    planned = store._rows("SELECT * FROM planned_games WHERE run_id=? AND game_index=?", (run_id, row["game_index"]))
    _require(len(planned) == 1, "certified source lacks its original seed reservation")
    run_rows = store._rows("SELECT * FROM runs WHERE run_id=?", (run_id,))
    claims = []
    labels = ("game", "dice", *(f"game_policy:{seat}" for seat in range(4)), *(f"race_policy:{seat}" for seat in range(4)))
    for seed, label in zip(seeds.all_seeds, labels):
        rows_claim = store._rows("SELECT * FROM seed_claims WHERE seed=?", (seed,))
        expected = {"seed": seed, "purpose": "train", "study_id": spec.study_id,
                    "game_index": row["game_index"], "stream": label}
        _require(len(rows_claim) == 1 and dict(rows_claim[0]) == expected, "source seed domain/claim differs")
        claims.append(expected)
    raw_record_hashes = {}
    for table, order in (("races", "race_index"), ("decisions", "decision_seq"), ("events", "race_index,sequence"),
                         ("score_entries", "sequence"), ("transitions", "decision_seq")):
        original_rows = store._rows(f"SELECT * FROM {table} WHERE game_id=? ORDER BY {order}", (source["game_id"],))
        raw_record_hashes[table] = content_hash([dict(value) for value in original_rows])
    snapshots = []
    for policy in spec.policies:
        registered = store._rows("SELECT * FROM snapshots WHERE snapshot_id=?", (policy.snapshot_id,))
        _require(len(registered) == 1, "certified source lacks a registered original snapshot")
        snapshots.append({"snapshot_id": policy.snapshot_id, "index_row_sha256": content_hash(dict(registered[0])),
                          "artifact_sha256": policy.artifact_sha256})
    descriptor = {"source": source, "game_index": row["game_index"], "original_spec_sha256": manifest["spec_hash"],
                  "original_spec": raw, "original_engine_fingerprint": raw["engine_fingerprint"],
                  "original_infrastructure_fingerprint": manifest["infrastructure_fingerprint"],
                  "original_race_fingerprint": race_fp, "origin_source_archive_sha256": origin["source_archive_sha256"] if origin else None,
                  "original_manifest_sha256": content_hash(manifest), "original_trace_sha256": row["trace_hash"],
                  "original_run_index_sha256": content_hash(dict(run_rows[0])),
                  "original_records_sha256": row["records_hash"], "original_summary_sha256": row["summary_hash"],
                  "original_game_index_sha256": content_hash(dict(row)), "planned_index_sha256": content_hash(dict(planned[0])),
                  "original_normalized_rows_sha256": raw_record_hashes, "seed_claims_sha256": content_hash(claims),
                  "seeds": seeds.to_dict(), "seed_sha256": content_hash(seeds.to_dict()), "snapshots": snapshots,
                  "model_artifacts": sorted(artifacts)}
    # Every input above was physically reread and checked in this pass. Only
    # the deterministic policy/record derivation may be reused, never records
    # read from the DB or a certificate's claim that they were validated.
    derived_key = content_hash(descriptor)
    expected = verification.get("derived", derived_key) if verification is not None else None
    if expected is None:
        rebuilt = store._records(trace, spec, seeds)
        rebuilt["game"] = store._summary(rebuilt, identity, seeds, row["game_index"], "completed")
        expected = content_hash(rebuilt)
        _require(expected == row["records_hash"], "original normalized payload differs from seeded behavior/trace")
        if verification is not None:
            verification.stage("derived", derived_key, expected)
    _require(expected == row["records_hash"], "original normalized payload differs from seeded behavior/trace")
    shared_artifacts = {}
    for digest, value in artifacts.items():
        shared = artifact_pool.setdefault(digest, value)
        _require(shared == value, "source model artifact changed within verification")
        shared_artifacts[digest] = shared
    return {"descriptor": descriptor, "spec": spec, "artifacts": shared_artifacts,
            "game": {**source, "transitions": transitions}}


def _collect(store, models, origins, current, verification=None):
    verification = verification or _Verification(store, origins, current)
    verification.begin_pass()
    games, manifests, configs, models_seen, context, artifact_pool = {}, {}, {}, {}, None, {}
    lineages = {}
    for model in models:
        original_bytes = verification.read_model_artifact(model.artifact_sha256)
        _require(original_bytes == model.to_bytes(), "frozen model bytes differ from their canonical artifact")
        rebuilt = FrozenAModel.empty(model.config)
        models_seen[rebuilt.artifact_sha256] = rebuilt.summary()
        lineage = [rebuilt.artifact_sha256]
        chain = content_hash({"empty_artifact_sha256": rebuilt.artifact_sha256})
        for raw_batch in model.source_batches:
            batch = json.loads(raw_batch)
            _require(batch["parent_artifact_sha256"] == rebuilt.artifact_sha256
                     and verification.read_model_artifact(rebuilt.artifact_sha256) == rebuilt.to_bytes(),
                     "certified model cumulative parent chain differs")
            inputs = []
            for source in batch["games"]:
                key = source["game_id"]
                if key not in games:
                    games[key] = _read_source(store, source, rebuilt.artifact_sha256, origins, current,
                                              manifests, artifact_pool, verification)
                item = games[key]
                _require(item["descriptor"]["source"] == source
                         and item["spec"].policies[source["focus_player_id"]].artifact_sha256 == rebuilt.artifact_sha256,
                         "one certified GameID has conflicting source lineage")
                source_context = (source["context_id"], tuple(source["opponent_snapshot_ids"]))
                context = source_context if context is None else context
                _require(source_context == context, "source certification cannot pool different exact opponent contexts")
                configs[canonical_json(item["spec"].game_config)] = item["spec"].game_config
                inputs.append(item["game"])
            chain = content_hash({"prefix_binding_sha256": chain, "batch": batch,
                                  "source_bindings": [content_hash(games[source["game_id"]]["descriptor"])
                                                      for source in batch["games"]]})
            proven_digest = verification.get("reconstruction", chain)
            if proven_digest is None:
                rebuilt = rebuilt.update_games(inputs, batch_id=batch["batch_id"])
                _require(verification.read_model_artifact(rebuilt.artifact_sha256) == rebuilt.to_bytes(),
                         "source batches do not reconstruct a registered byte-identical checkpoint")
                verification.stage("reconstruction", chain, rebuilt.artifact_sha256)
            else:
                rebuilt = verification.model(proven_digest)
            models_seen[rebuilt.artifact_sha256] = rebuilt.summary()
            lineage.append(rebuilt.artifact_sha256)
        _require(rebuilt.to_bytes() == original_bytes, "pure current Monte Carlo reconstruction differs from frozen model bytes")
        lineages[model.artifact_sha256] = tuple(lineage)
    _require(len(configs) <= 1, "source certification cannot mix different game configurations")
    return {"games": games, "models": models_seen, "configs": configs, "context": context, "lineages": lineages}


def _project(collected, models):
    """Preserve the old audit's requested graph while validating a union once."""
    game_ids, model_ids = set(), set()
    for model in models:
        model_ids.update(collected["lineages"][model.artifact_sha256])
        for raw_batch in model.source_batches:
            game_ids.update(item["game_id"] for item in json.loads(raw_batch)["games"])
    games = {key: value for key, value in collected["games"].items() if key in game_ids}
    configs = {canonical_json(item["spec"].game_config): item["spec"].game_config for item in games.values()}
    return {"games": games, "models": {key: value for key, value in collected["models"].items() if key in model_ids},
            "configs": configs, "context": collected["context"] if games else None}


def _fresh_recheck(store, models, origins, current, collected, verification):
    """Reread all original physical inputs after semantic execution, no shortcuts."""
    actual_current = _current_identity(store, current["source_archive_sha256"])
    actual_origins = _origins(store, [item["source_archive_sha256"] for item in origins])
    _require(actual_current == current and actual_origins == origins,
             "source archive/runtime changed during semantic verification")
    rechecked = _collect(store, models, origins, actual_current, verification)
    _require([item["descriptor"] for _, item in sorted(rechecked["games"].items())]
             == [item["descriptor"] for _, item in sorted(collected["games"].items())]
             and rechecked["models"] == collected["models"] and rechecked["lineages"] == collected["lineages"],
             "original source objects changed during semantic verification")
    _require(_current_identity(store, current["source_archive_sha256"]) == current,
             "current source/runtime changed during final original-object scan")
    return rechecked


def _bindings(parent, collected, origins, current):
    _require(parent.training_games > 0 and collected["games"], "semantic certificate requires a genuinely trained nonempty parent")
    return {"parent_artifact_sha256": parent.artifact_sha256, "parent_training_games": parent.training_games,
            "parent_source_batches_sha256": content_hash(list(parent.source_batches)),
            "original_source_game_ids": sorted(collected["games"]),
            "original_sources": [collected["games"][key]["descriptor"] for key in sorted(collected["games"])],
            "reconstructed_model_artifacts": sorted(collected["models"]), "origin_archives": origins, "current_source": current,
            "game_config": json.loads(next(iter(collected["configs"]))),
            "context_id": collected["context"][0], "opponent_snapshot_ids": list(collected["context"][1])}


def _checked_write(path, payload):
    _write_immutable(path, canonical_json({"payload": payload, "sha256": content_hash(payload)}).encode("utf-8"))


def _validate_attempt(store, item, current, attempt, *, verification=None):
    _require(set(attempt) == {"game_id", "receipt_sha256", "current_trace_sha256"}
             and attempt["game_id"] == item["descriptor"]["source"]["game_id"], "invalid semantic diagnostic attempt binding")
    if verification is not None:
        _require(verification.store is store, "semantic attempt comparison belongs to a different physical store")
    identities = (_sha(attempt["receipt_sha256"]), _sha(attempt["current_trace_sha256"]),
                  _sha(item["descriptor"]["original_trace_sha256"]))
    # Every use, including a hit, rereads and hashes all three actual artifacts.
    # A successful earlier comparison cannot conceal replacement bytes.
    raw_objects = tuple(store.read_artifact(identity) for identity in identities)
    actual_hashes = tuple(hashlib.sha256(raw).hexdigest() for raw in raw_objects)
    _require(actual_hashes == identities, "semantic diagnostic artifact bytes differ from their identities")
    comparison_key = content_hash({"source_descriptor": item["descriptor"], "current_source": current,
                                   "attempt": attempt, "actual_artifact_sha256": actual_hashes})
    if verification is not None and _cache_get(verification.attempt_comparisons, comparison_key) is True:
        return
    receipt, current_trace, original_trace = (json.loads(raw) for raw in raw_objects)
    _require(receipt.get("kind") == DIAGNOSTIC_KIND and receipt.get("purpose") == "diagnostic"
             and receipt.get("source_binding_sha256") == content_hash(item["descriptor"])
             and receipt.get("current_source") == current and receipt.get("current_trace_sha256") == attempt["current_trace_sha256"]
             and receipt.get("failure") is None and receipt.get("semantic_equal") is True
             and receipt.get("current_exact_replay_verified") is True
             and receipt.get("used_for_training_or_independent_evaluation") is False,
             "semantic diagnostic receipt differs from its actual frozen source")
    _require(_loaded_json_bytes(current_trace) == _loaded_json_bytes(_current_trace_view(original_trace, current)),
             "semantic diagnostic trace differs beyond permitted Game/Race source metadata")
    if verification is not None:
        _cache_put(verification.attempt_comparisons, comparison_key, True, _ATTEMPT_COMPARISON_CACHE_LIMIT)


def _ensure_semantics(store, items, current, *, diagnostic_id=None, progress=None, verification=None):
    owned_verification = verification is None
    verification = verification or _Verification(store, (), current)
    invocation = diagnostic_id or uuid.uuid4().hex
    _require(isinstance(invocation, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", invocation),
             "diagnostic_id must be a portable identifier")
    directory = store.root / "source-semantics" / "diagnostic-invocations" / invocation
    ordered = sorted(items, key=lambda item: item["descriptor"]["source"]["game_id"])
    results, actual = [], 0
    for index, item in enumerate(ordered):
        binding = item["descriptor"]
        game_id = binding["source"]["game_id"]
        cache_key = content_hash(binding)
        attempt_json = verification.get("replay", cache_key)
        attempt = json.loads(attempt_json) if attempt_json is not None else None
        if attempt is not None:
            _validate_attempt(store, item, current, attempt, verification=verification)
            results.append(attempt)
            continue
        if progress:
            progress(f"Semantic DIAGNOSTIC {index + 1}/{len(ordered)}: original TRAIN {game_id}")
        if actual == 0:
            _checked_write(directory / "plan.json", {"kind": "current-source-semantic-diagnostic-plan-v1", "current_source": current,
                                                    "source_game_ids": [entry["descriptor"]["source"]["game_id"] for entry in ordered]})
        trace, failure = _simulate_one(item["spec"].to_dict(), binding["seeds"], item["artifacts"])
        actual += 1
        trace_sha = store.put_artifact(canonical_json(trace).encode("utf-8")) if trace is not None else None
        original_trace = json.loads(store.read_artifact(binding["original_trace_sha256"]))
        equal = (failure is None and trace is not None
                 and canonical_json(trace) == canonical_json(_current_trace_view(original_trace, current)))
        replay_verified = False
        if equal:
            try:
                reproduced = replay_game_trace(trace)
                replay_verified = canonical_json(reproduced.trace()) == canonical_json(trace)
                _require(replay_verified, "current diagnostic exact replay differs")
            except Exception as error:
                failure = {"kind": "current_replay_validation_error", "exception_type": type(error).__name__,
                           "message": str(error)}
                equal = False
        receipt = {"kind": DIAGNOSTIC_KIND, "purpose": "diagnostic", "record_level": "Game",
                   "original_source": binding,
                   "source_binding_sha256": content_hash(binding), "current_source": current,
                   "current_spec": item["spec"].to_dict(), "current_trace_sha256": trace_sha,
                   "completed": bool(trace and trace.get("completed")), "failure": failure,
                   "semantic_equal": equal, "current_exact_replay_verified": replay_verified,
                   "original_game_id_preserved": True,
                   "model_updates_performed": 0, "new_train_game_records": 0, "independent_evaluation_games": 0,
                   "used_for_training_or_independent_evaluation": False}
        receipt_sha = store.put_artifact(canonical_json(receipt).encode("utf-8"))
        # Stable content-addressed proof belongs in deterministic reports;
        # each real execution still has its own separate Game-level receipt.
        _checked_write(directory / f"attempt-{index:06d}.json", {**receipt, "receipt_artifact_sha256": receipt_sha,
                                                               "invocation_id": invocation, "attempt_index": index})
        if not equal:
            _checked_write(directory / "rejected.json", {"kind": "source-semantics-whole-parent-rejection-v1",
                "failed_game_id": game_id, "attempt_receipt_sha256": receipt_sha,
                "requested_source_game_ids": [entry["descriptor"]["source"]["game_id"] for entry in ordered],
                "actual_diagnostic_games_this_invocation": actual, "certificate_issued": False,
                "source_games_filtered_or_relabelled": 0})
            raise ExperimentError(f"current-source semantic difference; whole parent rejected; diagnostic={receipt_sha}")
        attempt = {"game_id": game_id, "receipt_sha256": receipt_sha, "current_trace_sha256": trace_sha}
        _validate_attempt(store, item, current, attempt, verification=verification)
        verification.stage("replay", cache_key, canonical_json(attempt))
        results.append(attempt)
    _require(_current_identity(store) == current, "source changed during semantic revalidation")
    if owned_verification:
        verification.commit()
    return results


@dataclass(frozen=True, slots=True)
class VerifiedSourceSemantics:
    certificate_sha256: str
    payload_json: str

    @property
    def payload(self):
        return json.loads(self.payload_json)


def create_source_semantics_certificate(store, parent_artifact_sha256, origin_source_archive_sha256s,
                                        *, diagnostic_id=None, progress=None):
    from .training_a import _archive_sources

    parent_raw = store.read_artifact(_sha(parent_artifact_sha256))
    parent = FrozenAModel.from_bytes(parent_raw)
    _require(parent.to_bytes() == parent_raw, "semantic parent artifact is not canonical")
    archive = _archive_sources(store)
    current = _current_identity(store, archive)
    origins = _origins(store, origin_source_archive_sha256s)
    verification = _Verification(store, origins, current)
    collected = _collect(store, (parent,), origins, current, verification)
    bindings = _bindings(parent, collected, origins, current)
    attempts = _ensure_semantics(store, list(collected["games"].values()), current,
                                 diagnostic_id=diagnostic_id, progress=progress, verification=verification)
    # Read all original artifacts and normalized payloads again before issue.
    rechecked = _fresh_recheck(store, (parent,), origins, current, collected, verification)
    _require(_bindings(parent, rechecked, origins, current) == bindings, "original source changed during certification")
    payload = {"schema_version": 1, "kind": CERTIFICATE_KIND, "bindings": bindings, "diagnostic_attempts": attempts,
               "parameter_reconstruction": _PARAMETERS,
               "semantic_equivalence": _EQUIVALENCE,
               "new_training_samples": 0}
    digest = store.put_artifact(canonical_json(payload).encode("utf-8"))
    _checked_write(store.root / "source-semantics" / "certificates" / f"{digest}.json", payload)
    verification.commit()
    return digest


def _certificate_inputs(store, certificate_sha256, parent_artifact_sha256=None):
    payload = json.loads(store.read_artifact(_sha(certificate_sha256)))
    _require(set(payload) == {"schema_version", "kind", "bindings", "diagnostic_attempts", "parameter_reconstruction",
                              "semantic_equivalence", "new_training_samples"}
             and type(payload["schema_version"]) is int and payload["schema_version"] == 1
             and payload["kind"] == CERTIFICATE_KIND
             and payload["parameter_reconstruction"] == _PARAMETERS and payload["semantic_equivalence"] == _EQUIVALENCE
             and type(payload["new_training_samples"]) is int and payload["new_training_samples"] == 0,
             "unsupported source-semantics certificate schema")
    binding = payload["bindings"]
    if parent_artifact_sha256 is not None:
        _require(binding["parent_artifact_sha256"] == parent_artifact_sha256, "semantic certificate binds another frozen parent")
    current = _current_identity(store, binding["current_source"]["source_archive_sha256"])
    _require(binding["current_source"] == current, "semantic certificate is not bound to current Game/Race/infrastructure source")
    origins = _origins(store, [item["source_archive_sha256"] for item in binding["origin_archives"]])
    parent_raw = store.read_artifact(_sha(binding["parent_artifact_sha256"]))
    parent = FrozenAModel.from_bytes(parent_raw)
    _require(parent.to_bytes() == parent_raw, "certified frozen parent bytes are not canonical")
    return payload, parent, origins, current


def _verify_certificate_graph(store, certificate_sha256, models=None, *, parent_artifact_sha256=None,
                              expected_payload_json=None):
    payload, parent, origins, current = _certificate_inputs(store, certificate_sha256, parent_artifact_sha256)
    payload_json = canonical_json(payload)
    if expected_payload_json is not None:
        _require(payload_json == expected_payload_json, "source-semantics verification token differs")
    binding = payload["bindings"]
    requested = (parent,) if models is None else tuple(models)
    _require(requested and all(isinstance(model, FrozenAModel) and model.config == parent.config for model in requested),
             "certified models must share the frozen parent A configuration")
    for model in requested:
        shorter = min(len(model.source_batches), len(parent.source_batches))
        _require(model.source_batches[:shorter] == parent.source_batches[:shorter],
                 "model is neither an exact certified ancestor nor a continuation of the certified parent")
    # Parent authentication and requested descendants/ancestors share one
    # physically scanned graph; an exact parent is never collected twice.
    unique = {parent.artifact_sha256: parent}
    unique.update((model.artifact_sha256, model) for model in requested)
    graph_models = tuple(unique.values())
    verification = _Verification(store, origins, current)
    all_collected = _collect(store, graph_models, origins, current, verification)
    parent_collected = _project(all_collected, (parent,))
    _require(canonical_json(_bindings(parent, parent_collected, origins, current)) == canonical_json(binding),
             "certificate differs from actual complete source objects/checkpoint reconstruction")
    expected = sorted(parent_collected["games"])
    attempts = payload["diagnostic_attempts"]
    _require(isinstance(attempts, list) and [item.get("game_id") for item in attempts] == expected,
             "certificate omits/duplicates/reorders original source Game diagnostics")
    for attempt in attempts:
        _validate_attempt(store, parent_collected["games"][attempt["game_id"]], current, attempt,
                          verification=verification)
    collected = _project(all_collected, requested)
    certified = {item["source"]["game_id"]: item for item in binding["original_sources"]}
    for game_id, item in collected["games"].items():
        original = item["descriptor"]
        if game_id in certified:
            _require(original == certified[game_id], "certified original TRAIN source objects differ")
        else:
            _require(original["original_engine_fingerprint"] == current["game_fingerprint"]
                     and original["original_infrastructure_fingerprint"] == current["infrastructure_fingerprint"],
                     "uncertified historical TRAIN source cannot extend a certified lineage")
    if collected["context"] is not None:
        _require(collected["context"] == (binding["context_id"], tuple(binding["opponent_snapshot_ids"]))
                 and list(collected["configs"]) == [canonical_json(binding["game_config"])],
                 "continuation context/configuration differs from its semantic certificate")
    # A JSON flag/receipt is not execution proof. Only this process's actual
    # seeded execution and replay under freshly checked complete inputs can
    # bypass execution. Cold processes still execute every distinct source.
    all_attempts = _ensure_semantics(store, list(all_collected["games"].values()), current,
                                    verification=verification)
    _fresh_recheck(store, graph_models, origins, current, all_collected, verification)
    verification.commit()
    attempts = [item for item in all_attempts if item["game_id"] in collected["games"]]
    return VerifiedSourceSemantics(certificate_sha256, payload_json), parent, collected, attempts


def load_source_semantics_certificate(store, certificate_sha256, *, parent_artifact_sha256=None):
    verified, _, _, _ = _verify_certificate_graph(
        store, certificate_sha256, parent_artifact_sha256=parent_artifact_sha256)
    return verified


def verify_models_source_semantics(store, models, verified_certificate):
    _require(isinstance(verified_certificate, VerifiedSourceSemantics), "a verified source-semantics certificate is required")
    verified, parent, collected, attempts = _verify_certificate_graph(
        store, verified_certificate.certificate_sha256, tuple(models),
        expected_payload_json=verified_certificate.payload_json)
    payload = verified.payload
    binding, current = payload["bindings"], payload["bindings"]["current_source"]
    historical = sum(item["descriptor"]["original_engine_fingerprint"] != current["game_fingerprint"]
                     for item in collected["games"].values())
    original_games = []
    original_runs = {}
    for game_id, item in sorted(collected["games"].items()):
        source = item["descriptor"]
        run_id = source["source"]["run_id"]
        original_games.append({"original_game_id": game_id, "run_id": run_id, "game_index": source["game_index"],
                               "focus_player_id": source["source"]["focus_player_id"],
                               "original_spec_hash": source["original_spec_sha256"],
                               "original_engine_fingerprint": source["original_engine_fingerprint"],
                               "original_trace_sha256": source["original_trace_sha256"],
                               "original_records_sha256": source["original_records_sha256"],
                               "original_summary_sha256": source["original_summary_sha256"],
                               "purpose": "train", "completed": True, "replay_verified_at_origin": True,
                               "actual_current_source_semantics_verified": True})
        original_runs[run_id] = {"run_id": run_id, "original_spec_hash": source["original_spec_sha256"],
                                 "original_engine_fingerprint": source["original_engine_fingerprint"],
                                 "original_raw_spec": source["original_spec"]}
    audit = {"source_semantics_certificate_sha256": verified.certificate_sha256,
             "model_artifacts_checked": len(collected["models"]), "complete_train_games_checked": len(collected["games"]),
             "certified_parent_artifact_sha256": parent.artifact_sha256, "certified_parent_training_games": parent.training_games,
             "current_source": current, "historical_train_source_games": historical,
             "current_engine_train_source_games": len(collected["games"]) - historical,
             "old_training_data_used_for_current_training": bool(historical),
             "compatibility_verified_historical_sources_present": bool(historical),
             "compatibility_verified_historical_sources_eligible": True,
             "original_sources": [item["descriptor"] for _, item in sorted(collected["games"].items())],
             "original_games": original_games, "original_runs": [original_runs[key] for key in sorted(original_runs)],
             "current_semantic_diagnostic_attempts": attempts,
             "source_parser_projection": "only in-memory Game/Race source stamps; original specs/GameIDs/records unchanged",
             "parameter_reconstruction": payload["parameter_reconstruction"], "semantic_equivalence": payload["semantic_equivalence"],
             "diagnostic_games_used_as_new_training_samples": 0, "evaluation_games_used_for_updates": 0,
             "historical_infrastructure_policy": "all original objects rechecked; exact current-source semantic certificate required"}
    return collected["configs"], audit


def certified_source_conditions(store, models, certificate_sha256):
    # The batch verifier authenticates and revalidates this untrusted token;
    # avoid performing the same entire-parent reconstruction twice up front.
    raw = store.read_artifact(_sha(certificate_sha256))
    verified = VerifiedSourceSemantics(certificate_sha256, canonical_json(json.loads(raw)))
    return verify_models_source_semantics(store, tuple(models), verified)
