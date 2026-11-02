"""Immutable, content-addressed contracts for isolated experiment records.

This module declares provenance and deterministic seed streams. Declaring an
A/B model artifact does not implement its policy or authorize a trainer.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from typing import Mapping

from .engine import RULE_VERSION
from .game import GameConfig, GameEngine, RacerCard
from .racer import ROLE_DATA_VERSION
from .scoring import AwardSchedule
from .track import Track

SPEC_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1
SEED_SCHEME = "sha256-domain-v1"
OBJECTIVE = "expected_four_race_total_score"
OBSERVATION_VERSION = "public-observation-v1"
A_FIXED_CONTEXT_SCHEME = "fixed-training-context-v1"
A_GUARDED_FIXED_CONTEXT_SCHEME = "fixed-training-context-guarded-v1"
MAX_SEED = (1 << 63) - 2


class Purpose(StrEnum):
    TRAIN = "train"
    EVALUATION = "evaluation"
    CROSSPLAY = "crossplay"
    DIAGNOSTIC = "diagnostic"


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError(f"{name} must be a portable identifier of 1..96 characters")


def _integer(value: int, name: str, *, maximum: int | None = None) -> None:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be a nonnegative integer" + (f" <= {maximum}" if maximum is not None else ""))


def _boolean(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")


def _sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest")


def _keys(data: Mapping, expected: set[str], name: str) -> None:
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"incompatible {name} fields")


def _jsonable(value):
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("canonical JSON object keys must be strings")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_jsonable(item) for item in value]
        if all(type(item) is int for item in items):
            return sorted(items)
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json(value) -> str:
    """Stable UTF-8 JSON representation; reject NaN, unknown types and key loss."""
    return json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def content_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_object(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("parameters_json must be a JSON object string")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON parameter key")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=unique_pairs)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("parameters_json must be a JSON object string") from exc
    if not isinstance(result, dict):
        raise ValueError("parameters_json must encode an object")
    canonical_json(result)  # Also rejects nonfinite values accepted by json.loads.
    return result


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    model_type: str
    policy_kind: str
    model_version: str
    parameters_json: str = "{}"
    frozen: bool = True
    artifact_sha256: str | None = None
    training_games: int = 0
    observation_version: str = OBSERVATION_VERSION

    def __post_init__(self):
        if not isinstance(self.model_type, str) or self.model_type not in {"baseline", "A", "B"}:
            raise ValueError("model_type must be baseline, A or B")
        if not isinstance(self.policy_kind, str) or self.policy_kind not in {"random", "first_legal", "first_legal_memory", "model_artifact"}:
            raise ValueError("unknown policy_kind")
        _nonempty(self.model_version, "model_version")
        _boolean(self.frozen, "frozen")
        _integer(self.training_games, "training_games")
        if self.observation_version != OBSERVATION_VERSION:
            raise ValueError("unsupported observation version")
        parameters = _json_object(self.parameters_json)
        object.__setattr__(self, "parameters_json", canonical_json(parameters))
        if self.model_type == "baseline":
            if self.policy_kind not in {"random", "first_legal", "first_legal_memory"}:
                raise ValueError("a baseline must identify an implemented baseline policy")
            if self.policy_kind == "first_legal_memory":
                _keys(parameters, {"scheme", "memory_capacity", "implementation_sha256"}, "public-memory baseline")
                if (self.model_version != "first-legal-public-memory-v1"
                        or parameters["scheme"] != "public-geometry-least-used-v1"
                        or type(parameters["memory_capacity"]) is not int or parameters["memory_capacity"] != 4096):
                    raise ValueError("unsupported public-memory baseline version or capacity")
                _sha256(parameters["implementation_sha256"], "public-memory baseline implementation")
                if self.artifact_sha256 is not None or self.training_games != 0 or not self.frozen:
                    raise ValueError("public-memory baselines are frozen and untrained")
            elif parameters or self.artifact_sha256 is not None or self.training_games != 0 or not self.frozen:
                raise ValueError("implemented baselines are frozen, parameter-free and untrained")
        else:
            if self.policy_kind != "model_artifact":
                raise ValueError("A/B metadata requires a real model artifact, not a renamed baseline")
            _sha256(self.artifact_sha256, "artifact_sha256")
            if self.model_type == "A" and "deployment" in parameters:
                deployment = parameters["deployment"]
                if not isinstance(deployment, Mapping):
                    raise ValueError("incompatible A deployment fields")
                guarded = deployment.get("scheme") == A_GUARDED_FIXED_CONTEXT_SCHEME
                expected_parameters = {"algorithm", "feature_version", "mode", "deployment"}
                expected_deployment = {"scheme", "lookup_context_id"}
                if guarded:
                    expected_parameters.add("repetition_guard")
                    expected_deployment.add("source_snapshot_id")
                _keys(parameters, expected_parameters, "deployed A parameters")
                _keys(deployment, expected_deployment, "A deployment")
                if deployment["scheme"] not in {A_FIXED_CONTEXT_SCHEME, A_GUARDED_FIXED_CONTEXT_SCHEME}:
                    raise ValueError("unsupported frozen A deployment scheme")
                _sha256(deployment["lookup_context_id"], "lookup_context_id")
                if guarded:
                    _sha256(deployment["source_snapshot_id"], "source_snapshot_id")
                    guard = parameters["repetition_guard"]
                    _keys(guard, {"scheme", "implementation_sha256"}, "A deployment repetition guard")
                    _nonempty(guard["scheme"], "repetition guard scheme")
                    _sha256(guard["implementation_sha256"], "repetition guard implementation")
                if not self.frozen or self.training_games < 1:
                    raise ValueError("fixed-context deployment requires a frozen trained A snapshot")

    @property
    def game_policy_id(self) -> str:
        if self.policy_kind == "random":
            return "random-legal-game-v1"
        if self.policy_kind == "first_legal":
            return "first-legal-game-v1"
        if self.policy_kind == "first_legal_memory":
            return "first-legal-public-memory-game-v1"
        if self.model_type == "A":
            parameters = json.loads(self.parameters_json)
            algorithm = parameters.get("algorithm")
            if algorithm in {"mc-dirichlet-thompson-v1", "mc-dirichlet-thompson-v2", "mc-dirichlet-thompson-v3"}:
                suffix = "-fixed-context-v1" if "deployment" in parameters else ""
                if parameters.get("deployment", {}).get("scheme") == A_GUARDED_FIXED_CONTEXT_SCHEME:
                    suffix = "-fixed-context-guarded-v1"
                return "mc-thompson-a-game-" + algorithm.rsplit("-", 1)[1] + suffix
        return f"unimplemented:{self.model_type}:game"

    @property
    def race_policy_id(self) -> str:
        if self.policy_kind == "random":
            return "random-legal-v1"
        if self.policy_kind == "first_legal":
            return "keep-or-first-v2"
        if self.policy_kind == "first_legal_memory":
            return "keep-first-public-memory-race-v1"
        if self.model_type == "A":
            parameters = json.loads(self.parameters_json)
            algorithm = parameters.get("algorithm")
            if algorithm in {"mc-dirichlet-thompson-v1", "mc-dirichlet-thompson-v2", "mc-dirichlet-thompson-v3"}:
                suffix = "-fixed-context-v1" if "deployment" in parameters else ""
                if parameters.get("deployment", {}).get("scheme") == A_GUARDED_FIXED_CONTEXT_SCHEME:
                    suffix = "-fixed-context-guarded-v1"
                return "mc-thompson-a-race-" + algorithm.rsplit("-", 1)[1] + suffix
        return f"unimplemented:{self.model_type}:race"

    @property
    def snapshot_id(self) -> str:
        return content_hash(self.to_dict())

    @property
    def content_sha256(self) -> str:
        return self.snapshot_id

    def to_dict(self) -> dict:
        return {"schema_version": SNAPSHOT_SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "PolicySnapshot":
        _keys(data, {
            "schema_version", "model_type", "policy_kind", "model_version", "parameters_json",
            "frozen", "artifact_sha256", "training_games", "observation_version",
        }, "policy snapshot")
        if type(data["schema_version"]) is not int or data["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported policy snapshot schema")
        return cls(**{key: value for key, value in data.items() if key != "schema_version"})


@dataclass(frozen=True, slots=True)
class GameSeeds:
    game_seed: int
    dice_seed: int
    game_policy_seeds: tuple[int, ...]
    race_policy_seeds: tuple[int, ...]

    def __post_init__(self):
        object.__setattr__(self, "game_policy_seeds", tuple(self.game_policy_seeds))
        object.__setattr__(self, "race_policy_seeds", tuple(self.race_policy_seeds))
        if len(self.game_policy_seeds) != 4 or len(self.race_policy_seeds) != 4:
            raise ValueError("four independent policy seeds are required for each policy layer")
        for seed in self.all_seeds:
            _integer(seed, "stream seed", maximum=MAX_SEED)
        if self.dice_seed != self.game_seed + 1:
            raise ValueError("dice_seed must match the engine's game_seed + 1 convention")
        if len(set(self.all_seeds)) != len(self.all_seeds):
            raise ValueError("game, dice and policy streams must have distinct seeds")

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return (self.game_seed, self.dice_seed, *self.game_policy_seeds, *self.race_policy_seeds)

    def to_dict(self) -> dict:
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, data: Mapping) -> "GameSeeds":
        _keys(data, {"game_seed", "dice_seed", "game_policy_seeds", "race_policy_seeds"}, "game seeds")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class SeedPlan:
    master_seed: int
    purpose: Purpose
    namespace: str
    scheme: str = SEED_SCHEME

    def __post_init__(self):
        _integer(self.master_seed, "master_seed", maximum=MAX_SEED)
        object.__setattr__(self, "purpose", Purpose(self.purpose))
        _identifier(self.namespace, "seed namespace")
        if self.scheme != SEED_SCHEME:
            raise ValueError("unsupported seed derivation scheme")

    def derive(self, game_index: int) -> GameSeeds:
        _integer(game_index, "game_index", maximum=MAX_SEED)
        # Disjoint numeric domains make purpose isolation exact, rather than
        # relying only on the improbability of a hash collision. A 60-bit local
        # domain leaves headroom for the engine's dice_seed = game_seed + 1.
        prefix = tuple(Purpose).index(self.purpose) << 60
        mask = (1 << 60) - 1
        used = set()

        def derive_stream(label: str, *, paired=False) -> int:
            attempt = 0
            while True:
                raw = {
                    **self.to_dict(), "game_index": game_index,
                    "stream": label, "collision_attempt": attempt,
                }
                local = int(content_hash(raw), 16) & mask
                if paired:
                    local &= ~1
                seed = prefix | local
                claimed = {seed, seed + 1} if paired else {seed}
                if not used.intersection(claimed):
                    used.update(claimed)
                    return seed
                attempt += 1

        game_seed = derive_stream("game-shuffle", paired=True)
        return GameSeeds(
            game_seed, game_seed + 1,
            tuple(derive_stream(f"game-policy-seat-{seat}") for seat in range(4)),
            tuple(derive_stream(f"race-policy-seat-{seat}") for seat in range(4)),
        )

    def to_dict(self) -> dict:
        return {"master_seed": self.master_seed, "purpose": self.purpose.value,
                "namespace": self.namespace, "scheme": self.scheme}

    @classmethod
    def from_dict(cls, data: Mapping) -> "SeedPlan":
        _keys(data, {"master_seed", "purpose", "namespace", "scheme"}, "seed plan")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RunSpec:
    run_id: str
    study_id: str
    purpose: Purpose
    policies: tuple[PolicySnapshot, ...]
    seed_plan: SeedPlan
    game_config: GameConfig
    learning: bool = False
    objective: str = OBJECTIVE
    observation_version: str = OBSERVATION_VERSION
    rule_version: str = RULE_VERSION
    engine_fingerprint: str = field(default_factory=GameEngine._code_fingerprint)

    def __post_init__(self):
        _identifier(self.run_id, "run_id")
        _identifier(self.study_id, "study_id")
        object.__setattr__(self, "purpose", Purpose(self.purpose))
        object.__setattr__(self, "policies", tuple(self.policies))
        if len(self.policies) != 4 or not all(isinstance(policy, PolicySnapshot) for policy in self.policies):
            raise ValueError("exactly four explicit policy snapshots are required")
        if not isinstance(self.seed_plan, SeedPlan):
            raise ValueError("seed_plan must be a SeedPlan")
        if self.seed_plan.purpose != self.purpose or self.seed_plan.namespace != self.study_id:
            raise ValueError("seed plan purpose/namespace must match the run purpose/study_id")
        if not isinstance(self.game_config, GameConfig):
            raise ValueError("game_config must be a GameConfig")
        _boolean(self.game_config.official, "game_config.official")
        _boolean(self.learning, "learning")
        if self.objective != OBJECTIVE:
            raise ValueError("the objective must remain expected four-race total score")
        if self.observation_version != OBSERVATION_VERSION or any(
            policy.observation_version != self.observation_version for policy in self.policies
        ):
            raise ValueError("incompatible observation version")
        _sha256(self.engine_fingerprint, "engine_fingerprint")
        if self.rule_version != RULE_VERSION or self.engine_fingerprint != GameEngine._code_fingerprint():
            raise ValueError("new runs require the current rule and engine versions")
        if self.purpose != Purpose.DIAGNOSTIC and not self.game_config.official:
            raise ValueError("non-diagnostic runs require verified official game data")
        if self.learning and self.purpose != Purpose.TRAIN:
            raise ValueError("only train runs may update a learner")
        models = {policy.model_type for policy in self.policies}
        if self.purpose == Purpose.TRAIN and models not in ({"A"}, {"B"}):
            raise ValueError("initial training must be isolated 4A or 4B")
        if self.purpose == Purpose.CROSSPLAY and models != {"A", "B"}:
            raise ValueError("crossplay requires an explicit mixed A/B lineup")
        if self.purpose == Purpose.EVALUATION and {"A", "B"}.issubset(models):
            raise ValueError("mixed A/B comparison must use the separate crossplay purpose")
        if self.purpose in {Purpose.EVALUATION, Purpose.CROSSPLAY} and any(not policy.frozen for policy in self.policies):
            raise ValueError("evaluation and crossplay require frozen policy snapshots")

    @property
    def seat_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(policy.snapshot_id for policy in self.policies)

    @property
    def matchup(self) -> str:
        return "+".join(f"{sum(policy.model_type == model for policy in self.policies)}{model}"
                        for model in ("A", "B", "baseline")
                        if any(policy.model_type == model for policy in self.policies))

    @property
    def data_versions(self) -> dict:
        return {
            "role": ROLE_DATA_VERSION,
            "mild": self.game_config.mild_track.data_version,
            "wild": self.game_config.wild_track.data_version,
            "awards": self.game_config.award_schedule.data_version,
        }

    @property
    def spec_id(self) -> str:
        return content_hash(self.to_dict())

    @property
    def content_sha256(self) -> str:
        return self.spec_id

    def to_dict(self) -> dict:
        return {
            "schema_version": SPEC_SCHEMA_VERSION,
            "run_id": self.run_id, "study_id": self.study_id, "purpose": self.purpose.value,
            "policies": [policy.to_dict() for policy in self.policies],
            "seed_plan": self.seed_plan.to_dict(),
            "game_config": json.loads(canonical_json(self.game_config)),
            "learning": self.learning, "objective": self.objective,
            "observation_version": self.observation_version,
            "data_versions": self.data_versions,
            "rule_version": self.rule_version, "engine_fingerprint": self.engine_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "RunSpec":
        _keys(data, {
            "schema_version", "run_id", "study_id", "purpose", "policies", "seed_plan", "game_config",
            "learning", "objective", "observation_version", "data_versions", "rule_version", "engine_fingerprint",
        }, "run specification")
        if type(data["schema_version"]) is not int or data["schema_version"] != SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported run specification schema")
        config = data["game_config"]
        _keys(config, {"mild_track", "wild_track", "award_schedule", "deck", "official", "source",
                       "max_turns_per_race", "max_rolloff_rounds", "max_events_per_race"}, "game configuration")
        for track in (config["mild_track"], config["wild_track"]):
            _keys(track, {item.name for item in fields(Track)}, "track data")
        _keys(config["award_schedule"], {item.name for item in fields(AwardSchedule)}, "award data")
        for card in config["deck"]:
            _keys(card, {item.name for item in fields(RacerCard)}, "racer card")
        rebuilt = GameConfig(
            Track(**config["mild_track"]), Track(**config["wild_track"]), AwardSchedule(**config["award_schedule"]),
            deck=tuple(RacerCard(**card) for card in config["deck"]),
            official=config["official"], source=config["source"],
            max_turns_per_race=config["max_turns_per_race"], max_rolloff_rounds=config["max_rolloff_rounds"],
            max_events_per_race=config["max_events_per_race"],
        )
        result = cls(
            **{key: value for key, value in data.items()
               if key not in {"schema_version", "policies", "seed_plan", "game_config", "data_versions"}},
            policies=tuple(PolicySnapshot.from_dict(policy) for policy in data["policies"]),
            seed_plan=SeedPlan.from_dict(data["seed_plan"]), game_config=rebuilt,
        )
        if data["data_versions"] != result.data_versions:
            raise ValueError("recorded data versions do not match the role/track/award data")
        return result
