"""Score-valued first-visit Monte Carlo control and Thompson policy snapshots.

Each state/action uses empirical Dirichlet/Bayesian-bootstrap score weights
with explicit prior pseudo-mass. New observed returns add empirical atoms;
this is not an exact fixed-support categorical posterior. There is no fixed
point supply or Bernoulli win/loss reward. Contexts remain disjoint.
"""

import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import dataclass, field
from random import Random
from types import MappingProxyType

from .decisions import Action, ActionKind, RandomPolicy
from .experiment_spec import OBJECTIVE, canonical_json, content_hash
from .game import RandomGamePolicy
from .model_a_features import FEATURE_VERSION, action_key, encode_request, normalize_request

ALGORITHM = "mc-dirichlet-thompson-v3"
MODEL_SCHEMA_VERSION = 2
GAME_POLICY_ID = "mc-thompson-a-game-v3"
RACE_POLICY_ID = "mc-thompson-a-race-v3"
RETURN_DEFINITION = "final_total_minus_current_total"
VISIT_RULE = "first_visit_per_game_focus_player_state_action"


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _integer(value, name, *, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))


def _keys(data, expected, name):
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"incompatible {name} fields")


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be lowercase SHA256")


def _decode(raw):
    def unique(pairs):
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError("duplicate model JSON key")
            data[key] = value
        return data

    try:
        data = json.loads(raw, object_pairs_hook=unique)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError("invalid model JSON") from exc
    canonical_json(data)
    return data


@dataclass(frozen=True, slots=True)
class AConfig:
    # A modest, explicit prior over remaining score, not a score cap. Negative
    # outcomes and arbitrary new integer values are added without clipping.
    prior_atoms: tuple[tuple[int, float], ...] = ((-4, 0.25), (0, 0.5), (4, 0.5), (8, 0.25), (12, 0.25))
    min_samples: int = 3
    exploration_floor: float = 0.05
    repetition_limit: int = 3
    history_capacity: int = 256

    def __post_init__(self):
        atoms = tuple(tuple(atom) for atom in self.prior_atoms)
        if not atoms or any(len(atom) != 2 for atom in atoms):
            raise ValueError("a nonempty score/alpha prior is required")
        for score, alpha in atoms:
            _integer(score, "prior score")
            if type(alpha) not in {int, float} or not math.isfinite(alpha) or alpha <= 0:
                raise ValueError("Dirichlet prior alpha must be finite and positive")
        if len({score for score, _ in atoms}) != len(atoms):
            raise ValueError("prior score atoms must be unique")
        try:
            concentration = math.fsum(alpha for _, alpha in atoms)
        except OverflowError as exc:
            raise ValueError("total prior pseudo-mass must be finite") from exc
        if not math.isfinite(concentration) or concentration <= 0:
            raise ValueError("total prior pseudo-mass must be finite and positive")
        _integer(self.min_samples, "min_samples", minimum=1)
        if (type(self.exploration_floor) not in {int, float} or not math.isfinite(self.exploration_floor)
                or not 0 <= self.exploration_floor < 1):
            raise ValueError("exploration_floor must be finite and in [0,1)")
        _integer(self.repetition_limit, "repetition_limit", minimum=1)
        _integer(self.history_capacity, "history_capacity", minimum=self.repetition_limit)
        object.__setattr__(self, "prior_atoms", tuple(sorted((score, float(alpha)) for score, alpha in atoms)))
        object.__setattr__(self, "exploration_floor", float(self.exploration_floor))

    def to_dict(self):
        return {"prior_atoms": [list(atom) for atom in self.prior_atoms], "min_samples": self.min_samples,
                "exploration_floor": self.exploration_floor, "repetition_limit": self.repetition_limit,
                "history_capacity": self.history_capacity}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"prior_atoms", "min_samples", "exploration_floor", "repetition_limit", "history_capacity"}, "A configuration")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PosteriorRow:
    context_id: str
    state_key: str
    action_key: str
    counts: tuple[tuple[int, int], ...]

    def __post_init__(self):
        for name in ("context_id", "state_key", "action_key"):
            _label(getattr(self, name), name)
        counts = tuple(tuple(item) for item in self.counts)
        if not counts or any(len(item) != 2 for item in counts):
            raise ValueError("observed posterior counts must be nonempty score/count pairs")
        for score, count in counts:
            _integer(score, "return score")
            _integer(count, "observation count", minimum=1)
        if len({score for score, _ in counts}) != len(counts):
            raise ValueError("posterior return atoms must be unique")
        object.__setattr__(self, "counts", tuple(sorted(counts)))

    @property
    def observations(self):
        return sum(count for _, count in self.counts)

    def to_dict(self):
        return {"context_id": self.context_id, "state_key": self.state_key,
                "action_key": self.action_key, "counts": [list(item) for item in self.counts]}

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"context_id", "state_key", "action_key", "counts"}, "posterior row")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    value: float
    observations: int
    level: int | None
    state_key: str | None
    source: str

    def to_dict(self):
        return {"mean": self.mean, "value": self.value, "observations": self.observations,
                "level": self.level, "state_key": self.state_key, "source": self.source}


@dataclass(frozen=True, slots=True)
class Selection:
    index: int
    choice: object
    audit_json: str
    behavior_probability: float | None
    probability_source: str

    @property
    def audit(self):
        # A detached JSON value cannot mutate the frozen policy/model.
        return json.loads(self.audit_json)


class PublicDecisionMemory:
    """Bounded public action history; never a rule-engine cycle detector.

    Opponents scoring and the focal player losing points do not erase a stuck
    action pattern. Only a new race/context or a genuine focal score increase
    resets this memory. Geometry/legality keys are supplied by the observable
    encoder, separately from the score-aware value features.
    """

    def __init__(self, capacity=256):
        _integer(capacity, "history capacity", minimum=1)
        self.capacity = capacity
        self._records = deque()
        self._counts = Counter()
        self._domain = None
        self._own_score = None

    def reset(self):
        self._records.clear()
        self._counts.clear()
        self._domain = None
        self._own_score = None

    def prepare(self, context_id, encoded):
        domain = (context_id, encoded.race_index)
        own_score = encoded.public_scores[0]
        if self._domain != domain or (self._own_score is not None and own_score > self._own_score):
            self._records.clear()
            self._counts.clear()
        self._domain, self._own_score = domain, own_score
        key = encoded.repetition_key
        return key, tuple(self._counts[(key, action)] for action in encoded.action_keys)

    def remember(self, key, action):
        self._records.append((key, action))
        self._counts[(key, action)] += 1
        if len(self._records) > self.capacity:
            previous = self._records.popleft()
            self._counts[previous] -= 1
            if not self._counts[previous]:
                del self._counts[previous]

    @property
    def size(self):
        return len(self._records)


@dataclass(frozen=True, slots=True)
class FrozenAModel:
    config: AConfig = field(default_factory=AConfig)
    rows: tuple[PosteriorRow, ...] = ()
    training_game_ids: tuple[str, ...] = ()
    source_batches: tuple[str, ...] = ()
    _index: object = field(init=False, repr=False, compare=False)
    _canonical_bytes: bytes = field(init=False, repr=False, compare=False)
    _artifact_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.config, AConfig):
            raise ValueError("model requires AConfig")
        rows = tuple(self.rows)
        if any(not isinstance(row, PosteriorRow) for row in rows):
            raise ValueError("model rows require immutable PosteriorRow records")
        index = {(row.context_id, row.state_key, row.action_key): row for row in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate context/state/action posterior row")
        games = tuple(self.training_game_ids)
        for game_id in games:
            _label(game_id, "training game ID")
        if len(set(games)) != len(games):
            raise ValueError("duplicate training game ID")
        batches, sourced, batch_ids, context_counts, context_opponents = [], [], set(), Counter(), {}
        for raw in self.source_batches:
            batch = _decode(raw)
            _keys(batch, {"batch_id", "parent_artifact_sha256", "games"}, "source batch")
            _label(batch["batch_id"], "batch_id")
            _digest(batch["parent_artifact_sha256"], "parent artifact")
            if batch["batch_id"] in batch_ids or not isinstance(batch["games"], list) or not batch["games"]:
                raise ValueError("duplicate/empty source batch")
            batch_ids.add(batch["batch_id"])
            for game in batch["games"]:
                required = {"game_id", "context_id", "focus_player_id", "purpose", "run_id", "behavior_snapshot_id", "opponent_snapshot_ids"}
                _keys(game, required, "source game")
                _label(game["game_id"], "source game ID")
                _label(game["context_id"], "context_id")
                _integer(game["focus_player_id"], "focus_player_id", minimum=0)
                if game["focus_player_id"] > 3 or game["purpose"] != "train":
                    raise ValueError("source games must be train games with one standard focus player")
                for name in ("run_id", "behavior_snapshot_id"):
                    if game[name] is not None:
                        _label(game[name], name)
                opponents = game["opponent_snapshot_ids"]
                if opponents is not None and (not isinstance(opponents, list) or len(opponents) != 3 or any(not isinstance(item, str) or not item for item in opponents)):
                    raise ValueError("opponent lineage must contain three snapshot IDs")
                if opponents is not None:
                    previous = context_opponents.setdefault(game["context_id"], opponents)
                    if previous != opponents:
                        raise ValueError("one context cannot silently pool different opponent snapshot lineups")
                sourced.append(game["game_id"])
                context_counts[game["context_id"]] += 1
            batches.append(canonical_json(batch))
        if sorted(sourced) != sorted(games):
            raise ValueError("training games and explicit source lineage must agree exactly")
        for row in rows:
            if row.observations > context_counts[row.context_id]:
                raise ValueError("first-visit observations exceed independent source games in this context")
        object.__setattr__(self, "rows", tuple(sorted(rows, key=lambda row: (row.context_id, row.state_key, row.action_key))))
        object.__setattr__(self, "training_game_ids", tuple(sorted(games)))
        object.__setattr__(self, "source_batches", tuple(batches))
        object.__setattr__(self, "_index", MappingProxyType(index))
        # All stored inputs are now detached immutable tuples/records/strings.
        # Keep one canonical byte buffer, never a caller-mutable JSON tree;
        # each update constructs a new model and therefore a separate cache.
        raw = canonical_json(self.to_dict()).encode("utf-8")
        object.__setattr__(self, "_canonical_bytes", raw)
        object.__setattr__(self, "_artifact_sha256", hashlib.sha256(raw).hexdigest())

    @classmethod
    def empty(cls, config=None):
        return cls(config=config if config is not None else AConfig())

    @property
    def training_games(self):
        return len(self.training_game_ids)

    @property
    def artifact_sha256(self):
        return self._artifact_sha256

    def to_dict(self):
        return {
            "schema_version": MODEL_SCHEMA_VERSION, "algorithm": ALGORITHM,
            "feature_version": FEATURE_VERSION, "objective": OBJECTIVE,
            "return_definition": RETURN_DEFINITION, "visit_rule": VISIT_RULE,
            "config": self.config.to_dict(), "rows": [row.to_dict() for row in self.rows],
            "training_games": self.training_games, "training_game_ids": list(self.training_game_ids),
            "source_batches": [json.loads(raw) for raw in self.source_batches],
        }

    def to_bytes(self):
        return self._canonical_bytes

    @classmethod
    def from_dict(cls, data):
        _keys(data, {"schema_version", "algorithm", "feature_version", "objective", "return_definition", "visit_rule",
                     "config", "rows", "training_games", "training_game_ids", "source_batches"}, "model A artifact")
        if type(data["schema_version"]) is not int or data["schema_version"] != MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported model A schema")
        if (data["algorithm"], data["feature_version"], data["objective"], data["return_definition"], data["visit_rule"]) != (
            ALGORITHM, FEATURE_VERSION, OBJECTIVE, RETURN_DEFINITION, VISIT_RULE,
        ):
            raise ValueError("incompatible model algorithm/features/objective/return convention")
        _integer(data["training_games"], "training_games", minimum=0)
        if any(not isinstance(data[name], list) for name in ("rows", "training_game_ids", "source_batches")):
            raise ValueError("model collections must be JSON arrays")
        model = cls(AConfig.from_dict(data["config"]), tuple(PosteriorRow.from_dict(row) for row in data["rows"]),
                    tuple(data["training_game_ids"]), tuple(canonical_json(batch) for batch in data["source_batches"]))
        if model.training_games != data["training_games"]:
            raise ValueError("training game count differs from source lineage")
        return model

    @classmethod
    def from_bytes(cls, raw):
        return cls.from_dict(_decode(raw))

    def estimate(self, context_id, state_keys, action, *, mode="mean", rng=None, forced_level=None):
        _label(context_id, "context_id")
        _label(action, "action key")
        keys = tuple(state_keys)
        if not keys or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("distinct ordered fine-to-coarse state keys are required")
        if mode not in {"thompson", "mean"}:
            raise ValueError("model A mode must be thompson or mean")
        row, level = None, None
        if forced_level is not None:
            _integer(forced_level, "forced level", minimum=0)
            if forced_level >= len(keys):
                raise ValueError("forced level is outside the observable hierarchy")
            level = forced_level
            row = self._index.get((context_id, keys[level], action))
            source = "common_sample_threshold" if row is not None and row.observations >= self.config.min_samples else "common_coarse_sparse"
            if row is None:
                source = "prior_only"
        else:
            for current_level, key in enumerate(keys):
                candidate = self._index.get((context_id, keys[current_level], action))
                if candidate is not None and candidate.observations >= self.config.min_samples:
                    row, level = candidate, current_level
                    break
            source = "sample_threshold"
            if row is None:
                # Direct inspection can request one action; policy selection
                # below always compares every legal action at a shared level.
                for current_level in reversed(range(len(keys))):
                    candidate = self._index.get((context_id, keys[current_level], action))
                    if candidate is not None:
                        row, level = candidate, current_level
                        break
                source = "coarse_sparse" if row is not None else "prior_only"
        alphas = dict(self.config.prior_atoms)
        if row is not None:
            for score, count in row.counts:
                alphas[score] = alphas.get(score, 0.0) + count
        atoms = tuple(sorted(alphas.items()))
        total = math.fsum(alpha for _, alpha in atoms)
        mean = math.fsum(score * (alpha / total) for score, alpha in atoms)
        value = mean
        if mode == "thompson":
            if not isinstance(rng, Random):
                raise ValueError("Thompson sampling requires an owned Random instance")
            weights = [(score, rng.gammavariate(alpha, 1.0)) for score, alpha in atoms]
            mass = math.fsum(weight for _, weight in weights)
            if not mass > 0 or not math.isfinite(mass):
                raise ArithmeticError("Dirichlet sample underflow/overflow; no valid posterior draw")
            value = math.fsum(score * (weight / mass) for score, weight in weights)
        if not math.isfinite(mean) or not math.isfinite(value):
            raise ArithmeticError("nonfinite posterior estimate")
        return Estimate(mean, value, row.observations if row else 0, level, keys[level] if level is not None else None, source)

    def select(self, context_id, encoded, *, mode="thompson", rng, memory=None):
        if not isinstance(rng, Random):
            raise ValueError("selection requires a controlled Random instance")
        if not encoded.actions or len(encoded.actions) != len(encoded.action_keys):
            raise ValueError("encoded legal actions and action keys must agree")
        if len(set(encoded.action_keys)) != len(encoded.action_keys):
            raise ValueError("legal actions must have distinct semantic identities")
        if memory is not None and (not isinstance(memory, PublicDecisionMemory) or memory.capacity != self.config.history_capacity):
            raise ValueError("public memory must match the frozen history capacity")
        common_level = len(encoded.state_keys) - 1
        for level, state in enumerate(encoded.state_keys):
            if all((row := self._index.get((context_id, state, action))) is not None
                   and row.observations >= self.config.min_samples for action in encoded.action_keys):
                common_level = level
                break
        estimates = [self.estimate(context_id, encoded.state_keys, key, mode=mode, rng=rng, forced_level=common_level)
                     for key in encoded.action_keys]
        high = max(estimate.value for estimate in estimates)
        ties = [index for index, estimate in enumerate(estimates) if estimate.value == high]
        base_selected = ties[0] if len(ties) == 1 else rng.choice(ties)
        count = len(encoded.actions)
        history_key, history_counts = (None, (0,) * count) if memory is None else memory.prepare(context_id, encoded)
        keep = [index for index, action in enumerate(encoded.actions)
                if isinstance(action, Action) and action.kind == ActionKind.KEEP]
        # Declining an optional effect must not itself force that effect next
        # time. Uniform exploration still gives every legal action positive
        # probability; the history guard only redirects repeated non-KEEPs.
        overused = [count > 1 and index not in keep and value >= self.config.repetition_limit
                    for index, value in enumerate(history_counts)]
        fallback_by_proposed = []
        fallback_kinds = []
        for proposed in range(count):
            alternatives = [index for index in range(count) if index != proposed]
            available_keep = [index for index in keep if index != proposed]
            if proposed in keep:
                choices, kind = [proposed], "keep"
            elif available_keep:
                choices, kind = available_keep, "keep"
            elif alternatives:
                least = min(history_counts[index] for index in alternatives)
                choices, kind = [index for index in alternatives if history_counts[index] == least], "least_used"
            else:
                choices, kind = [proposed], "only_legal_action"
            fallback_by_proposed.append(choices)
            fallback_kinds.append(kind)
        repeated = overused[base_selected]
        fallback = fallback_by_proposed[base_selected]
        selected = rng.choice(fallback) if repeated and len(fallback) > 1 else fallback[0] if repeated else base_selected
        fallback_selected = selected if repeated else None
        component = "repetition" if repeated else "model"
        epsilon = self.config.exploration_floor
        exploration_taken = count > 1 and epsilon > 0 and rng.random() < epsilon
        if exploration_taken:
            selected = rng.randrange(count)
            component = "uniform"
        probabilities = None
        if count == 1:
            probability, probability_source = 1.0, "only_legal_action"
            probabilities = [1.0]
        elif mode == "mean":
            # The guard depends on the proposed base action, so marginalize all
            # greedy ties, including ties that would trigger a different branch.
            effective = [0.0] * count
            for proposed in ties:
                choices = fallback_by_proposed[proposed] if overused[proposed] else [proposed]
                for index in choices:
                    effective[index] += 1 / len(ties) / len(choices)
            probabilities = [(1 - epsilon) * value + epsilon / count for value in effective]
            probability, probability_source = probabilities[selected], "mean_public_history_uniform_mixture"
        else:
            probability, probability_source = None, "thompson_history_mixture_integral_not_computed"
        if memory is not None:
            memory.remember(history_key, encoded.action_keys[selected])
        audit = {"algorithm": ALGORITHM, "mode": mode, "context_id": context_id,
                 "feature_version": FEATURE_VERSION, "scope": encoded.scope, "race_index": encoded.race_index,
                 "action_keys": list(encoded.action_keys), "selected_index": selected,
                 "base_selected_index": base_selected, "common_level": common_level,
                 "estimates": [estimate.to_dict() for estimate in estimates],
                 "tied_best_indices": ties, "behavior_probability": probability,
                 "behavior_probability_source": probability_source,
                 "action_probabilities": probabilities, "behavior_probability_lower_bound": epsilon / count,
                 "exploration_floor": epsilon, "repetition_limit": self.config.repetition_limit,
                 "repetition_triggered": repeated, "repetition_count": history_counts[base_selected],
                 "history_key": history_key, "history_action_counts_before": list(history_counts),
                 "fallback_kind": fallback_kinds[base_selected] if repeated else None,
                 "fallback_action_indices": fallback if repeated else [],
                 "fallback_candidates_by_proposed": fallback_by_proposed,
                 "fallback_selected_index": fallback_selected,
                 "mixture_component": component, "uniform_exploration_taken": exploration_taken,
                 "history_size_after": memory.size if memory is not None else 0}
        return Selection(selected, encoded.actions[selected], canonical_json(audit), probability, probability_source)

    def update_games(self, games, *, batch_id=None):
        games = list(games)
        if not games:
            raise ValueError("a nonempty completed training batch is required")
        counts = {key: Counter(dict(row.counts)) for key, row in self._index.items()}
        game_ids = set(self.training_game_ids)
        sources = []
        for game in games:
            required = {"game_id", "context_id", "focus_player_id", "purpose", "transitions"}
            optional = {"run_id", "behavior_snapshot_id", "opponent_snapshot_ids"}
            if not isinstance(game, dict) or not required <= set(game) or set(game) - required - optional:
                raise ValueError("incompatible training game fields")
            game_id, context_id, focus = game["game_id"], game["context_id"], game["focus_player_id"]
            _label(game_id, "game_id")
            _label(context_id, "context_id")
            _integer(focus, "focus_player_id", minimum=0)
            if focus > 3 or game["purpose"] != "train":
                raise ValueError("only train games for one declared focus player may update model A")
            if game_id in game_ids:
                raise ValueError("game already contributes to this model; duplicate update refused")
            transitions = list(game["transitions"])
            relevant = [row for row in transitions if row["player_id"] == focus]
            if any("model_a_deployment" in row for row in relevant):
                raise ValueError("fixed-context deployment trajectories cannot update model A")
            if not relevant or sum(row["done"] is True for row in relevant) != 1 or relevant[-1]["done"] is not True:
                raise ValueError("focus trajectory must be complete with one final terminal decision")
            if any(type(row["done"]) is not bool for row in relevant):
                raise ValueError("transition terminal flags must be booleans")
            final_values = {row["final_total"] for row in relevant}
            if len(final_values) != 1:
                raise ValueError("focus trajectory must agree on final total")
            final_total = relevant[-1]["final_total"]
            _integer(final_total, "final total", minimum=0)
            first_visits = set()
            previous_id = 0
            for transition in relevant:
                _integer(transition["player_id"], "transition player ID", minimum=0)
                local_id = transition["decision_seq"]
                _integer(local_id, "decision sequence", minimum=1)
                if local_id <= previous_id:
                    raise ValueError("focus decisions must remain in chronological order")
                previous_id = local_id
                _integer(transition["score_before"], "score before decision", minimum=0)
                value = transition["return_to_go"]
                _integer(value, "remaining score return")
                if value != final_total - transition["score_before"]:
                    raise ValueError("remaining return repeats or omits already earned points")
                request, race_index = transition["request"], transition["race_index"]
                actual_race = race_index if transition["scope"] == "race" else None
                raw_request = normalize_request(request, race_index=actual_race)
                public = raw_request["observation"]
                observed_score = dict(public["score_totals"])[focus] if transition["scope"] == "race" else public["scores"][focus]
                if observed_score != transition["score_before"] or type(transition["final_total"]) is not int:
                    raise ValueError("return accounting differs from the actual observed score")
                encoded = encode_request(raw_request, race_index=actual_race)
                chosen_key = action_key(raw_request, transition["action"], race_index=actual_race)
                if chosen_key not in encoded.action_keys:
                    raise ValueError("training action must be legal in its actual request")
                request_player = request["player_id"] if isinstance(request, dict) else request.player_id
                if request_player != focus or encoded.scope != transition["scope"]:
                    raise ValueError("training request belongs to another player or phase")
                for state in encoded.state_keys:
                    key = (context_id, state, chosen_key)
                    if key not in first_visits:
                        counts.setdefault(key, Counter())[value] += 1
                        first_visits.add(key)
            game_ids.add(game_id)
            sources.append({"game_id": game_id, "context_id": context_id, "focus_player_id": focus, "purpose": "train",
                            "run_id": game.get("run_id"), "behavior_snapshot_id": game.get("behavior_snapshot_id"),
                            "opponent_snapshot_ids": game.get("opponent_snapshot_ids")})
        parent = self.artifact_sha256
        batch_id = batch_id if batch_id is not None else content_hash(sources)
        _label(batch_id, "batch_id")
        batch = canonical_json({"batch_id": batch_id, "parent_artifact_sha256": parent, "games": sources})
        rows = tuple(PosteriorRow(*key, tuple(counter.items())) for key, counter in counts.items())
        return FrozenAModel(self.config, rows, tuple(game_ids), (*self.source_batches, batch))

    def summary(self):
        return {"algorithm": ALGORITHM, "feature_version": FEATURE_VERSION,
                "training_games": self.training_games, "source_batches": len(self.source_batches),
                "contexts": len({row.context_id for row in self.rows}), "state_action_rows": len(self.rows),
                "hierarchical_first_visit_observations": sum(row.observations for row in self.rows),
                "observed_return_values": sorted({score for row in self.rows for score, _ in row.counts}),
                "decision_config": {"exploration_floor": self.config.exploration_floor,
                                    "repetition_limit": self.config.repetition_limit,
                                    "history_capacity": self.config.history_capacity},
                "artifact_sha256": self.artifact_sha256}


@dataclass(slots=True)
class SharedPublicContext:
    race_index: int | None = None


class AGamePolicy(RandomGamePolicy):
    policy_id = GAME_POLICY_ID

    def __init__(self, model, context_id, seed, mode="thompson", shared_context=None):
        super().__init__(seed)
        self.model, self.context_id, self.mode = model, context_id, mode
        self.shared_context = shared_context if shared_context is not None else SharedPublicContext()
        self.record_audits = []
        self.memory = PublicDecisionMemory(model.config.history_capacity)

    def choose(self, request):
        encoded = encode_request(request)
        if encoded.scope == "entry":
            self.shared_context.race_index = encoded.race_index
        selection = self.model.select(self.context_id, encoded, mode=self.mode, rng=self._random, memory=self.memory)
        self.record_audits.append(selection.audit)
        return selection.choice


class ARacePolicy(RandomPolicy):
    policy_id = RACE_POLICY_ID

    def __init__(self, model, context_id, seed, mode="thompson", shared_context=None, race_index=None):
        super().__init__(seed)
        self.model, self.context_id, self.mode = model, context_id, mode
        self.shared_context = shared_context if shared_context is not None else SharedPublicContext(race_index)
        self.record_audits = []
        self.memory = PublicDecisionMemory(model.config.history_capacity)

    def set_race_index(self, race_index):
        _integer(race_index, "race index", minimum=1)
        if race_index > 4:
            raise ValueError("race index must be 1..4")
        self.shared_context.race_index = race_index

    def choose(self, request):
        if self.shared_context.race_index is None:
            raise ValueError("model A race policy requires the actual public race index")
        encoded = encode_request(request, race_index=self.shared_context.race_index)
        selection = self.model.select(self.context_id, encoded, mode=self.mode, rng=self._random, memory=self.memory)
        self.record_audits.append(selection.audit)
        return selection.choice
