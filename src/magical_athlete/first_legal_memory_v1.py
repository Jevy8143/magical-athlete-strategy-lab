"""Versioned deterministic baseline using only bounded public Race memory.

The original first-legal baseline remains unchanged. This independent opponent
cycles least-used legal actions when the same public geometry/menu recurs.
"""

from collections import OrderedDict
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path

from .decisions import Action, ActionKind, DecisionRequest, Observation, PublicRacer
from .game import DraftRequest, EntryRequest, PublicGameObservation


SCHEME = "public-geometry-least-used-v1"
POLICY_KIND = "first_legal_memory"
MODEL_VERSION = "first-legal-public-memory-v1"
GAME_POLICY_ID = "first-legal-public-memory-game-v1"
RACE_POLICY_ID = "keep-first-public-memory-race-v1"
MEMORY_CAPACITY = 4096
IMPORTED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def baseline_parameters():
    return {"scheme": SCHEME, "memory_capacity": MEMORY_CAPACITY,
            "implementation_sha256": IMPORTED_IMPLEMENTATION_SHA256}


def require_snapshot(snapshot):
    actual = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (snapshot.model_type != "baseline" or snapshot.policy_kind != POLICY_KIND
            or snapshot.model_version != MODEL_VERSION or snapshot.artifact_sha256 is not None
            or snapshot.training_games != 0 or snapshot.frozen is not True
            or json.loads(snapshot.parameters_json) != baseline_parameters()
            or actual != IMPORTED_IMPLEMENTATION_SHA256):
        raise ValueError("public-memory baseline differs from its exact frozen implementation")


def _race_request(request):
    # Normalized persisted records contain detached dictionaries. Rebuilding
    # the same immutable public contract makes offline auditing sequential and
    # identical to the live policy, without model feature/state projections.
    if isinstance(request, dict):
        data = dict(request)
        observation = dict(data.pop("observation"))
        observation["racers"] = tuple(PublicRacer(**r) for r in observation["racers"])
        data["legal_actions"] = tuple(Action(**a) for a in data["legal_actions"])
        request = DecisionRequest(observation=Observation(**observation), **data)
    if not isinstance(request, DecisionRequest) or not request.legal_actions:
        raise ValueError("public-memory baseline requires a nonempty real Race request")
    return request


def public_key(request):
    """No scores, turn clock, decision IDs, live engine, queue or RNG state."""
    request = _race_request(request)
    observation = request.observation
    geometry = {
        "player_id": request.player_id, "window": request.window,
        "current_player_id": observation.current_player_id,
        "track_name": observation.track_name, "finish_position": observation.finish_position,
        "second_corner_position": observation.second_corner_position,
        "trip_positions": observation.trip_positions,
        "racers": [asdict(r) for r in sorted(observation.racers, key=lambda r: r.racer_id)],
        "finish_order": observation.finish_order, "latest_roll": observation.latest_roll,
        "current_move_value": observation.current_move_value,
        "magician_rerolls_remaining": observation.magician_rerolls_remaining,
        "dicemonger_rerolls_remaining": observation.dicemonger_rerolls_remaining,
        "legal_actions": [asdict(action) for action in request.legal_actions],
    }
    return _canonical(geometry)


class FirstLegalMemoryRacePolicy:
    policy_id = RACE_POLICY_ID

    def __init__(self):
        self.race_index = None
        self._memory = OrderedDict()
        self.last_audit = None

    def begin_race(self, race_index):
        if type(race_index) is not int or race_index not in (1, 2, 3, 4):
            raise ValueError("baseline reset requires a real public Race index")
        changed = race_index != self.race_index
        if changed:
            self._memory.clear()
            self.last_audit = None
            self.race_index = race_index
        return changed

    def choose(self, request):
        request = _race_request(request)
        key = public_key(request)
        priority = tuple(action for action in request.legal_actions if action.kind == ActionKind.KEEP)
        priority += tuple(action for action in request.legal_actions if action.kind != ActionKind.KEEP)
        hit = key in self._memory
        counts = self._memory.pop(key) if hit else (0,) * len(priority)
        selected = min(range(len(priority)), key=lambda index: counts[index])
        after = list(counts)
        after[selected] += 1
        floor = min(after)
        normalized = tuple(value - floor for value in after)
        self._memory[key] = normalized
        evicted = None
        if len(self._memory) > MEMORY_CAPACITY:
            old_key, _ = self._memory.popitem(last=False)
            evicted = hashlib.sha256(old_key).hexdigest()
        # The policy's operational state remains bounded. Full persisted Game
        # records retain every actual request/action and reconstructed audit.
        self.last_audit = {
            "scheme": SCHEME, "implementation_sha256": IMPORTED_IMPLEMENTATION_SHA256,
            "scope": "race", "race_index": self.race_index,
            "public_geometry_sha256": hashlib.sha256(key).hexdigest(), "cache_hit": hit,
            "relative_action_counts_before": list(counts), "relative_action_counts_after": list(normalized),
            "priority_actions": [asdict(action) for action in priority], "selected_priority_index": selected,
            "chosen_action": asdict(priority[selected]), "memory_entries": len(self._memory),
            "evicted_public_geometry_sha256": evicted,
        }
        return priority[selected]


class FirstLegalMemoryGamePolicy:
    policy_id = GAME_POLICY_ID

    def __init__(self, race_policy):
        self.race_policy = race_policy
        self.last_audit = None

    def choose(self, request):
        if isinstance(request, (DraftRequest, EntryRequest)):
            data = asdict(request)
        elif isinstance(request, dict):
            data = request
        else:
            raise ValueError("public-memory baseline requires a nonempty real Game request")
        observation = data["observation"]
        entry = observation["phase"] == "entry_selection"
        expected = EntryRequest if entry else DraftRequest
        if (observation["phase"] not in {"entry_selection", "draft"}
                or set(data) != {field.name for field in fields(expected)}
                or set(observation) != {field.name for field in fields(PublicGameObservation)}
                or not data["legal_card_ids"]):
            raise ValueError("public-memory baseline requires an exact public Game request")
        reset = self.race_policy.begin_race(observation["race_index"]) if entry else False
        choice = data["legal_card_ids"][0]
        self.last_audit = {
            "scheme": SCHEME, "implementation_sha256": IMPORTED_IMPLEMENTATION_SHA256,
            "scope": "game", "phase": observation["phase"],
            "race_index": observation["race_index"], "race_memory_reset": reset,
            "chosen_card_id": choice,
        }
        return choice


def build_pair():
    race = FirstLegalMemoryRacePolicy()
    return FirstLegalMemoryGamePolicy(race), race
