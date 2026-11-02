"""Versioned public turn-local Duel KEEP sweeps over the unchanged MC policy.

Only public requests are read. The real legal menu is never changed, no engine
state is inspected, and the bounded history commits the actually chosen action.
The original v1 guard remains an immutable, separately executable behavior.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from .decisions import ActionKind
from .experiment_spec import canonical_json
from .model_a_features import encode_request, normalize_request
from .model_a_repetition_guard_v1 import AllLegalExhaustionMemory, AllLegalExhaustionRacePolicy
from .model_a_repetition_guard_v1 import IMPORTED_IMPLEMENTATION_SHA256 as IMPORTED_BASE_SHA256

GUARD_VERSION = "public-turn-local-duel-keep-sweep-v2"
BASE_IMPLEMENTATION_SHA256 = "811df3e1e55807456b88b4e676e95278d12849cf00b76487ee7381f0dfa31345"
INITIAL_KEEP_WINDOWS = 16
MAXIMUM_KEEP_WINDOWS = 128
GEOMETRY_CAPACITY = 4096
GUARD_PARAMETERS = (("initial_keep_windows", INITIAL_KEEP_WINDOWS),
                    ("maximum_keep_windows", MAXIMUM_KEEP_WINDOWS),
                    ("same_geometry_threshold", 2), ("turn_duel_window_fallback_threshold", 8),
                    ("geometry_capacity", GEOMETRY_CAPACITY))
IMPORTED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def validate_base_implementation():
    actual = hashlib.sha256(Path(__file__).with_name("model_a_repetition_guard_v1.py").read_bytes()).hexdigest()
    if actual != BASE_IMPLEMENTATION_SHA256 or actual != IMPORTED_BASE_SHA256:
        raise ValueError("V2 base guard implementation differs from its frozen V1 dependency")


class ActualActionMemory(AllLegalExhaustionMemory):
    """Stage a selector proposal, then commit exactly one actual public action."""

    def __init__(self, capacity, repetition_limit):
        super().__init__(capacity, repetition_limit)
        self.pending = None

    def remember(self, key, action):
        if self.pending is not None:
            raise ValueError("uncommitted previous public action")
        self.pending = (key, action)

    @property
    def size(self):
        return min(self.capacity, super().size + (self.pending is not None))

    def commit(self, actual_action_key):
        if self.pending is None:
            raise ValueError("no selector proposal to commit")
        key, _proposal = self.pending
        self.pending = None
        super().remember(key, actual_action_key)


class PublicDuelSweepRacePolicy(AllLegalExhaustionRacePolicy):
    """Drop-in A Race policy, with a separately frozen public liveness guard."""

    def __init__(self, model, context_id, seed, mode="thompson", shared_context=None, race_index=None):
        validate_base_implementation()
        super().__init__(model, context_id, seed, mode, shared_context, race_index)
        self.memory = ActualActionMemory(model.config.history_capacity, model.config.repetition_limit)
        self.public_domain = None
        self.geometry_counts = OrderedDict()
        self.duel_windows = 0
        self.sweep_active = False
        self.remaining_keep_windows = 0
        self.next_sweep_length = INITIAL_KEEP_WINDOWS
        self.seeded_retry_count = 0

    def _reset_domain(self, domain):
        if domain != self.public_domain:
            self.public_domain = domain
            self.geometry_counts.clear()
            self.duel_windows = 0
            self.sweep_active = False
            self.remaining_keep_windows = 0
            self.next_sweep_length = INITIAL_KEEP_WINDOWS
            self.seeded_retry_count = 0

    def _see_geometry(self, raw):
        observation = raw["observation"]
        # No scores, decision IDs, future dice, pending queues or RNG state.
        geometry = canonical_json({"track_name": observation["track_name"],
            "racers": observation["racers"], "finish_order": observation["finish_order"],
            "legal_actions": raw["legal_actions"]})
        seen = self.geometry_counts.pop(geometry, 0) + 1
        self.geometry_counts[geometry] = seen
        if len(self.geometry_counts) > GEOMETRY_CAPACITY:
            self.geometry_counts.popitem(last=False)
        return geometry, seen

    def choose(self, request):
        index = self.shared_context.race_index
        if index is None:
            raise ValueError("model A race policy requires the actual public race index")
        raw = normalize_request(request, race_index=index)
        observation = raw["observation"]
        domain = (index, observation["turn_number"], observation["current_player_id"])
        self._reset_domain(domain)
        encoded = encode_request(raw, race_index=index)
        proposal = super().choose(raw)
        proposal_audit = self.record_audits.pop()
        legal = encoded.actions
        eligible = (raw["window"] == "duel"
                    and any(action.kind == ActionKind.KEEP for action in legal)
                    and any(action.kind == ActionKind.DUEL for action in legal)
                    and all(action.kind in {ActionKind.KEEP, ActionKind.DUEL} for action in legal))
        actual, forced_keep, seen, geometry = proposal, False, None, None
        reason = "base_policy_non_duel_window"
        before = self.remaining_keep_windows
        if eligible:
            geometry, seen = self._see_geometry(raw)
            self.duel_windows += 1
            if not self.sweep_active and (seen >= 2 or self.duel_windows >= 8):
                self.sweep_active = True
                self.remaining_keep_windows = self.next_sweep_length
                reason = "public_repeat_enters_keep_sweep"
            if self.sweep_active and self.remaining_keep_windows > 0:
                actual = next(action for action in legal if action.kind == ActionKind.KEEP)
                forced_keep = True
                if reason != "public_repeat_enters_keep_sweep":
                    reason = "continue_keep_sweep"
                self.remaining_keep_windows -= 1
            elif self.sweep_active:
                self.seeded_retry_count += 1
                reason = "one_real_seeded_base_proposal_between_sweeps"
                self.next_sweep_length = min(MAXIMUM_KEEP_WINDOWS, 2 * self.next_sweep_length)
                self.remaining_keep_windows = self.next_sweep_length
            else:
                reason = "base_policy_before_public_repeat"
        if actual not in legal:
            raise ValueError("guard action is outside its actual legal menu")
        selected = legal.index(actual)
        self.memory.commit(encoded.action_keys[selected])
        # Posterior diagnostics and actual previous history remain meaningful.
        # Proposal choice/probability details are retained in one detached nest.
        proposal_only = {"base_selected_index", "fallback_kind", "fallback_action_indices",
            "fallback_candidates_by_proposed", "fallback_selected_index", "repetition_count",
            "history_guard_released", "history_guard_release_reason", "history_effective_action_counts_before"}
        audit = {key: value for key, value in proposal_audit.items() if key not in proposal_only}
        audit.update(selected_index=selected, repetition_guard=GUARD_VERSION,
            repetition_guard_implementation_sha256=IMPORTED_IMPLEMENTATION_SHA256,
            base_proposal_audit=json.loads(canonical_json(proposal_audit)),
            actual_action=json.loads(canonical_json(actual)),
            history_size_after=self.memory.size,
            public_liveness={"parameters": dict(GUARD_PARAMETERS), "public_domain": list(domain),
                "public_geometry_sha256": hashlib.sha256(geometry.encode()).hexdigest() if geometry is not None else None,
                "public_geometry_seen": seen, "geometry_cache_size": len(self.geometry_counts),
                "turn_duel_windows": self.duel_windows, "sweep_active": self.sweep_active,
                "sweep_windows_before": before, "sweep_windows_after": self.remaining_keep_windows,
                "next_sweep_length": self.next_sweep_length, "seeded_base_retry_count": self.seeded_retry_count,
                "reason": reason, "deterministic_keep": forced_keep, "overrode_proposal": actual != proposal,
                "base_history_commits_actual_actions": True})
        if forced_keep:
            probabilities = [float(i == selected) for i in range(len(legal))]
            audit.update(behavior_probability=1.0,
                behavior_probability_source="deterministic_public_turn_duel_keep_sweep",
                action_probabilities=probabilities, behavior_probability_lower_bound=1.0,
                mixture_component="public_liveness_keep", repetition_triggered=True,
                uniform_exploration_taken=False)
        self.record_audits.append(audit)
        return actual
