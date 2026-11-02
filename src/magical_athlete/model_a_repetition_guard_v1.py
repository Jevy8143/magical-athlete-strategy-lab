"""Opt-in public-memory guard; frozen model tables and rules are unchanged."""
import hashlib
from pathlib import Path

from .decisions import Action, ActionKind
from .model_a import PublicDecisionMemory, ARacePolicy

GUARD_VERSION = "all-legal-exhaustion-release-v1"
IMPORTED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class AllLegalExhaustionMemory(PublicDecisionMemory):
    def __init__(self, capacity, repetition_limit):
        super().__init__(capacity)
        self.repetition_limit = repetition_limit
        self.last_guard = None

    def prepare(self, context_id, encoded):
        key, actual_counts = super().prepare(context_id, encoded)
        optional_race = (encoded.scope == "race" and len(encoded.actions) > 1
                         and any(isinstance(action, Action) and action.kind == ActionKind.KEEP
                                 for action in encoded.actions))
        release = optional_race and all(value >= self.repetition_limit for value in actual_counts)
        effective = (0,) * len(actual_counts) if release else actual_counts
        self.last_guard = {"scheme": GUARD_VERSION, "released": release,
                           "observed_counts_before": tuple(actual_counts),
                           "effective_counts_before": tuple(effective)}
        # Never reset the underlying bounded history or modify a legal menu.
        # Returning effective counts makes the unchanged v3 selector retain
        # its original posterior proposal instead of redirecting it to KEEP.
        return key, effective


class AllLegalExhaustionRacePolicy(ARacePolicy):
    def __init__(self, model, context_id, seed, mode="thompson", shared_context=None, race_index=None):
        super().__init__(model, context_id, seed, mode, shared_context, race_index)
        self.memory = AllLegalExhaustionMemory(model.config.history_capacity,
                                              model.config.repetition_limit)

    def choose(self, request):
        choice = super().choose(request)
        guard = self.memory.last_guard
        audit = self.record_audits[-1]
        # Native v3 probability math used effective counts. State both inputs;
        # do not misreport effective zeros as the actual public action history.
        audit["repetition_guard"] = GUARD_VERSION
        audit["repetition_guard_implementation_sha256"] = IMPORTED_IMPLEMENTATION_SHA256
        audit["history_guard_released"] = guard["released"]
        audit["history_effective_action_counts_before"] = list(guard["effective_counts_before"])
        audit["history_action_counts_before"] = list(guard["observed_counts_before"])
        audit["repetition_count"] = guard["observed_counts_before"][audit["base_selected_index"]]
        audit["history_guard_release_reason"] = ("all_current_legal_actions_at_quota" if guard["released"] else None)
        return choice
