"""Mean-only public Fish KEEP protection, retaining frozen V3 Duel behavior.

This is an explicit EVAL/DIAGNOSTIC policy version, never a TRAIN guard.
Only public requests and bounded actual-action history affect its choices.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

from .decisions import ActionKind
from .experiment_spec import canonical_json
from .model_a_features import encode_request, normalize_request
from .model_a_repetition_guard_v3 import ActualActionMemory, PublicDuelRetryRacePolicy
from .model_a_repetition_guard_v3 import IMPORTED_IMPLEMENTATION_SHA256 as IMPORTED_V3_SHA256

GUARD_VERSION = 'public-turn-local-duel-retry-fish-keep-v4'
V3_IMPLEMENTATION_SHA256 = '397be741a370b246919461cb4ed42839b0d31750029d1433794020a8ded787a8'
GEOMETRY_CAPACITY = 4096
SAME_GEOMETRY_THRESHOLD = 2
IMPORTED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def validate_base_implementation():
    actual = hashlib.sha256(Path(__file__).with_name('model_a_repetition_guard_v3.py').read_bytes()).hexdigest()
    if actual != V3_IMPLEMENTATION_SHA256 or actual != IMPORTED_V3_SHA256:
        raise ValueError('V4 requires its exact frozen V3 base implementation')

class DeferredActualActionMemory(ActualActionMemory):
    """Stage V3's choice until the mean wrapper selects the real action."""
    def __init__(self, capacity, repetition_limit):
        super().__init__(capacity, repetition_limit)
        self.staged_v3_commit = None

    def commit(self, action_key):
        if self.staged_v3_commit is not None:
            raise ValueError('uncommitted previous V3 proposal')
        self.staged_v3_commit = action_key

    def commit_actual(self, action_key):
        if self.staged_v3_commit is None:
            raise ValueError('missing V3 proposal before actual action')
        super().commit(action_key)
        self.staged_v3_commit = None

class PublicDuelFishKeepMeanRacePolicy(PublicDuelRetryRacePolicy):
    """One normal V3 proposal; repeated public Fish windows then choose KEEP."""
    def __init__(self, model, context_id, seed, mode='mean', shared_context=None, race_index=None):
        if mode != 'mean':
            raise ValueError('V4 is mean-only EVAL/DIAGNOSTIC; use frozen V3 for Thompson TRAIN')
        validate_base_implementation()
        super().__init__(model, context_id, seed, mode, shared_context, race_index)
        self.memory = DeferredActualActionMemory(model.config.history_capacity, model.config.repetition_limit)
        self.fish_domain = None
        self.fish_geometries = OrderedDict()
        self.fish_keep_active = False

    def choose(self, request):
        index = self.shared_context.race_index
        if index is None:
            raise ValueError('V4 requires the actual public race index')
        raw = normalize_request(request, race_index=index)
        observation = raw['observation']
        domain = (index, observation['turn_number'], observation['current_player_id'])
        if domain != self.fish_domain:
            self.fish_domain = domain
            self.fish_geometries.clear()
            self.fish_keep_active = False
        encoded = encode_request(raw, race_index=index)
        proposal = super().choose(raw)
        v3_audit = self.record_audits.pop()
        legal = encoded.actions
        eligible = (raw['window'] == 'move_start'
                    and any(action.kind == ActionKind.KEEP for action in legal)
                    and any(action.kind == ActionKind.SUCKERFISH_FOLLOW for action in legal)
                    and all(action.kind in (ActionKind.KEEP, ActionKind.SUCKERFISH_FOLLOW) for action in legal))
        geometry, seen = None, None
        if eligible:
            # Scores, decision IDs, pending queues and RNG state are excluded.
            geometry = canonical_json({'track_name': observation['track_name'], 'racers': observation['racers'],
                                       'finish_order': observation['finish_order'], 'legal_actions': raw['legal_actions']})
            seen = self.fish_geometries.pop(geometry, 0) + 1
            self.fish_geometries[geometry] = seen
            if len(self.fish_geometries) > GEOMETRY_CAPACITY:
                self.fish_geometries.popitem(last=False)
            if seen >= SAME_GEOMETRY_THRESHOLD:
                self.fish_keep_active = True
        actual = next(action for action in legal if action.kind == ActionKind.KEEP) if eligible and self.fish_keep_active else proposal
        selected = legal.index(actual)
        self.memory.commit_actual(encoded.action_keys[selected])
        # Keep the whole V3 proposal detached; its selected index/probability
        # is not the actual behavior when this wrapper chooses KEEP.
        audit = dict(v3_audit)
        audit.update(selected_index=selected, actual_action=json.loads(canonical_json(actual)),
                     repetition_guard=GUARD_VERSION,
                     repetition_guard_implementation_sha256=IMPORTED_IMPLEMENTATION_SHA256,
                     base_V3_proposal_audit=json.loads(canonical_json(v3_audit)),
                     history_size_after=self.memory.size,
                     public_fish_keep={'public_domain': list(domain), 'eligible': eligible,
                         'geometry_seen': seen, 'geometry_cache_size': len(self.fish_geometries),
                         'geometry_sha256': hashlib.sha256(geometry.encode()).hexdigest() if geometry else None,
                         'same_geometry_threshold': SAME_GEOMETRY_THRESHOLD, 'capacity': GEOMETRY_CAPACITY,
                         'active_until_public_turn_changes': self.fish_keep_active,
                         'actual_override': actual != proposal, 'actual_action_history_committed': True})
        if eligible and self.fish_keep_active:
            audit.update(behavior_probability=1.0, behavior_probability_lower_bound=1.0,
                         behavior_probability_source='deterministic_public_mean_repeat_fish_KEEP',
                         action_probabilities=[float(i == selected) for i in range(len(legal))],
                         mixture_component='public_mean_repeat_fish_KEEP', repetition_triggered=True,
                         uniform_exploration_taken=False)
        self.record_audits.append(audit)
        return actual
