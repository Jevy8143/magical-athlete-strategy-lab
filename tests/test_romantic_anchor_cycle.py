"""Real Wild anchor/Copy closure and independent all-choice FIFO evidence.

The role matrix starts a real normal turn and obtains every pending task from
movement and stop qualification.  Local finite-edge tests are explicitly
synthetic pre-resolved states used to exercise real movement, not certificates.
No test uses an event budget as evidence that a normal race has ended.
"""

import copy
import itertools
import json
from collections import deque
from dataclasses import asdict, replace

import pytest

from magical_athlete import romantic_anchor_cycle as proof
from magical_athlete.decisions import ActionKind, DecisionWindow
from magical_athlete.engine import DiagnosticLimitExceeded, RaceEngine, replay_trace
from magical_athlete.events import PendingEffect, RaceEvent
from magical_athlete.racer import Ability, Racer
from magical_athlete.scoring import ScoreLedger
from magical_athlete.state import RacerStatus
from magical_athlete.track import MILD_MILE, WILD_WILDS


# Independent explicit domains.  Do not let a new production whitelist member
# silently become an asserted-positive fixture.  MOUTH/CENTAUR/LEAPTOAD are
# checked while Copy itself moves: no other racer occupies12/14/16 or is passed.
ANCHORS = (
    Ability.SKIPPER, Ability.HECKLER, Ability.INCHWORM, Ability.LACKEY,
    Ability.ALCHEMIST, Ability.BLIMP, Ability.CENTAUR, Ability.CHEERLEADER,
    Ability.COACH, Ability.DICEMONGER, Ability.FLIP_FLOP, Ability.GENIUS,
    Ability.GUNK, Ability.HARE, Ability.HYPNOTIST, Ability.LEAPTOAD,
    Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN, Ability.MASTERMIND,
    Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.STICKLER, Ability.THIRD_WHEEL,
)
SPECTATORS = (
    Ability.ALCHEMIST, Ability.BABA_YAGA, Ability.BANANA, Ability.BLIMP,
    Ability.CENTAUR, Ability.CHEERLEADER, Ability.COACH, Ability.DICEMONGER,
    Ability.FLIP_FLOP, Ability.GENIUS, Ability.GUNK, Ability.HARE, Ability.HECKLER,
    Ability.HUGE_BABY, Ability.HYPNOTIST, Ability.INCHWORM, Ability.LACKEY,
    Ability.LEAPTOAD, Ability.LEGS, Ability.LOVEABLE_LOSER, Ability.MAGICIAN,
    Ability.MASTERMIND, Ability.MOUTH, Ability.PARTY_ANIMAL, Ability.ROCKET_SCIENTIST,
    Ability.SISYPHUS, Ability.SKIPPER, Ability.STICKLER, Ability.SUCKERFISH, Ability.THIRD_WHEEL,
)
COMBINATIONS = tuple((p, s) for p in ANCHORS for s in SPECTATORS if p != s)
MODES = ("anchor", "romantic", "keep", "mixed")


class FiniteDice:
    def __init__(self, prefix):
        self.prefix, self.used = tuple(prefix), []
        self.values = (*self.prefix, 6)

    def roll(self):
        assert len(self.used) < len(self.prefix), "closure consumed an unrequested future die"
        value = self.values[len(self.used)]
        self.used.append(value)
        return value


class QueueHistory(deque):
    def __init__(self, values=()):
        super().__init__(values)
        self.appended, self.popped, self.cleared = [], [], []

    def append(self, value):
        self.appended.append(value)
        return super().append(value)

    def popleft(self):
        value = super().popleft()
        self.popped.append(value)
        return value

    def clear(self):
        self.cleared.append(tuple(self))
        return super().clear()


class CopyChoices:
    policy_id = "romantic-anchor-actual-legal-copy-branches"

    def __init__(self, romantic, anchor, mode):
        self.romantic, self.anchor, self.mode = romantic, anchor, mode
        self.calls = []

    def choose(self, request):
        assert request.window == DecisionWindow.COPY_CHOICE
        self.calls.append(request)
        if self.mode == "keep":
            kept = next((a for a in request.legal_actions if a.kind == ActionKind.KEEP), None)
            if kept is not None:
                return kept
        if self.mode == "mixed" and len(self.calls) % 3 == 0:
            kept = next((a for a in request.legal_actions if a.kind == ActionKind.KEEP), None)
            if kept is not None:
                return kept
        if self.mode == "old_copy":
            target = self.romantic if len(self.calls) <= 2 else self.anchor
        elif self.mode == "romantic":
            target = self.romantic
        elif self.mode == "mixed":
            target = self.romantic if len(self.calls) % 2 else self.anchor
        else:
            target = self.anchor
        return next(a for a in request.legal_actions if a.kind == ActionKind.COPY_TARGET and a.target_id == target)


def plain(value):
    return json.loads(json.dumps(value))


def scene(*, anchor=Ability.THIRD_WHEEL, spectator=Ability.SKIPPER,
          ids=(2, 3, 1, 0), phase=12, mode="romantic", actor="romantic",
          profile="RC", budget=1024, prior_score=False):
    r, c, p, s = ids
    assert len({Ability.ROMANTIC, Ability.COPY_CAT, anchor, spectator}) == 4
    # For arbitrary P/S powers, R's real main20->22 is unaffected except for
    # Gunk: a leading P=Gunk is borrowed by C too (two real -1 sources),
    # whereas S=Gunk contributes one -1. No P starts its own action window.
    # The explicit ThirdWheel variant reproduces the actual21->24 Arrow22
    # main arrival and gives either notification order around current P.
    if actor == "anchor":
        assert anchor == Ability.THIRD_WHEEL and spectator == Ability.SKIPPER
        positions, current, roll = (22, phase, 21, 17), p, 3
        order = [p, r, c, s] if profile == "RC" else [p, c, r, s]
    else:
        assert actor == "romantic" and profile == "RC"
        positions, current = (20, phase, 22, 17), r
        roll = 4 if anchor == Ability.GUNK else 3 if spectator == Ability.GUNK else 2
        order = [r, c, p, s]
    racers = [Racer(r, "Romantic", positions[0]), Racer(c, "Copy Cat", positions[1]),
              Racer(p, anchor.value, positions[2]), Racer(s, spectator.value, positions[3])]
    ledger = ScoreLedger(ids)
    if prior_score:
        ledger.record(s, 1, 3, source="earlier complete Race", reason="existing score", phase="after_race")
        ledger.record(c, 1, 2, source="earlier complete Race", reason="existing points before penalty", phase="after_race")
        ledger.record(c, 1, -1, source="earlier complete Race", reason="existing penalty", phase="during_race")
    dice, policy = FiniteDice([roll]), CopyChoices(r, p, mode)
    engine = RaceEngine(WILD_WILDS, racers, order, dice, policies={c: policy},
                        race_index=2, max_events=budget, score_ledger=ledger)
    engine.pending_effects = QueueHistory(engine.pending_effects)
    return engine, policy, dice, current, ids


def execute_scene(**kwargs):
    engine, policy, dice, current, ids = scene(**kwargs)
    engine.take_turn(engine.racers[current])
    return engine, policy, dice, current, ids


def check_scene(result):
    engine, policy, dice, current, ids = result
    r, c, p, s = ids
    event, = [e for e in engine.events if e.reason == proof.REASON]
    certificate = json.loads(event.operation)
    original, executed, fresh = (certificate[name] for name in
                                ("original_pending_ids", "executed_pending_ids", "new_pending_ids"))
    assert original and executed[:len(original)] == original
    assert len(set(executed)) == len(executed)
    assert fresh and set(original).isdisjoint(fresh) and set(executed).isdisjoint(fresh)
    assert event.amount == certificate["queue_after"] == len(fresh)
    assert certificate["queue_before"] == len(original)
    popped = [item.queued_sequence for item in engine.pending_effects.popped]
    start = popped.index(original[0])
    assert popped[start:start + len(executed)] == executed
    assert not set(fresh).intersection(popped)
    assert any([item.queued_sequence for item in queue] == fresh for queue in engine.pending_effects.cleared)
    by_id = {e.sequence: e for e in engine.events}
    actual_jobs = [e for e in engine.events if e.kind == "ability_effect" and e.parent_sequence in executed]
    assert [e.parent_sequence for e in actual_jobs] == executed
    assert certificate["job_counts"] == {"R": sum(e.racer_id == r for e in actual_jobs),
                                         "C": sum(e.racer_id == c for e in actual_jobs)}
    assert certificate["job_counts"]["R"] >= 1
    assert sum(by_id[q].racer_id == r for q in fresh) == 1
    assert certificate["scores_unchanged"] and certificate["mechanical_state_returned"]
    assert certificate["finite_configurations"] == 16 and certificate["all_legal_action_edges"] == 48
    period = engine.events[event.parent_sequence - 1:event.sequence - 1]
    assert not any(e.kind in {"roll", "reroll", "duel_roll", "score", "trip", "recover",
                              "finish", "eliminate", "effect_cancelled", "race_end"} for e in period)
    assert certificate["copy_decision_event_ids"] == [e.sequence for e in period if e.kind == "decision"]
    for record in engine.decisions:
        assert record.action in record.request.legal_actions
        if event.parent_sequence <= record.event_sequence < event.sequence:
            assert record.request.window == DecisionWindow.COPY_CHOICE
    assert (engine.racers[r].position, engine.racers[p].position, engine.racers[s].position) == (22, 22, 17)
    assert engine.racers[c].position in (12, 14)
    assert engine.racers[c].copied_target in (p, r)
    assert engine.racers[c].copied_ability == (engine.racers[p].ability if engine.racers[c].copied_target == p else Ability.ROMANTIC)
    assert not engine.pending_effects and engine.termination_reason is None and not engine.finish_order
    assert len(engine.turns) == 1 and engine.turns[0].racer_id == current
    assert not engine._preparing and not engine._turn_in_progress
    assert dice.used == list(dice.prefix) and dice.values[len(dice.used):] == (6,)
    trace = plain(engine.trace())
    assert plain(replay_trace(trace).trace()) == trace
    return event, certificate


@pytest.mark.parametrize("anchor,spectator", COMBINATIONS,
                         ids=[f"{p.value}-{s.value}" for p, s in COMBINATIONS])
def test_every_distinct_original_anchor_and_spectator_has_real_legal_branches_and_exact_replay(anchor, spectator):
    # Four actual policies select every available action kind. Every pair has
    # an independent real engine/replay; no production whitelist decides this.
    for mode in MODES:
        check_scene(execute_scene(anchor=anchor, spectator=spectator, mode=mode))


@pytest.mark.parametrize("ids", tuple(itertools.permutations(range(4))))
@pytest.mark.parametrize("profile", ("RC", "CR"))
@pytest.mark.parametrize("phase", (12, 14))
def test_real_anchor_main_arrow_qualification_all_ids_copy_phases_and_both_notification_orders(ids, profile, phase):
    event, certificate = check_scene(execute_scene(ids=ids, phase=phase, profile=profile,
                                                   actor="anchor", mode="mixed", prior_score=True))
    assert certificate["clockwise_profile"] == profile
    assert event.parent_sequence > 1


@pytest.mark.parametrize("profile", ("RC", "CR"))
@pytest.mark.parametrize("mode", MODES)
def test_pure_romantic_or_anchor_keep_and_changing_targets_preserve_full_fifo_and_prior_scores(profile, mode):
    check_scene(execute_scene(actor="anchor", profile=profile, mode=mode, prior_score=True))


def independent_step(state, action, profile):
    position, copied_r, word = state
    if word[0] == "C":
        position = 14 if position == 12 else 12
        born = ""
    else:
        born = "R" if not copied_r else ("RC" if profile == "RC" else "CR")
    next_power = copied_r if action == "keep" else action == "copy_r"
    return position, next_power, word[1:] + born


@pytest.mark.parametrize("profile", ("RC", "CR"))
def test_independent_exact_fifo_all_action_closure_naturally_exhausts_16_states_and_48_edges(profile):
    seeds = {(p, power, word) for p in (12, 14) for power in (False, True) for word in ("R", "RC")}
    seen, todo, edges = set(seeds), deque(sorted(seeds)), set()
    while todo:
        before = todo.popleft()
        for action in ("keep", "copy_p", "copy_r"):
            after = independent_step(before, action, profile)
            assert after[2].count("R") == 1 and after[2]
            edges.add((before, action, after))
            if after not in seen:
                seen.add(after)
                todo.append(after)
    words = {"R", "RC", "CR", "CRC" if profile == "RC" else "CCR"}
    assert seen == {(p, power, word) for p in (12, 14) for power in (False, True) for word in words}
    assert len(seen) == 16 and len(edges) == 48 and max(len(x[2]) for x in seen) == 3
    actual_states, actual_edges, digest = proof._finite_proof(profile)
    assert set(actual_states) == seen and set(actual_edges) == edges and len(digest) == 64


@pytest.mark.parametrize("profile", ("RC", "CR"))
@pytest.mark.parametrize("phase", (12, 14))
@pytest.mark.parametrize("copied_r", (False, True))
@pytest.mark.parametrize("job", ("R", "C"))
@pytest.mark.parametrize("action", ("keep", "copy_p", "copy_r"))
def test_real_single_qualified_body_matches_independent_all_branch_algebra(profile, phase, copied_r, job, action):
    # Explicit synthetic pre-resolved state: tests real movement/capture/Copy
    # code only, and never calls the certificate capture/completed helpers.
    r, c, p, s = 2, 3, 1, 0
    racers = [Racer(r, "Romantic", 22), Racer(c, "Copy Cat", phase),
              Racer(p, "Third Wheel", 22), Racer(s, "Skipper", 17)]
    mode = {"keep": "keep", "copy_p": "anchor", "copy_r": "romantic"}[action]
    policy = CopyChoices(r, p, mode)
    engine = RaceEngine(WILD_WILDS, racers, [p, r, c, s] if profile == "RC" else [p, c, r, s],
                        FiniteDice([]), policies={c: policy}, max_events=256)
    engine.events.clear()
    engine.decisions.clear()
    policy.calls.clear()
    copy_actor = engine.racers[c]
    copy_actor.copied_target = r if copied_r else p
    copy_actor.copied_ability = Ability.ROMANTIC if copied_r else Ability.THIRD_WHEEL
    engine._turn_in_progress = True
    engine.pending_effects.clear()
    actor = engine.racers[r if job == "R" else c]
    original_score = engine.score_ledger.snapshot()
    engine._move(actor, 2, movement_kind="ability", source_id=actor.racer_id,
                 ability=Ability.ROMANTIC.value, parent_sequence=999, notify_ability=True)
    children = "".join("R" if q.racer_id == r else "C" for q in engine.pending_effects)
    expected = independent_step((phase, copied_r, job), action, profile)
    assert (copy_actor.position, copy_actor.copied_target == r, children) == expected
    assert engine.score_ledger.snapshot() == original_score
    assert len(policy.calls) == len(engine.decisions) == 1
    request = engine.decisions[0].request
    assert request.window == DecisionWindow.COPY_CHOICE and request.player_id == c
    assert {a.kind for a in request.legal_actions} == {ActionKind.KEEP, ActionKind.COPY_TARGET}
    assert {a.target_id for a in request.legal_actions if a.kind == ActionKind.COPY_TARGET} == {p, r}
    assert next(e for e in engine.events if e.kind == "ability_effect").parent_sequence == 999
    assert not any(e.kind in {"score", "roll", "trip", "finish", "eliminate"} for e in engine.events)
    assert engine.racers[p].position == 22 and engine.racers[s].position == 17


def observe(monkeypatch):
    captures, closures = [], []
    capture, completed = proof.capture, proof.completed
    live_engine = None

    def record_capture(engine, state=None):
        nonlocal live_engine
        snapshot = capture(engine, state)
        if live_engine is None:
            live_engine = engine
        if engine is live_engine:
            captures.append((copy.deepcopy(engine), copy.deepcopy(snapshot)))
        return snapshot

    def record_completed(engine, snapshot):
        certificate = completed(engine, snapshot)
        if certificate and engine is live_engine:
            closures.append((copy.deepcopy(engine), copy.deepcopy(snapshot), certificate))
        return certificate

    monkeypatch.setattr(proof, "capture", record_capture)
    monkeypatch.setattr(proof, "completed", record_completed)
    return captures, closures, completed


def test_real_old_copy_qualification_still_moves_after_copy_changes_to_the_anchor(monkeypatch):
    captures, closures, _ = observe(monkeypatch)
    engine, _, _, _, ids = execute_scene(mode="old_copy")
    check_scene((engine, engine.policies[ids[1]], engine.dice, ids[0], ids))
    assert captures and closures
    c, p = ids[1], ids[2]
    power, target, moved = None, None, []
    for event in engine.events:
        if event.kind == "copy_changed" and event.racer_id == c:
            target, power = event.target_id, event.ability
        if event.kind == "ability_effect" and event.racer_id == c and event.ability == "Romantic":
            if target == p and power == "Third Wheel":
                moved.append(event)
    assert moved
    assert all(engine.events[e.parent_sequence - 1].kind == "effect_queued" for e in moved)
    assert any(certificate["job_counts"]["C"] == 0 for _, _, certificate in closures)


@pytest.mark.parametrize("anchor", (Ability.MOUTH, Ability.CENTAUR, Ability.LEAPTOAD, Ability.STICKLER))
@pytest.mark.parametrize("phase", (12, 14))
def test_own_movement_sensitive_borrowed_anchor_is_passive_during_a_real_old_copy_job(anchor, phase):
    result = execute_scene(anchor=anchor, phase=phase, mode="old_copy")
    check_scene(result)
    engine, _, _, _, ids = result
    c, p = ids[1], ids[2]
    copied, target, witnessed = None, None, []
    for event in engine.events:
        if event.kind == "copy_changed" and event.racer_id == c:
            copied, target = event.ability, event.target_id
        elif event.kind == "ability_effect" and event.racer_id == c and event.ability == "Romantic":
            if (copied, target) == (anchor.value, p):
                witnessed.append(event)
    assert witnessed
    move = engine.events[witnessed[0].sequence]
    assert move.kind == "move" and move.before == phase and move.after == phase + 2
    if phase == 14:
        assert any(event.kind == "move" and event.racer_id == c and event.movement_kind == "track"
                   and (event.before, event.after) == (16, 12) for event in engine.events)


@pytest.mark.parametrize("phase", (12, 14))
@pytest.mark.parametrize("mode", MODES)
def test_real_current_copy_has_finished_unlocking_before_heckler_turn_end_enters_the_family(phase, mode):
    r, c, p, s = ids = (2, 3, 1, 0)
    racers = [Racer(r, "Romantic", 22), Racer(c, "Copy Cat", phase, tripped=True),
              Racer(p, "Heckler", 20), Racer(s, "Skipper", 17)]
    dice, policy = FiniteDice([]), CopyChoices(r, p, mode)
    engine = RaceEngine(WILD_WILDS, racers, [c, p, r, s], dice, policies={c: policy},
                        race_index=2, max_events=1024)
    engine.pending_effects = QueueHistory(engine.pending_effects)
    # Actual _start_turn releases Copy's own lock; actual recovery then actual
    # _end_turn moves native Heckler20->22. No test writes or removes a lock.
    engine.take_turn(engine.racers[c])
    _, certificate = check_scene((engine, policy, dice, c, ids))
    assert certificate["clockwise_profile"] == "CR"  # Current Copy is first clockwise.
    assert not engine._copy_locks and engine.turns[0].skipped_reason == "tripped"
    assert any(event.kind == "recover" and event.racer_id == c for event in engine.events)
    assert any(event.kind == "move" and event.racer_id == p and event.ability == "Heckler"
               and (event.before, event.after) == (20, 22) for event in engine.events)


@pytest.mark.parametrize("phase", (12, 14))
@pytest.mark.parametrize("mode", MODES)
def test_real_lackey_roll_response_certifies_before_copy_main_but_the_locked_main_still_executes(phase, mode):
    r, c, p, s = (2, 3, 1, 0)
    dice, policy = FiniteDice([6]), CopyChoices(r, p, mode)
    engine = RaceEngine(WILD_WILDS,
        [Racer(r, "Romantic", 22), Racer(c, "Copy Cat", phase),
         Racer(p, "Lackey", 22), Racer(s, "Skipper", 17)],
        [c, p, r, s], dice, policies={c: policy}, race_index=2, max_events=1024)
    engine.pending_effects = QueueHistory(engine.pending_effects)
    engine.take_turn(engine.racers[c])
    stopped, = [event for event in engine.events if event.reason == proof.REASON]
    certificate = json.loads(stopped.operation)
    assert certificate["clockwise_profile"] == "CR" and certificate["mechanical_state_returned"]
    assert certificate["scores_unchanged"] and certificate["job_counts"]["R"] >= 1
    before_main = [event for event in engine.events if event.kind == "main_move_resolved"]
    assert len(before_main) == 1 and before_main[0].sequence > stopped.sequence
    main = next(event for event in engine.events if event.kind == "move" and event.racer_id == c
                and event.movement_kind == "main")
    assert main.sequence > stopped.sequence and main.amount == 6 and main.before in (12, 14)
    assert main.after == main.before + 6
    assert engine.racers[c].position == main.after and main.after in (18, 20)
    assert len(engine.turns) == 1 and engine.turns[0].roll == engine.turns[0].final_move == 6
    assert any(event.kind == "ability_effect" and event.racer_id == p and event.ability == "Lackey"
               and event.sequence < stopped.sequence for event in engine.events)
    assert dice.used == [6] and not engine._copy_locks and not engine.pending_effects
    assert not engine.finish_order and engine.termination_reason is None
    trace = plain(engine.trace())
    assert plain(replay_trace(trace).trace()) == trace


CORRUPTIONS = (
    "changed_score", "changed_old_score", "changed_cached_total", "extra_roll", "extra_decision",
    "changed_effect_parent", "changed_queue_parent", "wrong_move", "wrong_arrow", "partial_last_event",
    "changed_copy_observation", "illegal_copy_action", "changed_copy_window", "changed_old_event",
    "changed_old_decision", "heterogeneous_pending", "reordered_pending", "duplicate_pending",
    "wrong_pending_source", "wrong_pending_target", "wrong_pending_step", "wrong_pending_kind",
    "wrong_pending_ability", "wrong_pending_trigger", "consumed_original_pending", "no_romantic_pending",
    "outside_copy", "outside_romantic", "outside_anchor", "outside_spectator", "three_occupants",
    "unknown_role", "duplicate_role", "inherited_anchor", "inactive_core", "false_finish", "unknown_prediction",
    "changed_copy_target", "changed_copy_power", "changed_trip", "changed_turns", "unprepared",
    "preparing", "no_active_turn", "changed_current_actor", "copy_lock", "refreshing", "movement", "arrow",
    "changed_next", "changed_genius", "changed_depth", "changed_arrival_counter", "wrong_track",
    "changed_initial_source", "changed_remaining_cards", "changed_previous_winner",
)


@pytest.mark.parametrize("corruption", CORRUPTIONS)
def test_completed_rejects_tampered_sources_bodies_controls_ledger_and_pending_without_clearing(corruption, monkeypatch):
    _, closures, completed = observe(monkeypatch)
    result = execute_scene(actor="anchor", mode="romantic", prior_score=True)
    check_scene(result)
    assert closures
    engine, snapshot, certificate = closures[0]
    r, c, p, s = result[-1]
    pending = tuple(engine.pending_effects)
    if corruption == "changed_score":
        engine.score_ledger.record(c, 2, 1, source="extra", reason="outside proof", phase="during_race")
    elif corruption == "changed_old_score":
        engine.score_ledger._entries[0] = replace(engine.score_ledger._entries[0], reason="altered prefix")
    elif corruption == "changed_cached_total":
        engine.score_ledger._totals[c] += 1
    elif corruption in {"extra_roll", "extra_decision"}:
        engine.events.append(RaceEvent(len(engine.events) + 1, "roll" if corruption == "extra_roll" else "decision", c, amount=1))
    elif corruption in {"changed_effect_parent", "changed_queue_parent", "wrong_move", "wrong_arrow"}:
        kind = "ability_effect" if corruption == "changed_effect_parent" else "effect_queued" if corruption == "changed_queue_parent" else "move"
        index = next(i for i in range(snapshot.event_start, len(engine.events)) if engine.events[i].kind == kind
                     and (corruption != "wrong_arrow" or engine.events[i].movement_kind == "track"))
        event = engine.events[index]
        engine.events[index] = replace(event, **({"parent_sequence": 1} if "parent" in corruption else {"after": event.after + 1}))
    elif corruption == "partial_last_event":
        engine.events.pop()
    elif corruption in {"changed_copy_observation", "illegal_copy_action", "changed_copy_window"}:
        assert certificate["copy_decision_event_ids"]
        index = snapshot.decision_start
        record = engine.decisions[index]
        if corruption == "changed_copy_observation":
            observation = replace(record.request.observation, score_totals=tuple((rid, value + 1) for rid, value in record.request.observation.score_totals))
            engine.decisions[index] = replace(record, request=replace(record.request, observation=observation))
        elif corruption == "changed_copy_window":
            engine.decisions[index] = replace(record, request=replace(record.request, window=DecisionWindow.TURN_START))
        else:
            engine.decisions[index] = replace(record, action=replace(record.action, target_id=s))
    elif corruption == "changed_old_event":
        engine.events[0] = replace(engine.events[0], reason="changed existing prefix")
    elif corruption == "changed_old_decision":
        engine.decisions[0] = replace(engine.decisions[0], event_sequence=0)
    elif corruption == "heterogeneous_pending":
        engine.pending_effects.append(PendingEffect(s, s, 1, 2))
    elif corruption == "reordered_pending":
        assert len(pending) > 1
        engine.pending_effects = deque(reversed(pending))
    elif corruption == "duplicate_pending":
        engine.pending_effects.append(pending[0])
    elif corruption.startswith("wrong_pending_"):
        changes = {"source": {"source_id": s}, "target": {"target_id": s}, "step": {"steps": 1},
                   "kind": {"kind": "trip"}, "ability": {"ability": "Skipper"}, "trigger": {"trigger_sequence": 1}}
        engine.pending_effects[0] = replace(pending[0], **changes[corruption[14:]])
    elif corruption == "consumed_original_pending":
        engine.pending_effects[0] = snapshot.original_pending[0]
    elif corruption == "no_romantic_pending":
        engine.pending_effects = deque(job for job in pending if job.racer_id != r)
    elif corruption.startswith("outside_"):
        engine.racers[{"outside_copy": c, "outside_romantic": r, "outside_anchor": p, "outside_spectator": s}[corruption]].position = 18
    elif corruption == "three_occupants":
        engine.racers[s].position = 22
    elif corruption == "unknown_role":
        engine.racers[s].ability = "future unknown role"
    elif corruption == "duplicate_role":
        engine.racers[s].ability = engine.racers[p].ability
    elif corruption == "inherited_anchor":
        engine.racers[p].inherited_ability = Ability.ROMANTIC
    elif corruption == "inactive_core":
        engine.racers[r].status = RacerStatus.ELIMINATED
    elif corruption == "false_finish":
        engine.finish_order.append(999)
    elif corruption == "unknown_prediction":
        engine.racers[s].prediction = 999
    elif corruption == "changed_copy_target":
        engine.racers[c].copied_target = s
    elif corruption == "changed_copy_power":
        engine.racers[c].copied_ability = Ability.SCOOCHER
    elif corruption == "changed_trip":
        engine.racers[c].tripped = not engine.racers[c].tripped
    elif corruption == "changed_turns":
        engine.racers[s].turns_taken += 1
    elif corruption == "unprepared":
        engine.racers[s].pre_race_resolved = False
    elif corruption == "preparing":
        engine._preparing = True
    elif corruption == "no_active_turn":
        engine._turn_in_progress = False
    elif corruption == "changed_current_actor":
        engine.turn_index = engine.turn_order.index(c)
    elif corruption == "copy_lock":
        engine._copy_locks.add(c)
    elif corruption == "refreshing":
        engine._refreshing_copies = True
    elif corruption == "movement":
        engine._movement_depth = 1
    elif corruption == "arrow":
        engine._arrow_visits = set()
    elif corruption == "changed_next":
        engine.next_turn_racer_id = s
    elif corruption == "changed_genius":
        engine._genius_prediction = 4
    elif corruption == "changed_depth":
        engine._resolution_depth += 1
    elif corruption == "changed_arrival_counter":
        engine._arrival_counter += 1
    elif corruption == "wrong_track":
        engine.track = MILD_MILE
    elif corruption == "changed_initial_source":
        engine.initial_racers[0]["position"] += 1
    elif corruption == "changed_remaining_cards":
        engine.remaining_abilities.append(Ability.GENIUS)
    elif corruption == "changed_previous_winner":
        engine.previous_winners = (Ability.LEGS,)
    else:
        raise AssertionError(corruption)
    assert not completed(engine, snapshot), corruption
    # A refusal is read-only: no queue clearing, score penalty or winner.
    assert engine.termination_reason is None


@pytest.mark.parametrize("qualification", ("romantic_old_exact_two", "copy_old_power"))
def test_capture_rechecks_original_stop_and_historical_copy_power_instead_of_trusting_pending(qualification, monkeypatch):
    captures, closures, _ = observe(monkeypatch)
    check_scene(execute_scene(actor="anchor", mode="romantic" if qualification == "romantic_old_exact_two" else "mixed"))
    if qualification == "romantic_old_exact_two":
        assert closures
        engine = closures[0][0]
        job = next(job for job in engine.pending_effects if job.racer_id == 2)
        # Change a historical C arrival, then let its actual later movement
        # restore the live12/14 position. Final roster/history agreement alone
        # cannot detect this false old three-occupant Romantic qualification.
        old = next(event for event in reversed(engine.events[:job.trigger_sequence - 1])
                   if event.kind == "stop" and event.racer_id == 3)
        assert any(event.kind == "move" and event.racer_id == 3
                   and event.sequence > job.trigger_sequence for event in engine.events)
        move = engine.events[old.parent_sequence - 1]
        assert move.kind == "move" and old.after in (12, 14)
        engine.events[move.sequence - 1] = replace(move, after=22)
        engine.events[old.sequence - 1] = replace(old, after=22)
    else:
        candidates = [(engine, snapshot) for engine, snapshot in captures
                      if any(job.racer_id == 3 for job in snapshot.original_pending)]
        assert candidates
        engine, snapshot = candidates[0]
        job = next(job for job in snapshot.original_pending if job.racer_id == 3)
        history = [event for event in engine.events[:job.trigger_sequence - 1]
                   if event.kind == "copy_changed" and event.racer_id == 3]
        assert history and history[-1].ability == "Romantic"
        old = history[-1]
        assert any(event.kind == "copy_changed" and event.racer_id == 3
                   and event.sequence > job.trigger_sequence for event in engine.events)
        engine.events[old.sequence - 1] = replace(old, target_id=1, ability="Third Wheel")
    assert proof.key(engine) is None
    with pytest.raises(ValueError):
        proof.capture(engine)


@pytest.mark.parametrize("profile", ("RC", "CR"))
@pytest.mark.parametrize("job", ("R", "C"))
def test_actual_event_budget_preserves_popped_unemitted_job_as_diagnostic_and_exactly_replays(profile, job):
    baseline = execute_scene(actor="anchor", profile=profile, mode="romantic")
    check_scene(baseline)
    engine, _, _, _, ids = baseline
    actor = ids[0] if job == "R" else ids[1]
    first = next(event for event in engine.events if event.kind == "ability_effect" and event.racer_id == actor)
    partial, _, _, current, _ = scene(actor="anchor", profile=profile, mode="romantic", budget=first.sequence - 1)
    with pytest.raises(DiagnosticLimitExceeded) as failure:
        partial.take_turn(partial.racers[current])
    trace = plain(failure.value.trace)
    assert trace["diagnostic_failure"] == "event_budget_exceeded" and not trace["completed"]
    assert not trace["finish_order"] and not any(e["reason"] == proof.REASON for e in trace["events"])
    assert first.parent_sequence in [job.queued_sequence for job in partial.pending_effects.popped]
    assert not any(e["kind"] == "ability_effect" and e["parent_sequence"] == first.parent_sequence for e in trace["events"])
    assert plain(replay_trace(trace).trace()) == {key: value for key, value in trace.items() if key != "diagnostic_failure"}
