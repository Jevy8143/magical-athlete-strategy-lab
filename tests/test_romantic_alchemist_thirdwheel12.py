"""Native Alchemist22/ThirdWheel12: actual choices and renewing FIFO jobs."""
import copy
from collections import Counter
from dataclasses import replace
import itertools
import json

import pytest

from magical_athlete import romantic_anchor_cycle as proof
from magical_athlete.decisions import DecisionWindow
from magical_athlete.engine import RaceEngine, replay_trace
from magical_athlete.racer import Ability, Racer
from magical_athlete.scoring import ScoreLedger
from magical_athlete.track import WILD_WILDS
from test_romantic_anchor_cycle import CopyChoices, FiniteDice, QueueHistory, observe, plain


def scene(phase=14, mode="mixed", profile="CR", ids=(0, 3, 1, 2)):
    r, c, p, s = ids
    racers = [Racer(r, "Romantic", 22), Racer(c, "Copy Cat", phase),
              Racer(p, "Alchemist", 18), Racer(s, "Third Wheel", 12)]
    order = [p, c, r, s] if profile == "CR" else [p, r, c, s]
    dice, policy = FiniteDice([4]), CopyChoices(r, p, mode)
    ledger = ScoreLedger(ids)
    ledger.record(s, 1, 3, source="previous completed Race", reason="earlier", phase="after_race")
    engine = RaceEngine(WILD_WILDS, racers, order, dice, policies={c: policy},
                        race_index=2, max_events=1500, score_ledger=ledger)
    engine.pending_effects = QueueHistory(engine.pending_effects)
    return engine, policy, dice, p, ids


@pytest.mark.parametrize("phase,mode,profile", tuple(itertools.product((12, 14), ("keep", "romantic", "anchor", "mixed", "old_copy"), ("RC", "CR"))))
def test_real_main_birth_all_choices_original_jobs_natural_return_and_replay(phase, mode, profile):
    engine, policy, dice, current, ids = scene(phase, mode, profile)
    engine.take_turn(engine.racers[current])
    certificate_event, = [event for event in engine.events if event.reason == proof.REASON]
    certificate = json.loads(certificate_event.operation)
    assert certificate['proof_domain'] == proof.THIRD_WHEEL12_PROFILE
    assert certificate['queue_length_unbounded'] is True
    assert certificate['control_configurations'] == 8 and certificate['all_legal_control_edges'] == 24
    assert 'finite_configurations' not in certificate
    original, executed, fresh = (certificate[key] for key in ('original_pending_ids', 'executed_pending_ids', 'new_pending_ids'))
    assert executed[:len(original)] == original
    assert set(original).isdisjoint(fresh) and set(executed).isdisjoint(fresh)
    assert certificate['persistent_romantic_jobs'] >= 1
    assert certificate['scores_unchanged'] and certificate['all_actual_copy_choices_retained']
    assert certificate['mechanical_state_returned'] and certificate['all_original_jobs_executed']
    assert all(record.action in record.request.legal_actions for record in engine.decisions)
    assert all(record.request.window == DecisionWindow.COPY_CHOICE for record in engine.decisions)
    period = engine.events[certificate_event.parent_sequence-1:certificate_event.sequence-1]
    assert not any(event.kind in {'roll', 'duel_roll', 'score', 'modifier', 'trip', 'recover', 'effect_cancelled'} for event in period)
    actual_popped = [job.queued_sequence for job in engine.pending_effects.popped]
    assert all(q in actual_popped for q in original) and not set(fresh).intersection(actual_popped)
    assert not engine.pending_effects and engine.termination_reason is None
    assert len(engine.turns) == 1 and engine.turns[0].racer_id == current and not engine._turn_in_progress
    assert dice.used == [4]
    assert any(event.kind == 'move' and event.racer_id == current and (event.before, event.after) == (18,22) for event in engine.events)
    trace = plain(engine.trace())
    assert plain(replay_trace(trace).trace()) == trace


def independent_step(state, action, profile):
    position, power, word = state
    head = word[0]
    births = ('R' if not power else profile) if head == 'R' or position == 14 else ''
    return (position if head == 'R' else 26-position,
            power if action == 'keep' else action == 'copy_r', word[1:]+births)


def test_real_old_Copy_R_instance_survives_later_Alchemist_choice_and_births_R_on12():
    engine, _, _, current, ids = scene(14, 'anchor', 'CR')
    engine.take_turn(engine.racers[current])
    r,c,p,s=ids
    target=None
    old_R_instances=[]
    for event in engine.events:
        if event.kind=='copy_changed' and event.racer_id==c: target=event.target_id
        if event.kind=='ability_effect' and event.racer_id==c and event.ability==Ability.ROMANTIC.value and target==p:
            old_R_instances.append(event.sequence)
    assert old_R_instances, 'a queued real Copy-Romantic instance must execute while Copy borrows Alchemist'
    by_sequence={event.sequence:event for event in engine.events}
    assert any(event.kind=='effect_queued' and event.racer_id==r and by_sequence[event.parent_sequence].racer_id==c
               and by_sequence[event.parent_sequence].after==12 for event in engine.events)


@pytest.mark.parametrize("profile", ("RC", "CR"))
def test_independent_unbounded_word_productions_preserve_R_for_all_legal_choices(profile):
    # Arbitrary tails include long/mixed words, with no finite-queue claim.
    words = ['R', 'RC', 'CR', 'RCRRCC', 'C'*31+'R', 'R'*40+'C'*23]
    for position, power, word, action in itertools.product((12,14),(False,True),words,('keep','copy_r','copy_p')):
        expected = independent_step((position,power,word),action,profile)
        assert proof._renewing_successor((position,power,word),action,profile) == expected
        assert expected[2].count('R') >= word.count('R')
        assert expected[2]


@pytest.mark.parametrize("corruption", ('TW_position','TW_power','anchor_power','no_R','old_job_unserved','missing_C_return_R','wrong_birth_parent','changed_score'))
def test_rejects_outside_domain_unserved_original_or_changed_real_birth(corruption, monkeypatch):
    captures, closures, completed = observe(monkeypatch)
    live, _, _, current, ids = scene(14,'romantic','CR')
    live.take_turn(live.racers[current])
    assert closures
    engine, snapshot, _ = copy.deepcopy(closures[0])
    r,c,p,s=ids
    pending = tuple(engine.pending_effects)
    if corruption == 'TW_position': engine.racers[s].position = 11
    elif corruption == 'TW_power': engine.racers[s].ability = Ability.BABA_YAGA
    elif corruption == 'anchor_power': engine.racers[p].ability = Ability.COACH
    elif corruption == 'no_R':
        engine.pending_effects = QueueHistory(job for job in pending if job.racer_id == c)
    elif corruption == 'old_job_unserved':
        snapshot = replace(snapshot, original_pending=(*snapshot.original_pending,pending[0]))
    elif corruption in {'missing_C_return_R','wrong_birth_parent'}:
        stops={event.sequence:event for event in engine.events if event.kind=='stop'}
        index=next(i for i,event in enumerate(engine.events) if i>=snapshot.event_start and event.kind=='effect_queued'
                   and event.racer_id==r and stops[event.parent_sequence].racer_id==c and stops[event.parent_sequence].after==12)
        event=engine.events[index]
        engine.events[index]=replace(event,kind='effect_cancelled') if corruption=='missing_C_return_R' else replace(event,parent_sequence=event.parent_sequence-1)
    elif corruption == 'changed_score':
        engine.score_ledger.record(c,2,1,source='outside',reason='extra',phase='during_race')
    before=tuple(engine.pending_effects)
    assert not completed(engine,snapshot)
    assert tuple(engine.pending_effects)==before
