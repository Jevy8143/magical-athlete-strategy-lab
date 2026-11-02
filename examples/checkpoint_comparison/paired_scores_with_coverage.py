"""The actual native diagnostic requires explicit full-cohort coverage.

Keep the original complete384 script unchanged. This reader never imputes
missing scores: it reports a CI only for a fully completed predeclared panel,
and discloses all three panels' complete/diagnostic/unstarted statuses together.
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

BASE = Path(__file__).with_name('paired_scores.py')
BASE_SHA = 'b8e780fa5e897abc5de329e51141a467682dc1e2de87c388ef1e098a219c0d2b'
if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
    raise ValueError('the original complete384 arithmetic source changed')
spec = importlib.util.spec_from_file_location('predeclared_complete_paired_scores', BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require = base.require
STATES = {'completed', 'diagnostic', 'error', 'quarantined', 'unstarted', 'missing'}


def summarize(data):
    require(type(data) is dict and set(data) == {'schema_version','purpose','models','panels','pairs'}, 'unknown coverage dataset fields')
    require(type(data['schema_version']) is int and data['schema_version'] == 1
            and data['purpose'] == 'frozen_eval_score_export_with_full_coverage', 'coverage schema differs')
    require(set(data['models']) == {'left','right'}, 'two frozen models required')
    for arm,count in (('left',896),('right',1024)):
        model = data['models'][arm]
        require(set(model) == {'artifact_sha256','training_games'} and type(model['training_games']) is int
                and model['training_games'] == count and type(model['artifact_sha256']) is str
                and re.fullmatch(r'[0-9a-f]{64}',model['artifact_sha256']), 'model binding differs')
    require(data['models']['left']['artifact_sha256'] == base.PARENT_896, 'wrong frozen896 parent')
    require(set(data['panels']) == set(base.ROLES), 'coverage must retain all three predeclared panels')
    for panel,setup in data['panels'].items():
        require(type(setup) is dict and set(setup) == {'inference_role','pairs_per_seat','master_seed','study_id','opponent_snapshot_ids','actual_context_id','lookup_context_id','mean_guard'}, 'unknown panel fields')
        require(setup['inference_role'] == base.ROLES[panel] and type(setup['pairs_per_seat']) is int and setup['pairs_per_seat'] == 32, 'inference role or size changed')
        require(type(setup['opponent_snapshot_ids']) is list and len(setup['opponent_snapshot_ids']) == 3
                and all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}',value) for value in [*setup['opponent_snapshot_ids'],setup['actual_context_id'],setup['lookup_context_id']]), 'actual opponent/context identity missing')
        require(base.canonical(setup['mean_guard']) == base.canonical({'scheme':'public-turn-local-duel-retry-fish-keep-v4','implementation_sha256':'548f19b411b29924e103344e05310327abd38cbd60d4c363d71c7eb385dbe644'}), 'mean guard differs')
    require(type(data['pairs']) is list and len(data['pairs']) == 384, 'all384 planned pairs must be disclosed')
    seen,streams = set(),set()
    for row in data['pairs']:
        require(type(row) is dict and set(row) == {'panel','focal_seat','pair_index','seeds','left_status','right_status','left_race_scores','right_race_scores'}, 'unknown coverage row fields')
        panel,seat,index = row['panel'],row['focal_seat'],row['pair_index']
        require(panel in base.ROLES and type(seat) is int and seat in range(4) and type(index) is int and seat*32 <= index < (seat+1)*32, 'wrong declared pair coordinate')
        require((panel,seat,index) not in seen, 'duplicate pair')
        seen.add((panel,seat,index))
        setup = data['panels'][panel]
        require(base.canonical(row['seeds']) == base.canonical(base.derive_seeds(setup['master_seed'],setup['study_id'],index)), 'declared ten streams differ')
        seeds = row['seeds'];values = [seeds['game_seed'],seeds['dice_seed'],*seeds['game_policy_seeds'],*seeds['race_policy_seeds']]
        require(len(set(values)) == 10 and not streams.intersection(values), 'seed collision or replacement')
        streams.update(values)
        for arm in ('left','right'):
            status,scores = row[arm+'_status'],row[arm+'_race_scores']
            require(status in STATES, 'unknown actual arm status')
            if status == 'completed':
                require(type(scores) is list and len(scores) == 4 and all(type(race) is list and len(race) == 4 and all(type(score) is int for score in race) for race in scores), 'a completed Game needs all official signed scores')
                require(all(sum(race[player] for race in scores) >= 0 for player in range(4)), 'negative final balance')
            else:
                require(scores is None, 'no invented or partial four-Race score for an incomplete arm')
    results = {}
    for panel,role in base.ROLES.items():
        own = [row for row in data['pairs'] if row['panel'] == panel]
        complete = all(row['left_status'] == row['right_status'] == 'completed' for row in own)
        value = {'inference_role':role,'planned_pairs':128,
                 'complete_pairs':sum(row['left_status'] == row['right_status'] == 'completed' for row in own),
                 'left_attempt_statuses':dict(Counter(row['left_status'] for row in own)),
                 'right_attempt_statuses':dict(Counter(row['right_status'] for row in own)),
                 'all_predeclared_pairs_complete':complete,'statistics':None}
        if complete:
            setup = data['panels'][panel]
            value['statistics'] = base.paired_statistics(own,32,setup['master_seed'],setup['study_id'])
        results[panel] = value
    return {'schema_version':1,'models':data['models'],'panels':results,
            'primary_native_incremental_learning_completed':results['native']['all_predeclared_pairs_complete'],
            'all_three_panels_coverage_disclosed':True,'unconditional_CI_for_incomplete_panels':False,
            'sample_replacement_or_score_imputation':False,'panel_pooling':False,
            'multiplicity':'95% pointwise intervals without multiplicity correction; random/public-memory exploratory.',
            'independent_unit':'One paired Game seed and focal seat; the three opponents are not extra independent observations.',
            'seed_scope':'Declared shared streams, independently checked against every recorded actual arm; an unstarted arm has no trajectory.',
            'scope':'Score-table arithmetic only. The private exporter separately authenticates actual seeds, policies and official ledgers.',
            'Game_database_model_or_policy_calls':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    raw=args.input.read_bytes();result=summarize(json.loads(raw));result['input_sha256']=hashlib.sha256(raw).hexdigest()
    encoded=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8',newline='\n')as stream:stream.write(encoded)
    print(encoded,end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
