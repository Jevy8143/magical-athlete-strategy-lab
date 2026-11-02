"""Recompute the fixed512 learning curve's exploratory paired DEV statistics.

The equal-seat bootstrap and production pooled bootstrap are different finite
resampling procedures. Both are reported, separately from earlier primaries.
"""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import statistics

BASE=Path(__file__).with_name('paired_scores.py')
if hashlib.sha256(BASE.read_bytes()).hexdigest()!='b8e780fa5e897abc5de329e51141a467682dc1e2de87c388ef1e098a219c0d2b':raise ValueError('frozen paired arithmetic changed')
s=importlib.util.spec_from_file_location('frozen_paired_arithmetic',BASE);base=importlib.util.module_from_spec(s);s.loader.exec_module(base)
need=base.require;can=base.canonical
PILOT='deadline-c05-1024-1536-20261102-v1';MASTER=202611023512
PARENT='42e77dbd462c896fb0d04c387bb72be9b2245fb418ffd47f240df3a8ae29ef12'
SOURCE='2ddf3ee89893cc9409f714325706d5de53e7c845d05aef4b74e8fcf813140e28'
PLAN='b8fc4747d89b398529c45bd48b8d3413634bad94ab92c0a26e65ec80c97be41a'
A0='f1fa6a80c0bddb940350ecd6b81e9daae118debe385e391a68837e911fc856f9'
C98='98b3ff25ba53ae00c88ead5d0d75e9dc340941d77a71c6a2b4f8133eafd6d59d'
STATES={'completed','diagnostic','error','quarantined','missing','unstarted'}

def model(value,count):
    need(type(value)is dict and set(value)=={'artifact_sha256','training_games'}and type(value['training_games'])is int
         and value['training_games']==count and type(value['artifact_sha256'])is str and re.fullmatch('[0-9a-f]{64}',value['artifact_sha256']),'actual frozen model binding missing')

def pooled(rows):
    study=PILOT+'-holdout';plan={'master_seed':MASTER,'purpose':'evaluation','namespace':study,'scheme':'sha256-domain-v1'}
    prefix=hashlib.sha256(can(plan)).hexdigest()
    ordered=sorted(rows,key=lambda row:prefix+':'+str(row['pair_index']))
    left=[sum(race[row['focal_seat']]for race in row['left_race_scores'])for row in ordered]
    right=[sum(race[row['focal_seat']]for race in row['right_race_scores'])for row in ordered]
    differences=[r-l for l,r in zip(left,right)]
    identity={'statistic':'paired-game-bootstrap-v1','master_seed':MASTER}
    seed=int(hashlib.sha256(can(identity)).hexdigest(),16)&((1<<63)-1);rng=random.Random(seed)
    replicates=sorted(statistics.fmean(rng.choices(differences,k=len(differences)))for _ in range(2000))
    return {'paired_games':128,'initial_mean_total':statistics.fmean(left),'final_mean_total':statistics.fmean(right),
        'mean_paired_total_difference':statistics.fmean(differences),'paired_bootstrap_95_percent_interval':[replicates[49],replicates[1949]],
        'bootstrap_replicates':2000,'bootstrap_seed':seed}

def summarize(data):
    need(type(data)is dict and set(data)=={'schema_version','purpose','plan_sha256','source_archive_sha256','models','committed_prefix','environment','coverage','pairs','production_pooled_statistics'},'unknown fixed-cohort fields')
    need(type(data['schema_version'])is int and data['schema_version']==1 and data['purpose']=='fixed512_learning_curve_development'
         and data['plan_sha256']==PLAN and data['source_archive_sha256']==SOURCE,'fixed development plan/source changed')
    expected={'pilot_id':PILOT,'master_seed':MASTER,'pairs_per_seat':32,'inference_role':'development_learning_curve_exploratory',
        'opponent_snapshot_ids':[A0]*3,'actual_context_id':C98,'lookup_context_id':C98,'fixed_lookup_deployment':False,
        'training_guard':'public-turn-local-duel-fixed-keep-uniform-retry-v3','evaluation_guard':'public-turn-local-duel-retry-fish-keep-v4','exploration_floor':0.05}
    need(can(data['environment'])==can(expected),'declared native DEV environment differs')
    need(set(data['models'])=={'left','right'},'two planned DEV arms required');model(data['models']['left'],1024)
    need(data['models']['left']['artifact_sha256']==PARENT,'initial1024 reset or replaced')
    prefix=data['committed_prefix'];need(set(prefix)=={'artifact_sha256','training_games','new_updates'}and type(prefix['new_updates'])is int and prefix['new_updates']in range(5),'invalid actual committed prefix')
    model({key:prefix[key]for key in ('artifact_sha256','training_games')},1024+128*prefix['new_updates'])
    if prefix['new_updates']==0:need(prefix['artifact_sha256']==PARENT,'zero-update prefix must remain the actual parent')
    right=data['models']['right']
    if right is not None:
        model(right,1536);need(prefix['new_updates']==4 and prefix['artifact_sha256']==right['artifact_sha256'],'final DEV cannot promote an intermediate checkpoint')
    need(type(data['coverage'])is list and len(data['coverage'])==768,'all512 TRAIN plus256 DEV planned statuses must remain disclosed')
    expected_coordinates=set()
    for batch in range(4):
        for seat in range(4):
            expected_coordinates.update((f'b{batch}',seat,index)for index in range(batch*128+seat*32,batch*128+(seat+1)*32))
    for stage in ('eval-initial','eval-final'):
        for seat in range(4):expected_coordinates.update((stage,seat,index)for index in range(seat*32,(seat+1)*32))
    coverage={}
    for row in data['coverage']:
        need(type(row)is dict and set(row)=={'stage','focal_seat','game_index','status'},'unknown coverage metadata')
        stage,seat,index=row['stage'],row['focal_seat'],row['game_index']
        need(type(seat)is int and type(index)is int and (stage,seat,index)in expected_coordinates and row['status']in STATES,'invalid planned stage/coordinate/status')
        need((stage,seat,index)not in coverage,'duplicate actual coverage coordinate');coverage[stage,seat,index]=row['status']
    need(set(coverage)==expected_coordinates,'some declared attempts disappeared')
    for batch in range(prefix['new_updates']):
        need(all(status=='completed'for (stage,_,_),status in coverage.items()if stage==f'b{batch}'),'a committed128 cannot include incomplete TRAIN')
    if right is not None:
        need(all(status=='completed'for (stage,_,_),status in coverage.items()if stage in ('b0','b1','b2','b3','eval-initial')),'final DEV cannot follow an incomplete strict initial/TRAIN stage')
    need(type(data['pairs'])is list and len(data['pairs'])==128,'exactly128 declared DEV pairs required')
    seen=set()
    for row in data['pairs']:
        need(type(row)is dict and set(row)=={'focal_seat','pair_index','seeds','left_status','right_status','left_race_scores','right_race_scores'},'unknown DEV row fields')
        seat,index=row['focal_seat'],row['pair_index'];need(type(seat)is int and seat in range(4)and type(index)is int and seat*32<=index<(seat+1)*32 and (seat,index)not in seen,'duplicate or invalid DEV pair');seen.add((seat,index))
        need(can(row['seeds'])==can(base.derive_seeds(MASTER,PILOT+'-holdout',index)),'actual paired ten streams differ')
        for arm,stage in (('left','eval-initial'),('right','eval-final')):
            status,scores=row[arm+'_status'],row[arm+'_race_scores'];need(status==coverage[stage,seat,index],'score/coverage status mismatch')
            if status=='completed':
                need(type(scores)is list and len(scores)==4 and all(type(race)is list and len(race)==4 and all(type(v)is int for v in race)for race in scores),'four official signed Race scores per player required')
                need(all(sum(race[player]for race in scores)>=0 for player in range(4)),'negative final balance')
                if arm=='right':need(right is not None,'completed final DEV has no actual model')
            else:need(scores is None,'incomplete arm cannot have an invented final result')
    complete=all(row['left_status']==row['right_status']=='completed'for row in data['pairs'])
    if right is None:need(all(row['right_status']=='unstarted'for row in data['pairs']),'missing final model is allowed only before final DEV starts')
    separate=base.paired_statistics(data['pairs'],32,MASTER,PILOT+'-holdout')if complete else None
    production=pooled(data['pairs'])if complete else None
    original=data['production_pooled_statistics']
    if original is not None:
        need(complete and all(can(original[key])==can(value)for key,value in production.items()),'retained production pooled result differs from its separately recomputed formula')
    return {'schema_version':1,'inference_role':'development_learning_curve_exploratory','models':data['models'],'committed_prefix':prefix,
        'planned_DEV_pairs':128,'DEV_pairs_complete':complete,'all768_attempt_statuses_retained':True,
        'coverage_by_stage':{stage:dict(Counter(status for (kind,_,_),status in coverage.items()if kind==stage))for stage in ('b0','b1','b2','b3','eval-initial','eval-final')},
        'equal_seat_stratified_paired_statistics':separate,'production_sorted_pairing_key_pooled_statistics':production,
        'retained_production_pooled_report_present':original is not None,
        'two_finite_bootstrap_CIs_need_not_match':True,'earlier_primary_studies_replaced_or_pooled':False,
        'scope':'Predeclared fixed512 development learning curve, not a new confirmatory primary or a basis for adaptive extra training. Pointwise95% intervals, no multiplicity adjustment. Incomplete DEV receives no success-only CI.',
        'independent_unit':'paired Game seed and focal seat, not the three opponents','Game_DB_model_or_policy_calls':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();raw=a.input.read_bytes();v=summarize(json.loads(raw));v['input_sha256']=hashlib.sha256(raw).hexdigest();encoded=json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if a.output:a.output.open('x',encoding='utf-8',newline='\n').write(encoded)
    print(encoded,end='')

if __name__=='__main__':main()
