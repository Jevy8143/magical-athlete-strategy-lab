"""Recompute the separate post-repair128-pair native study, using only stdlib."""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE=Path(__file__).with_name('paired_scores.py')
BASE_SHA='b8e780fa5e897abc5de329e51141a467682dc1e2de87c388ef1e098a219c0d2b'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('frozen paired arithmetic changed')
spec=importlib.util.spec_from_file_location('predeclared_paired_arithmetic',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
need=base.require
SOURCE='2ddf3ee89893cc9409f714325706d5de53e7c845d05aef4b74e8fcf813140e28'
CHILD='42e77dbd462c896fb0d04c387bb72be9b2245fb418ffd47f240df3a8ae29ef12'
A0='f1fa6a80c0bddb940350ecd6b81e9daae118debe385e391a68837e911fc856f9'
CONTEXT='98b3ff25ba53ae00c88ead5d0d75e9dc340941d77a71c6a2b4f8133eafd6d59d'
V4={'scheme':'public-turn-local-duel-retry-fish-keep-v4','implementation_sha256':'548f19b411b29924e103344e05310327abd38cbd60d4c363d71c7eb385dbe644'}
STUDY='deadline-896-1024-native-postrepair-v1';MASTER=202611023302
INITIAL_DATA='d609931b608b5d9c75128d022687ace4710228f1af9b45a4204c06b75e277194'
STATES={'completed','diagnostic','error','quarantined','unstarted','missing'}

def summarize(data):
    need(type(data)is dict and set(data)=={'schema_version','purpose','source_archive_sha256','models','study','initial_study','pairs'},'unknown native-only fields')
    need(type(data['schema_version'])is int and data['schema_version']==1 and data['purpose']=='postrepair_native_frozen_eval_with_full_coverage'
         and data['source_archive_sha256']==SOURCE,'native-only study/source differs')
    need(base.canonical(data['models'])==base.canonical({'left':{'artifact_sha256':base.PARENT_896,'training_games':896},
         'right':{'artifact_sha256':CHILD,'training_games':1024}}),'frozen actual models differ')
    expected={'study_id':STUDY,'master_seed':MASTER,'pairs_per_seat':32,'inference_role':'primary_postrepair_native_incremental_learning',
              'opponent_snapshot_ids':[A0]*3,'actual_context_id':CONTEXT,'lookup_context_id':CONTEXT,'mean_guard':V4,'exploration_floor':0.05}
    need(base.canonical(data['study'])==base.canonical(expected),'new128 declaration differs')
    need(base.canonical(data['initial_study'])==base.canonical({'public_data_sha256':INITIAL_DATA,'preserved':True,'pooled_or_replaced':False}),
         'initial failed cohort must remain separately disclosed')
    need(type(data['pairs'])is list and len(data['pairs'])==128,'only this study128 real declared pairs are permitted')
    seen=set();streams=set()
    for row in data['pairs']:
        need(type(row)is dict and set(row)=={'focal_seat','pair_index','seeds','left_status','right_status','left_race_scores','right_race_scores'},'unknown native pair fields')
        seat,index=row['focal_seat'],row['pair_index']
        need(type(seat)is int and seat in range(4)and type(index)is int and seat*32<=index<(seat+1)*32,'wrong pair/seat coordinate')
        need((seat,index)not in seen,'duplicate pair');seen.add((seat,index))
        need(base.canonical(row['seeds'])==base.canonical(base.derive_seeds(MASTER,STUDY,index)),'fresh predeclared ten streams differ')
        seeds=row['seeds'];values=[seeds['game_seed'],seeds['dice_seed'],*seeds['game_policy_seeds'],*seeds['race_policy_seeds']]
        need(len(set(values))==10 and not streams.intersection(values),'cross-pair seed collision');streams.update(values)
        for arm in ('left','right'):
            status,scores=row[arm+'_status'],row[arm+'_race_scores'];need(status in STATES,'unknown actual status')
            if status=='completed':
                need(type(scores)is list and len(scores)==4 and all(type(race)is list and len(race)==4 and all(type(v)is int for v in race)for race in scores),'complete official four-by-four signed scores required')
                need(all(sum(race[player]for race in scores)>=0 for player in range(4)),'negative final balance')
            else:need(scores is None,'no fabricated/partial final score for incomplete arm')
    complete=all(row['left_status']==row['right_status']=='completed'for row in data['pairs'])
    return {'schema_version':1,'models':data['models'],'source_archive_sha256':SOURCE,
        'inference_role':'primary_postrepair_native_incremental_learning','planned_pairs':128,'planned_arm_attempts':256,
        'left_attempt_statuses':dict(Counter(row['left_status']for row in data['pairs'])),
        'right_attempt_statuses':dict(Counter(row['right_status']for row in data['pairs'])),
        'complete_pairs':sum(row['left_status']==row['right_status']=='completed'for row in data['pairs']),
        'all_predeclared_pairs_complete':complete,'statistics':base.paired_statistics(data['pairs'],32,MASTER,STUDY)if complete else None,
        'initial_failed_study_retained_separately':True,'pooling_or_replacement':False,
        'interval_scope':'95%pointwise,2000seat-stratified paired bootstrap replicates; no multiplicity correction; no success-only CI',
        'study_scope':'A new study declared after observing the initial failure, using disjoint new seeds. It does not complete or erase the initial primary study.',
        'policy_scope':'Both frozen mean V4 arms retain exploration_floor0.05 and public history/guard controls; three A0 V3 opponents; fixed nativeC98 lookup.',
        'unit':'One paired Game seed/focal seat, not three independent opponent seats.',
        'authenticity_scope':'Arithmetic only; the private export separately authenticates actual seeds/policies and official signed ledgers.',
        'Game_database_model_or_policy_calls':0}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();raw=a.input.read_bytes();result=summarize(json.loads(raw));result['input_sha256']=hashlib.sha256(raw).hexdigest()
    encoded=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n'
    if a.output:a.output.open('x',encoding='utf-8',newline='\n').write(encoded)
    print(encoded,end='')

if __name__=='__main__':main()
