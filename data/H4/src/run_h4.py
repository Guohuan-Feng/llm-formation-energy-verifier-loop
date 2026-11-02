"""Finite H4 stage4.1 only. No operation or scientific call occurs on import."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
SRC=Path(__file__).resolve().parent
ROOT=SRC.parents[3]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(SRC))
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import csv
import ctypes
import io
import json
import math
import os
for _thread_variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable]='1'
import statistics
import time
import traceback
import numpy as np
import h4_common as c
import h4_data as data
import h4_prompts as prompts
import h4_models as models
import h4_stats as stats
import h4_transport as transport

PRODUCTION_FILES=['h4_common.py','h4_data.py','h4_prompts.py','h4_models.py',
                  'h4_sandbox.py','h4_stats.py','h4_transport.py','run_h4.py']
STAGE='4.1'

class ScientificStop(RuntimeError): pass

def process_identity():
    import ctypes.wintypes as w
    class FT(ctypes.Structure): _fields_=[('low',w.DWORD),('high',w.DWORD)]
    creation,exit_,kernel,user=FT(),FT(),FT(),FT()
    api=ctypes.windll.kernel32
    api.GetCurrentProcess.restype=w.HANDLE
    api.GetProcessTimes.argtypes=[w.HANDLE,ctypes.POINTER(FT),ctypes.POINTER(FT),ctypes.POINTER(FT),ctypes.POINTER(FT)]
    api.GetProcessTimes.restype=w.BOOL
    if not api.GetProcessTimes(api.GetCurrentProcess(),
                ctypes.byref(creation),ctypes.byref(exit_),ctypes.byref(kernel),ctypes.byref(user)):
        raise OSError('GetProcessTimes failed')
    ticks=(creation.high<<32)|creation.low
    return {'pid':os.getpid(),'parent_pid':os.getppid(),'creation_time_100ns':ticks,
            'creation_utc':__import__('datetime').datetime.fromtimestamp(ticks/1e7-11644473600,
                __import__('datetime').timezone.utc).isoformat(),
            'driver':str(Path(__file__).resolve()),'arguments':sys.argv[1:],
            'executable':sys.executable,'stage':STAGE,'utc':c.utc()}

def prepare():
    for p in (c.OUT,c.DATA,c.PHASE): c.safe(p).mkdir(parents=True,exist_ok=True)
    if (c.DATA/'frozen_config.json').exists():
        return c.verify()
    if (c.OUT/'config.json').exists():
        raise RuntimeError('Unsealed config exists; refusing overwrite')
    if any(c.OUT.iterdir()): raise RuntimeError('Public H4 output must be empty before freeze')
    bundle=data.load_bundle()
    protocol=c.safe(c.DATA/'protocol'); protocol.mkdir(parents=True,exist_ok=True)
    schema=protocol/'features.schema.json'; instructions=protocol/'instructions.txt'
    catalog=protocol/'model_catalog.json'
    c.bytes_write(schema,prompts.schema_bytes(),exclusive=True)
    c.text(instructions,prompts.INSTRUCTIONS,exclusive=True)
    c.bytes_write(catalog,(c.ROOT/'work/h-data/H3/model_catalog.json').read_bytes(),exclusive=True)
    protocol_files={'features_schema_path':str(schema),'instructions_path':str(instructions),'catalog_path':str(catalog)}
    requests=prompts.build_requests(bundle,protocol_files=protocol_files)
    request_files=[]
    for request in requests:
        spec={**request,'schema':prompts.FEATURE_SCHEMA,'instructions':prompts.INSTRUCTIONS,
              'stage':STAGE,'max_format_repairs':1,'admissible_format_failure':False}
        if spec['schema_sha256']!=transport.digest((transport.canonical(spec['schema'])+'\n').encode()):
            raise RuntimeError('Frozen schema bytes differ from actual transport schema')
        prompts.audit_input_bound(spec['prompt']+transport.FORMAT_REPAIR,protocol_files=protocol_files)
        p=c.PHASE/'requests'/(spec['request_id']+'.json')
        c.write(p,spec,exclusive=True)
        request_files.append({'path':str(p),'sha256':c.sha(p),'request_id':spec['request_id']})
    source_hashes={str(SRC/name):c.sha(SRC/name) for name in PRODUCTION_FILES}
    inputs={**bundle['input_hashes'],**{str(p):c.sha(p) for p in (schema,instructions,catalog)}}
    cfg={'protocol':'H4-stage4.1-one-shot-frozen-v1','frozen_utc':c.utc(),'stage':STAGE,
         'runtime':c.RUNTIME,'data':data.configuration(bundle=bundle),
         'prompts':prompts.configuration(requests,bundle=bundle),
         'protocol_files':protocol_files,'prompt_request_files':request_files,
         'models':{'parameters':models.HGB_PARAMETERS,'fit_weights':'unweighted as H3',
                   'CPU_math_threads':1,
                   'feature_failure_threshold':.2,'sandbox_workers':2,'timeout_ms':50,
                   'candidate_count_per_seed':20,'selection':'minimum row validation MAE; exact ties earliest index',
                   'candidate_order':[[s,i] for i in range(20) for s in (0,1,2)],
                   'final_mask':'fullTRAIN2000 only after selecting the candidate; no TEST selection'},
         'statistics':stats.configuration(),
         'operational':{'global_all_H4_CLI_call_cap':300,'all_H4_logical_cap':180,
             'max_concurrent_CLI':4,'rate_limit_wait_seconds':[60,120,240],
             'same_request_rate_limit_retries':3,'max_format_repairs':1,
             'format_repair_suffix':transport.FORMAT_REPAIR,'format_repair_suffix_sha256':transport.digest(transport.FORMAT_REPAIR.encode()),
             'three_final_failed_different_requests':'stop','usage_limit':'stop with reset information',
             'pilot_logical_requests':8,'forecast_threshold_seconds':36000,
             'forecast_formula':'elapsed pilot + remaining logical request mean latency/4 + remaining candidate mean wall + 3*mean candidate wall for winners + 600s terminal/audit allowance',
             'feature_and_CLI_execution_overlap':False,'per_candidate_resume':'whole unfinished candidate only; successful requests/candidates immutable',
             'no_automatic_restart':True,'anti_sleep':'SetThreadExecutionState continuous/system/display required',
             'flush':'every request and every100 structures'},
         'stage_boundary':{'run_only':'4.1','next_stage':'wait for explicit continue',
             'later_stages_prepared':False,'future_query_template_spacegroup':'not present in H2; must freeze separate label-free template metadata before stage4.2',
             'forbidden':['H4A modification','H5','Tc','DFT','CHGNet synthetic baseline','unlisted feature tuning']},
         'source_hashes':source_hashes,'input_hashes':inputs,'old_readonly_hashes':c.snapshot_old()}
    c.write(c.OUT/'config.json',cfg,exclusive=True)
    c.text(c.OUT/'llm_calls.jsonl','',exclusive=True)
    c.text(c.OUT/'run.log','',exclusive=True)
    c.text(c.OUT/'predictions.csv','arm_id,condition,seed,split,id,prediction_eV_per_atom\n',exclusive=True)
    c.update('prepared','frozen_before_first_call',0,LLM_calls_new=0,TEST_targets_read=0,
             candidate_total=60,candidates_complete=0,final_prediction_arms_complete=0)
    c.readme('prepared')
    c.write(c.DATA/'frozen_config.json',{'path':str(c.OUT/'config.json'),'sha256':c.sha(c.OUT/'config.json'),
            'frozen_utc':c.utc(),'stage':STAGE},exclusive=True)
    result=c.verify(cfg)
    c.log('Frozen stage4.1 protocol:60 requests, each300 examples; no TEST label access; no LLM call yet')
    return result

def load_requests(cfg):
    specs=[c.read(item['path']) for item in cfg['prompt_request_files']]
    return sorted(specs,key=lambda r:(r['candidate_index'],r['seed']))

def _request_batch(client,specs):
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(client.request,spec,prompts.validator):spec for spec in specs}
        for future in as_completed(pending):
            spec=pending[future]
            result=future.result()
            results.append(result)
            c.write(c.PHASE/'logical_results'/(spec['request_id']+'.json'),result)
            summary=client.summary()
            n=summary['first_valid_cached']
            c.update('running','feature_generation',min(40,40*n/60),
                LLM_calls_new=summary['cli_attempt_count'],logical_successes=n,
                logical_failures=sum(r.get('status')!='ok' for r in client._results),
                latest_request_id=spec['request_id'],transport=summary)
    return results

def require_success(results,client):
    if client.stop.is_set() or any(item.get('status')!='ok' for item in results):
        raise ScientificStop('Feature generation not complete: '+str(client.summary()['stop_reason'] or
              [(item['request_id'],item['last_status']) for item in results if item.get('status')!='ok']))

def candidate_progress(event):
    # Only sanitized aggregate fields; labels, structures and generated sources stay private/ledger.
    if event.get('phase')=='candidate_complete':
        c.log('Candidate validation complete seed='+str(event['seed'])+' index='+str(event['candidate_index']))
    elif event.get('phase')=='winner_prediction_complete':
        c.log('Winner TEST prediction complete seed='+str(event['seed'])+'; no TEST targets read')
    c.write(c.PHASE/'current_operation.json',{'utc':c.utc(),**event})

def evaluate_specs(bundle,specs,results):
    by_id={r['request_id']:r for r in results}
    complete=[]
    for spec in specs:
        result=by_id[spec['request_id']]
        if result['status']!='ok': raise ScientificStop('Missing candidate response')
        seed=spec['seed']; split=bundle['perseed'][seed]
        frozen_split=c.read(c.OUT/'config.json')['data']['perseed'][str(seed)]
        if any(split[key]!=frozen_split[key] for key in ('S0_ids','O_ids','fit_ids','val_ids')) or set(split['O_ids'])!=set(data.reveal_training(seed,bundle=bundle)):
            raise RuntimeError('Frozen seed split changed')
        metadata=models.evaluate_candidate(bundle,seed,spec['candidate_index'],result['response']['features'],
            data.reveal_training(seed,bundle=bundle),split,
            c.PHASE/'candidates'/spec['request_id'],progress=candidate_progress)
        complete.append(metadata)
        existing=list((c.PHASE/'candidates').glob('*/candidate.json'))
        c.update('running','candidate_validation',40+45*len(existing)/60,
                candidates_complete=len(existing),latest_candidate=spec['request_id'])
    return complete

def forecast(pilot_results,pilot_candidates,elapsed):
    if len(pilot_results)!=8 or len(pilot_candidates)!=8: raise ValueError('Pilot requires eight complete candidates')
    latencies=[r['logical_elapsed_seconds'] for r in pilot_results]
    if any(r.get('cache_hit') for r in pilot_results):
        raise ScientificStop('Pilot forecast missing; do not substitute cache-hit latency for measured calls')
    mean_call=statistics.mean(latencies)
    mean_candidate=statistics.mean(r['elapsed_seconds'] for r in pilot_candidates)
    # Full candidate TRAIN execution+fit scaled by1/2 for1000TEST; conservative full fit allowance retained.
    final_allowance=3*mean_candidate
    remaining_calls=52*mean_call/4
    remaining_candidates=52*mean_candidate
    total=elapsed+remaining_calls+remaining_candidates+final_allowance+600
    return {'stage':STAGE,'utc':c.utc(),'pilot_logical_requests':8,
        'pilot_actual_CLI_attempts':sum(r['cli_attempt_count'] for r in pilot_results),
        'pilot_request_seconds':latencies,'pilot_candidate_seconds':[r['elapsed_seconds'] for r in pilot_candidates],
        'mean_request_seconds':mean_call,'mean_candidate_validation_seconds':mean_candidate,
        'measured_pilot_elapsed_seconds':elapsed,'remaining_generation_seconds':remaining_calls,
        'remaining_candidate_validation_seconds':remaining_candidates,
        'winner_refit_TEST_allowance_seconds':final_allowance,'terminal_and_audit_allowance_seconds':600,
        'estimated_total_seconds':total,'estimated_remaining_seconds':total-elapsed,
        'threshold_seconds':36000,'proceed':total<=36000,
        'method':'measured8 logical requests /4 future slots; measured8 serial candidate walls;3 candidate walls for winners;600s final statistics/audit',
        'caveat':'forecast only; different generated functions and provider latency may change future wall time'}

def prediction_arms(finals):
    arms=[]
    for metadata in finals:
        item=metadata['artifacts']['predictions']
        if c.sha(item['path'])!=item['sha256']: raise RuntimeError('Winner prediction hash changed')
        with np.load(item['path'],allow_pickle=False) as archive:
            ids=archive['ids'].tolist(); values=archive['predictions'].tolist()
        arms.append({'arm_id':'O_seed'+str(metadata['seed']),'condition':'O','seed':metadata['seed'],
              'status':'complete','method':'HGB_Magpie132_LLM_features',
              'prediction_rows':[{'id':rid,'prediction_eV_per_atom':float(value)} for rid,value in zip(ids,values)]})
    return arms

def export_predictions(arms):
    stream=io.StringIO(newline='')
    columns=['arm_id','condition','seed','split','id','prediction_eV_per_atom']
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n'); writer.writeheader()
    for arm in arms:
        for row in arm['prediction_rows']:
            writer.writerow({'arm_id':arm['arm_id'],'condition':'O','seed':arm['seed'],'split':'TEST',**row})
    c.text(c.OUT/'predictions.csv',stream.getvalue())

def run():
    cfg=c.read(c.OUT/'config.json')
    c.verify(cfg)
    with c.RunLock(),c.Awake():
        if (c.PHASE/'evaluation_started.json').exists():
            raise ScientificStop('Stage4.1 terminal evaluation marker already exists; no repeat TEST scoring')
        if (c.PHASE/'run_started.json').exists():
            raise ScientificStop('A previous stage4.1 run exists; manual recovery must explicitly authorize unfinished work')
        started=time.monotonic()
        identity=process_identity()
        c.write(c.DATA/'process_identity.json',identity)
        c.write(c.PHASE/'run_started.json',{'utc':c.utc(),'identity':identity,'config_sha256':c.sha(c.OUT/'config.json')},exclusive=True)
        c.log('Finite stage4.1 run started; anti-sleep active; four CLI slots, serial2worker sandbox')
        client=None
        try:
            bundle=data.load_bundle()
            specs=load_requests(cfg)
            with transport.FiniteTransport(c.RUNTIME,c.DATA/'transport',c.OUT,cfg['protocol_files'],c.log,
                    max_calls=300,max_logical_requests=180,max_workers=4) as client:
                pilot_results=_request_batch(client,specs[:8]); require_success(pilot_results,client)
                pilot_candidates=evaluate_specs(bundle,specs[:8],pilot_results)
                measured=forecast(pilot_results,pilot_candidates,time.monotonic()-started)
                c.write(c.PHASE/'pilot_forecast.json',measured,exclusive=True)
                c.readme('running' if measured['proceed'] else 'stopped_forecast_over_10h',forecast=measured)
                c.log('Pilot8 forecast '+format(measured['estimated_total_seconds']/3600,'.3f')+' hours')
                c.update('running' if measured['proceed'] else 'stopped_forecast_over_10h','pilot_forecast',
                         pilot_forecast=measured)
                if not measured['proceed']:
                    raise ScientificStop('Pilot forecast exceeds10 hours; waiting for user')
                rest_results=_request_batch(client,specs[8:]); require_success(rest_results,client)
                all_results=pilot_results+rest_results
                if len(client.responses)!=60 or len(all_results)!=60: raise RuntimeError('Require60 complete independent candidates')
                c.verify(cfg,full=False)
                candidates=pilot_candidates+evaluate_specs(bundle,specs[8:],rest_results)
                selected=[]
                for seed in (0,1,2):
                    items=[r for r in candidates if r['seed']==seed]
                    if len(items)!=20 or {r['candidate_index'] for r in items}!=set(range(20)):
                        raise RuntimeError('Selection requires exactly20 complete candidates per seed')
                    best=models.select_best(items); selected.append(best)
                c.write(c.PHASE/'selected_candidates.json',{'stage':STAGE,'utc':c.utc(),
                    'selection':'minimum fixed validation rowMAE, earliest tie',
                    'candidate_count':60,'selected':selected,'TEST_targets_read':False},exclusive=True)
                finals=[]
                for best in selected:
                    seed=best['seed']
                    finals.append(models.refit_predict(bundle,seed,best,data.reveal_training(seed,bundle=bundle),
                        bundle['test_ids'],c.PHASE/'final'/('O_seed'+str(seed)),progress=candidate_progress))
                    c.update('running','winner_TEST_predictions',85+3*len(finals),
                             final_prediction_arms_complete=len(finals))
                c.verify(cfg)
                arms=prediction_arms(finals)
                stats._normalise(arms,bundle['test_ids'])
                c.write(c.PHASE/'committed_prediction_arms.json',arms,exclusive=True)
                export_predictions(arms)
                marker=c.PHASE/'evaluation_started.json'
                c.write(marker,{'stage':STAGE,'condition':'O','scoring_pass':1,
                    'TEST_targets_previously_read':False,'utc':c.utc(),
                    'config_sha256':c.sha(c.OUT/'config.json'),
                    'predictions_sha256':c.sha(c.OUT/'predictions.csv'),
                    'selected_candidates_sha256':c.sha(c.PHASE/'selected_candidates.json'),
                    'prediction_arms_sha256':c.sha(c.PHASE/'committed_prediction_arms.json')},exclusive=True)
                c.update('running','terminal_scoring',95)
                terminal=stats.evaluate(arms,bundle,evaluation_marker=marker)
                c.write(c.PHASE/'terminal_results.json',terminal,exclusive=True)
                models._write_npz(c.PHASE/'private_pairing_cache.npz',**stats.private_pairing_cache())
                elapsed=time.monotonic()-started
                c.readme('complete; waiting for continue',forecast=measured)
                result=c.update('complete','stage4_1_complete_waiting_for_continue',100,
                    results=terminal,selected=[{'seed':b['seed'],'candidate_index':b['candidate_index'],
                        'validation_mae':b['validation_mae']} for b in selected],
                    final_models=finals,transport=client.summary(),LLM_calls_new=client.n_calls,
                    elapsed_seconds=elapsed,TEST_targets_read=1000,TEST_scoring_passes=1,
                    candidate_total=60,candidates_complete=60,final_prediction_arms_complete=3,
                    next_stage_started=False)
                audit=verify_terminal(cfg)
                c.write(c.PHASE/'terminal_audit.json',audit,exclusive=True)
                c.write(c.PHASE/'run_stop.json',{'status':'complete','utc':c.utc(),
                    'elapsed_seconds':elapsed,'next_stage_started':False},exclusive=True)
                c.log('Stage4.1 complete; auditPASS; waiting for explicit continue; no later stage started')
                return {'status':'complete','elapsed_seconds':elapsed,'LLM_calls_new':client.n_calls}
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
            status='stopped_forecast_over_10h' if 'exceeds10 hours' in str(exc) else ('stopped_requires_user' if isinstance(exc,ScientificStop) else 'failed')
            c.write(c.PHASE/'run_stop.json',{'status':status,'utc':c.utc(),'error':reason,
                'elapsed_seconds':time.monotonic()-started,
                'transport':client.summary() if client else None})
            c.update(status,'stopped',error=reason,transport=client.summary() if client else None,
                     elapsed_seconds=time.monotonic()-started)
            c.readme(status,forecast=c.read(c.PHASE/'pilot_forecast.json') if (c.PHASE/'pilot_forecast.json').exists() else None)
            c.text(c.PHASE/'traceback.txt',traceback.format_exc())
            c.log('STOP '+reason)
            return {'status':status,'error':reason}
        finally:
            c.write(c.PHASE/'process_finished.json',{'pid':os.getpid(),'utc':c.utc(),
                'elapsed_seconds':time.monotonic()-started})

def verify_terminal(cfg=None):
    cfg=c.read(c.OUT/'config.json') if cfg is None else cfg
    audit=c.verify(cfg)
    m=c.read(c.OUT/'metrics.json')
    if m.get('status')=='complete':
        if m.get('candidates_complete')!=60 or m.get('final_prediction_arms_complete')!=3:
            raise RuntimeError('Terminal model coverage changed')
        if m.get('TEST_scoring_passes')!=1: raise RuntimeError('Terminal score-pass count changed')
        selected=c.read(c.PHASE/'selected_candidates.json')['selected']
        for seed in (0,1,2):
            records=[models._read_sealed(p) for p in (c.PHASE/'candidates').glob('*/candidate.json')
                     if ('O_seed'+str(seed)+'_') in p.parent.name]
            if len(records)!=20 or {r['candidate_index'] for r in records}!=set(range(20)):
                raise RuntimeError('Candidate set changed')
            best=models.select_best(records)
            if best['record_sha256']!=selected[seed]['record_sha256']: raise RuntimeError('Selected winner changed')
            for r in records: models._check_artifacts(r)
        for item in m['final_models']:
            record=models._read_sealed(item['metadata_path'])
            if record['record_sha256']!=item['record_sha256']: raise RuntimeError('Final model metadata changed')
            models._check_artifacts(record)
        with (c.OUT/'predictions.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
        if len(rows)!=3000 or len({(r['seed'],r['id']) for r in rows})!=3000: raise RuntimeError('Prediction coverage changed')
        for seed in ('0','1','2'):
            own=[r for r in rows if r['seed']==seed]
            if {r['id'] for r in own}!=set(cfg['data']['test_ids']) or any(r['split']!='TEST' for r in own):
                raise RuntimeError('Public predictions must be fixed full TEST1000 only')
        marker=c.read(c.PHASE/'evaluation_started.json')
        if marker['predictions_sha256']!=c.sha(c.OUT/'predictions.csv'): raise RuntimeError('Scored prediction bytes changed')
        results=c.read(c.PHASE/'terminal_results.json')
        if results!=m['results']: raise RuntimeError('Terminal result publication changed')
        if results['TEST_target_accessor_calls']!=1: raise RuntimeError('Target accessor count changed')
        audit.update(candidates=60,final_arms=3,predictions=3000,TEST_scoring_passes=1)
    ledger=transport._read_lines(c.OUT/'llm_calls.jsonl')
    reserved=[r for r in ledger if r['record_type']=='reservation']
    results=[r for r in ledger if r['record_type']=='result']
    if len(reserved)>300 or len({r['request_id'] for r in reserved})>60:
        raise RuntimeError('Stage/global call cap violated')
    successes=[r for r in results if r['status']=='ok']
    if len({r['request_id'] for r in successes})!=len(successes): raise RuntimeError('A successful request was replayed')
    for r in successes:
        if transport.digest(transport.canonical(r['validated_response']).encode())!=r['validated_response_sha256']:
            raise RuntimeError('Successful generated feature source hash changed')
    audit.update(status=m['status'],CLI_reservations=len(reserved),CLI_results=len(results),
                 successful_requests=len(successes),no_other_stage_started=True)
    return audit

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','run','verify'])
    action=parser.parse_args().action
    result=prepare() if action=='prepare' else run() if action=='run' else verify_terminal()
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__': main()
