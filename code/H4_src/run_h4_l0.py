"""Finite stage4.2 only; frozen before requests, no automatic restart."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
SRC=Path(__file__).resolve().parent;ROOT=SRC.parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(SRC))
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import io
import json
import statistics
import time
import traceback
import numpy as np
import h4_l0_common as c
import h4_l0_data as data
import h4_l0_prompts as prompts
import h4_l0_queries as queries
import h4_l0_models as models
import h4_l0_stats as stats
import h4_l0_wire as wire
import h4_transport as transport
import h4_prompts as old_prompts
import h4_models as model_base

FILES=['h4_l0_common.py','h4_l0_data.py','h4_l0_prompts.py','h4_l0_queries.py',
       'h4_l0_models.py','h4_l0_stats.py','h4_l0_wire.py','run_h4_l0.py']
class ScientificStop(RuntimeError):pass

def state_path(seed):return c.PHASE/'states'/('seed'+str(seed)+'.json')

def progress(event):
    c.write(c.PHASE/'current_operation.json',{'utc':c.utc(),**event})
    if event.get('phase') in ('round_complete','winner_prediction_complete'):
        c.log('L0 '+event['phase']+' seed='+str(event['seed'])+' round='+str(event.get('round_index')))

def prepare():
    if (c.PHASE/'frozen_config.json').exists():return verify_terminal()
    started=time.monotonic();preserved=c.preserve_stage4_1()
    metadata=data.prepare_spacegroups();bundle=data.load_bundle()
    initial=[]
    for seed in (0,1,2):
        labels=data.initial_labels(seed,bundle=bundle)
        model=models.initial_state(bundle,seed,labels,c.PHASE/'initial'/('L0_seed'+str(seed)),progress=progress)
        state={'seed':seed,'condition':'L0','round_completed':0,'labels':labels,'carry':0,
            'round_model_paths':[],'query_history':[],'feedback_model_path':model['metadata_path'],
            'final_fill':None,'status':'initialized'}
        if state_path(seed).exists():
            if c.read(state_path(seed))!=state:raise RuntimeError('Pre-freeze initial state differs')
        else:c.write(state_path(seed),state,exclusive=True)
        spec=prompts.build_request(seed,1,model['active_features'],model['validation_mae'],model['worst20'],labels,bundle=bundle)
        request_file=c.PHASE/'initial_requests'/(spec['request_id']+'.json')
        if request_file.exists():
            if c.read(request_file)!=spec:raise RuntimeError('Pre-freeze first request differs')
        else:c.write(request_file,spec,exclusive=True)
        initial.append({'request_id':spec['request_id'],'canonical_prompt_sha256':spec['canonical_prompt_sha256'],
            'input_budget':spec['input_budget'],'format_repair_input_budget':spec['format_repair_input_budget']})
    if prompts.response_schema(bundle,75)!=prompts.response_schema(bundle,1500):
        raise RuntimeError('L0 response schema must be fixed across dynamic quotas')
    original_cfg=c.read(c.ARCHIVE/'config.json')
    cfg={'protocol':'H4-stage4.2-L0-new-freeze-v1','stage':'4.2','condition':'L0','frozen_utc':c.utc(),
        'runtime':c.RUNTIME,'data':data.configuration(bundle=bundle),'prompts':prompts.configuration(bundle=bundle),
        'models':{'parameters':models.HGB_PARAMETERS,'fit_weights':'unweighted as H3 and4.1',
            'CPU_math_threads':1,'sandbox_workers':2,'timeout_ms':50,'failure_threshold':.2,
            'initialization':'Magpie-only S0500; initial feedback not eligible for winner',
            'round_sequence':'request features+query; select/reveal; evaluate new set on current labels; save feedback',
            'winner':'minimum saved row validation MAE among20 rounds1..20; earliest tie; no reranking after fill',
            'next_prompt_current_features':'complete active_features after normal inner-fit failure mask, no input-budget pruning'},
        'statistics':stats.configuration(),'first_round_requests':initial,
        'budget':{'H4_global_CLI_cap':300,'prior_stage4_1_calls':60,'remaining_before_L0':240,
            'logical_round_requests':60,'maximum_format_repairs':1,'rate_limit_wait_seconds':[60,120,240],
            'rate_limit_retries':3,'maximum_CLI_workers':4,'actual_round_dependency_parallelism':3,
            'CLI_and_sandbox_overlap':False,'final_failure_streak_stop':3,
            'invalid_query_after_one_repair':'if complete features remain valid, preserve them and use empty query; carry quota',
            'invalid_features_after_one_repair':'scientific stop; no invented feature set',
            'pilot':'all3 seeds first2 rounds=6 requests; includes first seed first2 rounds',
            'forecast_gate_seconds':36000,'forecast_formula':'elapsed +18*max(mean request per seed) + sum remaining round labelled upper bounds * mean measured validation seconds per labelled row +18*mean measured query/oracle/IO overhead +600s terminal + winner extraTRAIN/TEST allowance',
            'no_automatic_restart':True,'stage4_3_started':False},
        'prior_stage4_1':{'config_path':str(c.ARCHIVE/'config.json'),'config_sha256':c.sha(c.ARCHIVE/'config.json'),
            'marker_sha256':preserved['stage4_1_marker_sha256'],'preservation_manifest':str(c.PHASE/'stage4_1_preservation.json')},
        'protocol_files':original_cfg['protocol_files'],
        'source_hashes':{str(c.SRC/name):c.sha(c.SRC/name) for name in FILES},
        'input_hashes':{**bundle['source_hashes'],**original_cfg['input_hashes'],
            **{str(p):c.sha(p) for p in (c.ROOT/'work/h-data/H4A/src/h4a_queries.py',c.ROOT/'work/h-data/H4A/src/h4a_strict_wire.py')}},
        'prepare_elapsed_seconds':time.monotonic()-started,'POOL_spacegroup_preparation':{
            'path':str(data.SPACEGROUPS),'sha256':c.sha(data.SPACEGROUPS),'parameters':metadata['parameters'],'versions':metadata['versions'],
            'performed_before_driver_prepare':True,'prior_preprocessing_forecast_allowance_seconds':120}}
    if cfg['models']['parameters']!=original_cfg['models']['parameters']:
        raise RuntimeError('HGB parameters changed from4.1')
    c.write(c.PHASE/'config.json',cfg,exclusive=True)
    c.write(c.PHASE/'frozen_config.json',{'sha256':c.sha(c.PHASE/'config.json'),'utc':c.utc()},exclusive=True)
    c.old.bytes_write(c.OUT/'config.json',(c.PHASE/'config.json').read_bytes())
    c.write(c.OUT/'metrics.json',{'stage':'4.2','condition':'L0','status':'prepared','phase':'frozen_before_first_L0_call',
        'updated_utc':c.utc(),'rounds_complete':0,'rounds_total':60,'LLM_calls_stage4_2':0,'LLM_calls_global':60,
        'TEST_targets_read_this_condition':0,'current_L0_scoring_pass':0,'prior_H4_TEST_scoring_passes':1,
        'prior_stage4_1_results':str(c.ARCHIVE/'metrics.json'),'next_stage_started':False})
    c.readme('prepared');c.log('L0 stage4.2 frozen; stage4.1 preserved; independentL0 TEST gate unused')
    return c.verify(cfg)

def resolve_response(result,*,bundle,capacity):
    if result['status']=='ok':
        payload=result['response'];prompts.validate_response(payload,bundle=bundle,capacity=capacity)
        return payload,{'empty_query_fallback':False}
    if result.get('last_status')=='format_invalid' and result.get('admissible_format_failure'):
        payload=json.loads(result.get('raw_response_text') or '',parse_constant=transport._reject_constant)
        if not isinstance(payload,dict) or 'features' not in payload:raise ScientificStop('Invalid feature response after repair')
        old_prompts.validator({'features':payload['features']})
        return {'features':payload['features'],'queries':[],'rationale':''},{'empty_query_fallback':True,
            'last_validation_error':result.get('validation_error'),'original_response_preserved':True}
    raise ScientificStop('L0 request failed: '+str(result.get('request_id'))+' '+str(result.get('last_status')))

def request_round(client,bundle,states,round_index,cfg):
    round_started=time.monotonic();specs={};result={}
    fixed_schema=cfg['prompts']['round1_schema_example']
    for seed in (0,1,2):
        state=states[seed]
        if state['round_completed']!=round_index-1:raise RuntimeError('Round ordering changed')
        feedback=model_base._read_sealed(state['feedback_model_path'])
        spec=prompts.build_request(seed,round_index,feedback['active_features'],feedback['validation_mae'],
            feedback['worst20'],state['labels'],carry=state['carry'],bundle=bundle)
        if spec['schema']!=fixed_schema:raise RuntimeError('Frozen round schema changed')
        if round_index==1:
            expected=c.read(c.PHASE/'initial_requests'/(spec['request_id']+'.json'))
            if spec!=expected:raise RuntimeError('First-round frozen prompt changed')
        c.write(c.PHASE/'requests'/(spec['request_id']+'.json'),spec,exclusive=True)
        specs[seed]=spec
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(client.request,spec,prompts.validator(bundle,spec['capacity'])):seed for seed,spec in specs.items()}
        for future in as_completed(pending):
            seed=pending[future];value=future.result();result[seed]=value
            c.write(c.PHASE/'logical_results'/(specs[seed]['request_id']+'.json'),value,exclusive=True)
    if client.stop.is_set():raise ScientificStop('Transport stop: '+str(client.summary()['stop_reason']))
    requests_wall=time.monotonic()-started
    round_models=[]
    for seed in (0,1,2):
        state=states[seed];spec=specs[seed]
        payload,validation_audit=resolve_response(result[seed],bundle=bundle,capacity=spec['capacity'])
        query=queries.select_only_no_reveal({'queries':payload['queries'],'rationale':payload['rationale']},
            arm='L0',seed=seed,round_index=round_index,labeled_ids=sorted(state['labels']),bundle=bundle,carry=state['carry'])
        selected=query['selected_ids']
        if set(selected)&set(state['labels']):raise RuntimeError('Duplicate label reveal')
        fresh=data.reveal(selected,seed,bundle=bundle) if selected else {}
        labels={**state['labels'],**fresh};data.validate_labels(labels,seed,bundle=bundle)
        record={'stage':'4.2','seed':seed,'round_index':round_index,'request_id':spec['request_id'],
            'canonical_prompt_sha256':spec['canonical_prompt_sha256'],'response_sha256':result[seed].get('response_sha256'),
            'query_payload':{'queries':payload['queries'],'rationale':payload['rationale']},
            'query_selection':query,'query_response_audit':validation_audit,'labels_before':len(state['labels']),
            'labels_after':len(labels),'labelled_ids_sha256':data.json_hash(sorted(labels))}
        c.write(c.PHASE/'query_records'/('L0_seed'+str(seed)+'_round'+str(round_index).zfill(2)+'.json'),record,exclusive=True)
        model=models.evaluate_round(bundle,seed,round_index,payload['features'],labels,
            c.PHASE/'rounds'/('L0_seed'+str(seed))/('round'+str(round_index).zfill(2)),progress=progress)
        state.update(labels=labels,carry=query['carry_out'],round_completed=round_index,status='round_complete',
            feedback_model_path=model['metadata_path'])
        state['round_model_paths'].append(model['metadata_path']);state['query_history'].append(record)
        c.write(state_path(seed),state)
        round_models.append(model)
        n=sum(s['round_completed'] for s in states.values())
        c.update('running','loop_round_validation',rounds_complete=n,round_index=round_index,
            seed=seed,LLM_calls_stage4_2=client.n_calls-60,LLM_calls_global=client.n_calls,
            labels_by_seed={str(k):len(v['labels']) for k,v in states.items()},transport=client.summary())
    total=time.monotonic()-round_started
    return {'round_index':round_index,'requests':[result[s] for s in (0,1,2)],
        'request_wall_seconds':requests_wall,'models':round_models,
        'round_wall_seconds':total,'other_round_overhead_seconds':max(0,total-requests_wall-sum(m['elapsed_seconds'] for m in round_models))}

def forecast(pilot,elapsed,prior_prepare_seconds=0):
    if len(pilot)!=2 or [r['round_index'] for r in pilot]!=[1,2]:raise ValueError('Pilot must be first2 rounds')
    requests=[x for p in pilot for x in p['requests']];models_=[x for p in pilot for x in p['models']]
    if len(requests)!=6 or any(x.get('cache_hit') for x in requests):raise ScientificStop('No cache-hit pilot timing substitution')
    latency=max(statistics.mean([p['requests'][s]['logical_elapsed_seconds'] for p in pilot]) for s in (0,1,2))
    per_row=statistics.mean(x['elapsed_seconds']/x['labelled_count'] for x in models_)
    future_rows=3*sum(500+75*r for r in range(3,21))
    generation=18*latency;validation=future_rows*per_row
    final=3*3000*per_row
    overhead=18*statistics.mean(p['other_round_overhead_seconds'] for p in pilot)
    total=prior_prepare_seconds+elapsed+generation+validation+final+overhead+600
    return {'utc':c.utc(),'pilot_rounds':2,'pilot_requests':6,'actual_CLI_attempts':sum(x['cli_attempt_count'] for x in requests),
        'elapsed_seconds_at_pilot':elapsed,'prior_prepare_seconds':prior_prepare_seconds,
        'mean_request_seconds_by_seed':[statistics.mean([p['requests'][s]['logical_elapsed_seconds'] for p in pilot]) for s in (0,1,2)],
        'measured_validation_seconds_per_labelled_row':per_row,'remaining_generation_seconds':generation,
        'remaining_validation_seconds':validation,'remaining_query_oracle_IO_overhead_seconds':overhead,
        'final_winner_allowance_seconds':final,'terminal_audit_allowance_seconds':600,
        'estimated_total_seconds':total,'estimated_remaining_seconds':total-elapsed-prior_prepare_seconds,'proceed':total<=36000,
        'threshold_seconds':36000,'assumption':'3 dependent seed requests per round, serial2worker sandbox; future labels use full75 quota as conservative upper count',
        'caveat':'different feature costs/provider latency can change future speed; forecast is not measured whole-stage time'}

def prediction_arms(finals,bundle):
    arms=[]
    for item in finals:
        a=item['artifacts']['predictions']
        if c.sha(a['path'])!=a['sha256']:raise RuntimeError('Final prediction hash changed')
        with np.load(a['path'],allow_pickle=False) as f:ids=f['ids'].tolist();values=f['predictions'].tolist()
        if ids!=bundle['test_ids']:raise RuntimeError('Final TEST ID order changed')
        arms.append({'arm_id':'L0_seed'+str(item['seed']),'condition':'L0','seed':item['seed'],'status':'complete',
            'method':'HGB_Magpie132_LLM_loop_features','prediction_rows':[
                {'id':rid,'prediction_eV_per_atom':float(v)} for rid,v in zip(ids,values)]})
    return arms

def export_predictions(arms):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,
        fieldnames=['arm_id','condition','seed','split','id','prediction_eV_per_atom'],lineterminator='\n')
    for arm in arms:
        for row in arm['prediction_rows']:writer.writerow({'arm_id':arm['arm_id'],'condition':'L0','seed':arm['seed'],'split':'TEST',**row})
    c.old.bytes_write(c.OUT/'predictions.csv',(c.ARCHIVE/'predictions.csv').read_bytes()+stream.getvalue().encode())

def curve_and_report(states,terminal,finals,summary,elapsed):
    records=[]
    for seed,state in states.items():
        for p in state['round_model_paths']:
            m=model_base._read_sealed(p);records.append({k:m[k] for k in ('seed','round_index','labelled_count','fit_count','validation_count','validation_mae','validation_group_mae')})
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    c.text(c.PHASE/'report/validation_curve.csv',stream.getvalue())
    configdir=c.safe(c.PHASE/'matplotlib');configdir.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(configdir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    for seed in (0,1,2):
        own=[r for r in records if r['seed']==seed]
        ax.plot([r['round_index'] for r in own],[r['validation_mae'] for r in own],marker='.',label='seed '+str(seed))
    ax.set(xlabel='Round',ylabel='Validation row MAE (eV/atom)',xticks=list(range(1,21)),title='L0 validation MAE; memberships grow with revealed labels')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    image=c.safe(c.PHASE/'report/L0_validation_MAE.png');image.parent.mkdir(parents=True,exist_ok=True);fig.savefig(image,dpi=160);plt.close(fig)
    report=['# H4 stage4.2 L0 results','Status: complete. Stage4.3 is not started.',
        'TEST1000;333 template groups;2000 shared group bootstrap draws. Means are arithmetic means of seed metrics.',
        '|Weight|Seed|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|','|---|---|---|---|---|']
    for mode in ('row_weighted','group_weighted'):
        for seed in ('0','1','2','mean'):
            scores=terminal['arithmetic_seed_mean'] if seed=='mean' else terminal['per_seed'][seed]['scores']
            cells=[f"{scores[mode][metric]['point']:.6f} [{scores[mode][metric]['ci95'][0]:.6f}, {scores[mode][metric]['ci95'][1]:.6f}]" for metric in ('mae','rmse','r2')]
            report.append('|'+mode+'|'+seed+'|'+'|'.join(cells)+'|')
    report+=['','|Seed|Query labels|Final uniform fill|Final labels|Winner round|Kept features|', '|---|---|---|---|---|---|']
    for item in finals:
        state=states[item['seed']];fill=state['final_fill']['selected_n']
        report.append('|'+str(item['seed'])+'|'+str(1500-fill)+'|'+str(fill)+'|2000|'+str(item['selected_round_index'])+'|'+str(len(item['full_train_features']['kept_names']))+'|')
    report+=['',f'Run elapsed seconds: {elapsed:.3f}. Global CLI calls: {summary["cli_attempt_count"]}; L0 CLI calls: {summary["cli_attempt_count"]-60}.',
        'All raw generated sets, query match/reveal/carry records, failure diagnostics, token ledgers and frozen hashes are retained privately.',
        'Validation labels are allowed in feedback; validation memberships grow across rounds. Current code is the complete scientifically retained feature set, never budget truncation.',
        'O was scored previously once; L0 is independently scored once. The original O marker, predictions and metrics are preserved; no O rescoring occurred.',
        'This stage reports L0 only; paired arm and memory conclusions are reserved for later authorized stages.']
    c.text(c.PHASE/'report/H4_stage4_2_report.md','\n'.join(report)+'\n')
    return {'validation_curve':str(image),'curve_csv':str(c.PHASE/'report/validation_curve.csv'),'report':str(c.PHASE/'report/H4_stage4_2_report.md')}

def run():
    cfg=c.read(c.PHASE/'config.json');c.verify(cfg)
    with c.RunLock(),c.Awake():
        if (c.PHASE/'run_started.json').exists() or (c.PHASE/'evaluation_started.json').exists():
            raise ScientificStop('Prior L0 run or evaluation gate exists; manual recovery required; no automatic replay')
        started=time.monotonic();client=None
        c.write(c.PHASE/'process_identity.json',c.process_identity())
        c.write(c.PHASE/'run_started.json',{'utc':c.utc(),'config_sha256':c.sha(c.PHASE/'config.json')},exclusive=True)
        try:
            bundle=data.load_bundle();states={s:c.read(state_path(s)) for s in (0,1,2)}
            for seed in (0,1,2):
                if data.initial_labels(seed,bundle=bundle)!=states[seed]['labels']:raise RuntimeError('Frozen S0 labels changed')
            pilot=[]
            with wire.StrictSchemaTransport(c.RUNTIME,c.DATA/'transport',c.OUT,cfg['protocol_files'],c.log,
                    max_calls=300,max_logical_requests=180,max_workers=4) as client:
                if client.n_calls!=60:raise RuntimeError('Global ledger must begin L0 with exactly60 old calls')
                for rnd in range(1,21):
                    current=request_round(client,bundle,states,rnd,cfg)
                    if rnd<=2:pilot.append(current)
                    if rnd==2:
                        measured=forecast(pilot,time.monotonic()-started,cfg.get('prepare_elapsed_seconds',0)+120)
                        c.write(c.PHASE/'pilot_forecast.json',measured,exclusive=True)
                        c.readme('running' if measured['proceed'] else 'stopped_forecast_over_10h',measured)
                        c.update('running','pilot_forecast',pilot_forecast=measured)
                        if not measured['proceed']:raise ScientificStop('Pilot forecast exceeds10 hours; waiting for user')
                    c.verify(cfg,full=False)
                selected=[];finals=[]
                for seed in (0,1,2):
                    state=states[seed];rounds=[model_base._read_sealed(p) for p in state['round_model_paths']]
                    best=models.select_best(rounds);selected.append(best)
                c.write(c.PHASE/'selected_rounds.json',selected,exclusive=True)
                c.write(c.PHASE/'selection_pre_fill.json',{'utc':c.utc(),'selected_rounds_sha256':c.sha(c.PHASE/'selected_rounds.json'),
                    'states_before_fill':{str(s):c.sha(state_path(s)) for s in (0,1,2)},'fill_labels_not_yet_revealed':True},exclusive=True)
                for seed in (0,1,2):
                    state=states[seed]
                    fill=queries.final_fill_only_no_reveal(arm='L0',seed=seed,labeled_ids=sorted(state['labels']),
                        bundle=bundle,total_new_revealed=len(state['labels'])-500)
                    state['labels'].update(data.reveal(fill['selected_ids'],seed,bundle=bundle) if fill['selected_ids'] else {})
                    data.validate_labels(state['labels'],seed,bundle=bundle,require_n=2000)
                    state.update(final_fill=fill,status='final_labels_ready');c.write(state_path(seed),state)
                for seed,best in enumerate(selected):
                    finals.append(models.finalize(bundle,seed,best,states[seed]['labels'],bundle['test_ids'],
                        c.PHASE/'final'/('L0_seed'+str(seed)),progress=progress))
                    c.update('running','winner_TEST_predictions',final_prediction_arms_complete=len(finals))
                arms=prediction_arms(finals,bundle);stats._normalise(arms,bundle['test_ids'])
                c.write(c.PHASE/'committed_prediction_arms.json',arms,exclusive=True);export_predictions(arms)
                c.verify(cfg)
                marker=c.PHASE/'evaluation_started.json'
                c.write(marker,{'stage':'4.2','condition':'L0','scoring_pass':1,'current_L0_scoring_pass':1,
                    'prior_H4_TEST_scoring_passes':1,'TEST_targets_previously_read_for_this_condition':False,
                    'prior_stage4_1_marker_sha256':c.sha(c.DATA/'phase4_1/evaluation_started.json'),
                    'utc':c.utc(),'config_sha256':c.sha(c.PHASE/'config.json'),
                    'committed_prediction_arms_sha256':c.sha(c.PHASE/'committed_prediction_arms.json'),
                    'public_predictions_sha256':c.sha(c.OUT/'predictions.csv')},exclusive=True)
                c.update('running','terminal_scoring')
                terminal=stats.evaluate(arms,bundle,evaluation_marker=marker)
                c.write(c.PHASE/'terminal_results.json',terminal,exclusive=True)
                model_base._write_npz(c.PHASE/'private_pairing_cache.npz',**stats.private_pairing_cache())
                elapsed=time.monotonic()-started;summary=client.summary()
                report=curve_and_report(states,terminal,finals,summary,elapsed)
                c.readme('complete',c.read(c.PHASE/'pilot_forecast.json'))
                c.update('complete','stage4_2_complete_waiting_for_continue',results=terminal,final_models=finals,
                    rounds_complete=60,rounds_total=60,query_summary={str(s):{'final_labels':len(state['labels']),
                        'revealed_by_queries':1500-state['final_fill']['selected_n'],'uniform_fill':state['final_fill']['selected_n']}
                        for s,state in states.items()},reports=report,elapsed_seconds=time.monotonic()-started,
                    transport=summary,LLM_calls_global=client.n_calls,LLM_calls_stage4_2=client.n_calls-60,
                    TEST_targets_read_this_condition=1000,current_L0_scoring_pass=1,total_H4_TEST_scoring_passes=2,
                    next_stage_started=False)
                audit=verify_terminal();c.write(c.PHASE/'terminal_audit.json',audit,exclusive=True)
                c.write(c.PHASE/'run_stop.json',{'status':'complete','utc':c.utc(),'next_stage_started':False},exclusive=True)
                c.log('L0 stage4.2 complete; preservedO; waiting for explicit continue; no4.3')
                return {'status':'complete','elapsed_seconds':time.monotonic()-started}
        except BaseException as exc:
            reason=type(exc).__name__+': '+str(exc)
            status='stopped_forecast_over_10h' if 'exceeds10 hours' in str(exc) else 'stopped_requires_user' if isinstance(exc,(ScientificStop,prompts.PromptBudgetExceeded)) else 'failed'
            c.write(c.PHASE/'run_stop.json',{'status':status,'utc':c.utc(),'error':reason,
                'transport':client.summary() if client else None,'elapsed_seconds':time.monotonic()-started})
            c.update(status,'stopped',error=reason,transport=client.summary() if client else None)
            c.readme(status,c.read(c.PHASE/'pilot_forecast.json') if (c.PHASE/'pilot_forecast.json').exists() else None)
            c.text(c.PHASE/'traceback.txt',traceback.format_exc());c.log('STOP L0 '+reason)
            return {'status':status,'error':reason}
        finally:c.write(c.PHASE/'process_finished.json',{'pid':os.getpid(),'utc':c.utc(),'elapsed_seconds':time.monotonic()-started})

def verify_terminal():
    cfg=c.read(c.PHASE/'config.json');audit=c.verify(cfg);m=c.read(c.OUT/'metrics.json')
    ledger=transport._read_lines(c.OUT/'llm_calls.jsonl');reserve=[r for r in ledger if r['record_type']=='reservation'];results=[r for r in ledger if r['record_type']=='result']
    if len(reserve)>300:raise RuntimeError('Global300 cap violated')
    success=[r for r in results if r['status']=='ok']
    if len({r['request_id'] for r in success})!=len(success):raise RuntimeError('Successful request replayed')
    for r in success:
        if transport.digest(transport.canonical(r['validated_response']).encode())!=r['validated_response_sha256']:raise RuntimeError('Response hash changed')
    if m['status']=='complete':
        selected=c.read(c.PHASE/'selected_rounds.json')
        for seed in (0,1,2):
            state=c.read(state_path(seed))
            if state['round_completed']!=20 or len(state['labels'])!=2000:raise RuntimeError('Final label budget incomplete')
            mapping=cfg['data']['perseed'][str(seed)]['fit_group_mapping']
            rounds=[model_base._read_sealed(p) for p in state['round_model_paths']]
            winner=models.select_best(rounds)
            if winner['record_sha256']!=selected[seed]['record_sha256']:raise RuntimeError('Winner changed after fill')
            expected=c.read(c.ARCHIVE/'config.json')['data']['perseed'][str(seed)]['S0_ids']
            if not set(expected)<=set(state['labels']):raise RuntimeError('S0 member lost')
            if sum(q['query_selection']['selected_n'] for q in state['query_history'])+state['final_fill']['selected_n']!=1500:raise RuntimeError('Reveal ledger does not total1500')
            seen=set(expected);carry=0;pool=set(cfg['data']['pool_ids'])
            if len(state['query_history'])!=20:raise RuntimeError('Query history must have20 rounds')
            for index,record in enumerate(state['query_history'],1):
                selection=record['query_selection'];chosen=selection['selected_ids']
                if record['round_index']!=index or selection['carry_in']!=carry or selection['capacity']!=75+carry:
                    raise RuntimeError('Query quota/carry changed')
                if len(chosen)!=len(set(chosen)) or set(chosen)&seen or not set(chosen)<=pool:
                    raise RuntimeError('Revealed labels contain duplicates/nonPOOL members')
                if selection['selected_n']!=len(chosen) or selection['carry_out']!=75+carry-len(chosen):
                    raise RuntimeError('Query selection/cardinality mismatch')
                if sum(q['n'] for q in record['query_payload']['queries'])>75+carry:
                    raise RuntimeError('Requested query labels exceed round allowance')
                seen.update(chosen);carry=selection['carry_out']
                if set(rounds[index-1]['identity']['train_ids'])!=seen:
                    raise RuntimeError('Round training members differ from query reveals')
            fill=state['final_fill']['selected_ids']
            if len(fill)!=len(set(fill)) or set(fill)&seen or not set(fill)<=pool:
                raise RuntimeError('Final fill identity mismatch')
            if seen|set(fill)!=set(state['labels']):raise RuntimeError('Final labels differ from frozen reveal/fill ledger')
            for record in rounds:
                model_base._check_artifacts(record)
                if record['identity']['fixed_template_assignment_sha256']!=model_base._json_hash(mapping):raise RuntimeError('Split binding changed')
        for record in m['final_models']:
            sealed=model_base._read_sealed(record['metadata_path'])
            if sealed['record_sha256']!=record['record_sha256']:raise RuntimeError('Final metadata changed')
            model_base._check_artifacts(sealed)
        terminal=c.read(c.PHASE/'terminal_results.json')
        if terminal!=m['results']:raise RuntimeError('Published terminal result changed')
        marker=c.read(c.PHASE/'evaluation_started.json')
        stats._marker(c.PHASE/'evaluation_started.json')
        if terminal['evaluation_marker']['sha256']!=c.sha(c.PHASE/'evaluation_started.json'):
            raise RuntimeError('Terminal marker changed after scoring')
        if marker['committed_prediction_arms_sha256']!=c.sha(c.PHASE/'committed_prediction_arms.json'):
            raise RuntimeError('Scored private predictions changed')
        if terminal['TEST_target_accessor_calls']!=1 or terminal['current_L0_scoring_pass']!=1 or terminal['prior_H4_TEST_scoring_passes']!=1:
            raise RuntimeError('TEST access-pass disclosure changed')
        if marker['public_predictions_sha256']!=c.sha(c.OUT/'predictions.csv'):raise RuntimeError('Scored prediction bytes changed')
        arms=c.read(c.PHASE/'committed_prediction_arms.json');stats._normalise(arms,cfg['data']['test_ids'])
        with (c.OUT/'predictions.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
        own=[r for r in rows if r['condition']=='L0']
        if len(rows)!=6000 or len(own)!=3000 or len({(r['seed'],r['id']) for r in own})!=3000:raise RuntimeError('Prediction coverage changed')
        expected={(arm['arm_id'],str(arm['seed']),row['id']):float(row['prediction_eV_per_atom']).hex()
            for arm in arms for row in arm['prediction_rows']}
        actual={(row['arm_id'],row['seed'],row['id']):float(row['prediction_eV_per_atom']).hex() for row in own}
        if actual!=expected or any(row['split']!='TEST' for row in own):
            raise RuntimeError('Public predictions differ from the exact scored L0 predictions')
        if m.get('current_L0_scoring_pass')!=1 or m.get('next_stage_started') is not False:raise RuntimeError('Evaluation/stage boundary changed')
        for artifact in m['reports'].values():
            if not Path(artifact).is_file():raise RuntimeError('Missing report artifact')
        audit.update(rounds=60,seeds=3,TEST_predictions=3000,L0_scoring_passes=1,prior_O_scoring_passes=1)
    audit.update(status=m['status'],CLI_calls_global=len(reserve),CLI_results=len(results),no4_3=True)
    return audit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify']);action=parser.parse_args().action
    if action=='prepare':
        with c.Awake():result=prepare()
    else:result=run() if action=='run' else verify_terminal()
    if sys.stdout is not None:print(json.dumps(result,ensure_ascii=False,allow_nan=False))
if __name__=='__main__':main()
