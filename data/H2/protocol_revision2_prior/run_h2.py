"""Finite H2 calibration, budget gate, retained datasets and once-only RFs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import json
import math
import multiprocessing as mp
import time
import traceback
import h2_common as c

FORMAL_SOURCES=['run_h2.py','h2_admission.py','h2_forecast.py','h2_evaluation.py','p1_baseline.py','launch_h2_formal.ps1']

class BudgetStop(Exception):pass

def freeze_formal():
    c.setup();cfg=c.read(c.OUT/'config.json');c.verify_config(cfg)
    if not c.read(c.OUT/'metrics.json').get('benchmark_complete'):raise RuntimeError('Complete benchmark before formal freeze')
    cfg['formal_source_hashes']=c.source_hashes(FORMAL_SOURCES)
    cfg['bootstrap_seed']=20261023
    cfg['frozen_inputs_sha256']=cfg['frozen_plan_sha256']
    cfg['known_compositions_sha256']=cfg['generation']['matbench_composition_inventory_sha256']
    cfg['source_hashes']={**cfg['benchmark_source_hashes'],**cfg['formal_source_hashes']}
    repeat=c.read(c.OUT/'metrics.json').get('serial_repeat')
    if repeat:
        repeat=dict(repeat)
        repeat['n_abs_difference_above_1meV']=repeat['n_energy_differences_above_1_meV']
        repeat['n_abs_difference_above_5meV']=repeat['n_energy_differences_above_5_meV']
        cfg['serial_repeat']=repeat
    cfg['input_paths']={'frozen_plan':str(c.DATA/'frozen_plan.json.gz'),'p1_pilot':cfg['generation']['p1_pilot_path']}
    cfg['S0_selection_method']='NumPy default_rng(seed).choice on ID-sorted final retained POOL, seeds0/1/2, 500 each; split_manifest is authoritative'
    cfg['real_label_rule']='All frozen4000 real rows attempted. Finite strictly converged CHGNet labels usable regardless of synthetic geometry/energy eligibility flags, which remain reported. No silent replacement/imputation.'
    cfg['gpu_admission_policy']='Selected worker count is the maximum; initial graph angles>30000 or sites>160 admit1, angles>12000 or sites>80 admit at most2, otherwise selected maximum. All candidates still attempted; this is scheduling, not a material filter. CUDA allocator capped at70percent physical GPU memory divided by actual concurrent workers.'
    cfg['resource_failure_policy']='Recoverable CUDA OutOfMemoryError is an explicit resource_oom unlabeled attempt, never a physical rejection or energy. No retry, replacement or formula change; continue frozen candidate order to quotas/reserve/budget. Real IDs are never replaced and missing labels are disclosed. Other CUDA context errors stop the phase.'
    c.write(c.OUT/'config.json',cfg)

def verify(cfg):
    c.verify_config(cfg)
    assert cfg['formal_source_hashes']==c.source_hashes(FORMAL_SOURCES),'Formal source changed after freeze'

def batch(candidates,workers,budget,phase,on_result=None):
    """A bounded100-row unit, with hard budget cancellation and saved partials."""
    if not candidates:return []
    capacity=min(workers,len(candidates))
    workers=max(w for w in (1,2,4) if w<=capacity)
    from h2_relax import init_worker,relax_candidate
    import os
    os.environ['H2_GPU_WORKERS']=str(workers)
    lookup={x['id']:x for x in candidates};results={}
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker)
    try:
        futures={executor.submit(relax_candidate,x):x['id'] for x in candidates}
        timeout=max(1.,budget.remaining()-60)
        for f in as_completed(futures,timeout=timeout):
            result=f.result();rid=result['row']['id'];result['initial_structure']=lookup[rid]['initial_structure'];results[rid]=result
            result['row']['actual_gpu_workers']=workers
            if result['row']['error']:c.log(f'ERROR {phase} {rid}: {result["row"]["error"]}')
            if on_result:on_result(result)
        executor.shutdown(wait=True)
        return [results[x['id']] for x in candidates]
    except BaseException as exc:
        c.gz_write(c.DATA/'interrupted_formal_batch.json.gz',{'phase':phase,'results':list(results.values()),'saved_utc':c.now()})
        for process in list((executor._processes or {}).values()):
            if process.is_alive():process.terminate()
        executor.shutdown(wait=True,cancel_futures=True)
        if isinstance(exc,TimeoutError):raise BudgetStop('Remaining24h budget reached; completed partial batch saved') from exc
        raise

def admitted_batch(candidates,selected_workers,budget,phase,on_result=None):
    from h2_admission import group_candidates
    results={}
    try:
        for actual_workers,subset in group_candidates(candidates,selected_workers):
            for record in batch(subset,actual_workers,budget,phase,on_result=on_result):
                record['row']['planning_worker_seconds']=record['row']['relax_seconds']*selected_workers/record['row']['actual_gpu_workers']
                results[record['row']['id']]=record
    except BaseException:
        interrupted=c.DATA/'interrupted_formal_batch.json.gz'
        if interrupted.exists():
            for record in c.gz_read(interrupted).get('results',[]):results[record['row']['id']]=record
        c.gz_write(c.DATA/'interrupted_admitted_batch.json.gz',{'phase':phase,'results':list(results.values()),'saved_utc':c.now()})
        raise
    return [results[x['id']] for x in candidates]

def benchmark_summaries():return c.read(c.OUT/'metrics.json')['benchmark']

def stop_readme(reason,metrics):
    lines=['# H2 已保存进度并停止',reason,'未启动 H3，未作尚未完成的 RF 外部评测。',
           '## GPU并行基准','|worker|个/小时（含启动）|峰值显存MiB|最大能量差eV/atom|','|---:|---:|---:|---:|']
    counts=metrics.get('retained_counts',{})
    lines.insert(3,f"正式保留TEST {counts.get('TEST',0)}/1000、POOL {counts.get('POOL',0)}/5000；真实已尝试 {metrics.get('n_real_completed',0)}/4000。")
    for s in metrics.get('benchmark',[]):
        lines.append(f"|{s['workers']}|{s['throughput_per_hour']:.2f}|{s['peak_device_memory_used_MiB']}|{s['energy_max_abs_difference_eV_per_atom']}|")
    forecast=metrics.get('forecast',{})
    if forecast:
        upper=forecast.get('remaining_total_seconds_upper')
        if forecast.get('status')=='synthetic_only_over_budget':
            lines.append(f"仅合成部分点估计 {forecast['remaining_synthetic_seconds_point']/3600:.2f} h，已超过剩余预算 {forecast['remaining_seconds']/3600:.2f} h；未计真实4000与RF。")
        elif upper is not None:lines.append(f"预测剩余耗时上界 {upper/3600:.2f} h；预测时剩余预算 {forecast.get('remaining_seconds',0)/3600:.2f} h。")
        else:lines.append('可靠耗时预测不可用：'+'；'.join(forecast.get('blocking_reasons',[])))
    lines+=['输入、结构、标签及部分结果保存在 work/h-data/H2，POOL标签默认隐藏。',
            '收敛要求原始原子力和原始晶胞过滤器力均≤0.1 eV/Å；最多300步。']
    if metrics.get('resource_revision'):
        lines.append('首轮硬件基准显存故障与同批重测已留档，失败耗时计入24h；本轮按用户要求仅重新生成正式合成候选，真实4000与40个基准不变。')
    if metrics.get('protocol_revision'):
        lines.append('工程一致性门槛由1改为≤5 meV/atom，约原P1 CHGNet-vs-DFT误差58meV的1/10，全部收敛/物理标志仍须相同；不是精度保证。')
        p=metrics.get('prefilter_counts',{})
        lines.append('标签前排除：惰性目标替换选项='+str(p.get('excluded_target_noble_gas'))+'个，预缩放V/N<4 Å³='+str(p.get('initial_volume_per_atom_below_4_A3'))+'个提案，3 Å角对>20000='+str(p.get('initial_angle_pairs_3A_above_20000'))+'个提案；几何独立计数，交/并集见metrics.json。')
    if metrics.get('serial_repeat'):
        r=metrics['serial_repeat'];lines.append(f"同40串行复测仅作诊断：最大/均值能差 {1000*r['energy_max_abs_difference_eV_per_atom']:.6f}/{1000*r['energy_mean_abs_difference_eV_per_atom']:.6f} meV/atom，不设新门槛；无法据单次重复确定唯一因果。")
    assert len(lines)<=20;c.safe(c.OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def precalibration_budget_gate(cfg,metrics):
    """Avoid calibration work when synthetic work alone already exceeds budget."""
    selected=next(s for s in metrics['benchmark'] if s['workers']==cfg['selected_workers'])
    serial=next(s for s in metrics['benchmark'] if s['workers']==1)
    success=serial['n_accepted_physics']/serial['n']
    if success<=0:raise RuntimeError('Benchmark has no successful synthetic labels')
    targets=cfg['generation']['target_retained_sizes']
    attempts={split:math.ceil(target/success) for split,target in targets.items()}
    synthetic_point=sum(attempts.values())*3600/selected['throughput_per_hour']
    ledger=c.read(c.DATA/'budget.json')
    if ledger.get('active_phase'):raise RuntimeError('Close previous active phase before pre-calibration budget gate')
    remaining=c.BUDGET_SECONDS-ledger['spent_seconds']
    if synthetic_point<=remaining:return False
    forecast={'status':'synthetic_only_over_budget','fits_budget':False,
        'method':'Pre-calibration gate: selected consistent benchmark end-to-end throughput and serial40 strict-physics point yield; no real calibration needed when synthetic-only point estimate already exceeds budget.',
        'selected_workers':cfg['selected_workers'],'remaining_seconds':remaining,
        'remaining_synthetic_seconds_point':synthetic_point,'remaining_total_seconds_upper':None,
        'synthetic_yield_point':success,'benchmark_accepted':serial['n_accepted_physics'],'benchmark_n':serial['n'],
        'target_retained_sizes':targets,'synthetic_candidates_needed_point':attempts,
        'real_cost_included':False,'evaluation_cost_included':False,
        'caveat':'Planning point estimate, not a deterministic runtime lower bound; no timing or labels for unrun real candidates are fabricated.'}
    c.write(c.DATA/'forecast.json',forecast)
    reason=f"Synthetic-only point estimate {synthetic_point/3600:.2f}h exceeds remaining total24h budget {remaining/3600:.2f}h; real4000 and RF cost not included. Saved before real calibration/formal labeling."
    updated=c.metrics_update(status='stopped_budget_forecast',phase='precalibration_budget_gate',
        forecast=forecast,calibration_complete=False,formal_started=False,stop_reason=reason,
        n_real_completed=0,retained_counts={'TEST':0,'POOL':0},active_spent_seconds=ledger['spent_seconds'],
        old_artifacts_unchanged=c.snapshot_old()==cfg['old_artifact_hashes'])
    stop_readme(reason,updated);c.log(reason)
    return True

def calibrate():
    c.setup();cfg=c.read(c.OUT/'config.json');verify(cfg)
    m=c.read(c.OUT/'metrics.json')
    if (c.DATA/'calibration_results.json.gz').exists():raise RuntimeError('Completed calibration exists; no silent phase rerun')
    if precalibration_budget_gate(cfg,m):return
    budget=c.Budget('real_calibration');status='failed'
    try:
        with c.Awake():
            from h2_forecast import build_forecast
            plan=c.gz_read(c.DATA/'frozen_plan.json.gz');ids=set(plan['real_calibration_ids'])
            cases=[x for x in plan['real_candidates'] if x['id'] in ids]
            assert len(cases)==len(ids)==60
            c.metrics_update(status='running',phase='real_calibration',calibration_requested=len(cases))
            done=0
            def calibration_progress(result):
                nonlocal done
                done+=1
                if done%15==0:
                    budget.heartbeat();c.metrics_update(calibration_completed=done)
                    c.log(f'H2 real-size calibration {done}/60 ({done/60:.0%}); labels count toward frozen4000')
            t=time.perf_counter();results=admitted_batch(cases,cfg['selected_workers'],budget,'real_calibration',on_result=calibration_progress);wall=time.perf_counter()-t
            c.gz_write(c.DATA/'calibration_results.json.gz',results)
            c.write(c.DATA/'calibration_timing.json',{'wall_seconds':wall,'n':len(cases),'workers':cfg['selected_workers']})
            c.metrics_update(calibration_completed=len(results),n_real_completed=len(results))
            raw_benchmark={str(w):c.gz_read(c.DATA/f'benchmark_results_workers{w}.json.gz') for w in (1,2,4)}
            forecast=build_forecast(plan,cfg['generation'],results,m['benchmark'],cfg['selected_workers'],budget.remaining(),calibration_wall_seconds=wall,benchmark_results=raw_benchmark)
            c.write(c.DATA/'forecast.json',forecast)
            if not forecast['fits_budget']:
                blocked=bool(forecast.get('blocking_reasons'))
                status='stopped_forecast_unavailable' if blocked else 'stopped_budget_forecast'
                reason=('Reliable forecast unavailable: '+'; '.join(forecast['blocking_reasons'])) if blocked else 'Predicted remaining H2 work exceeds remaining total24h'
                m=c.metrics_update(status=status,phase='budget_gate',forecast=forecast,calibration_complete=True,
                    stop_reason=reason+'; all60 real calibration attempts saved. No scale/route change chosen.')
                stop_readme(m['stop_reason'],m);c.log(m['stop_reason']);return
            status='complete';c.metrics_update(status='ready_formal',phase='formal_pending',forecast=forecast,calibration_complete=True)
            c.log(f'Real calibration60 complete; conservative remaining estimate={forecast["remaining_total_seconds_upper"]/3600:.2f}h, available={budget.remaining()/3600:.2f}h. Formal H2 may start.')
    except BudgetStop as exc:
        status='stopped_budget';m=c.metrics_update(status=status,phase='real_calibration',stop_reason=str(exc));stop_readme(str(exc),m);c.log(str(exc))
    except BaseException as exc:
        c.metrics_update(status='failed',phase='real_calibration',error=f'{type(exc).__name__}: {exc}');c.log(traceback.format_exc());raise
    finally:budget.close(status)

def write_predictions(rows):
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with c.safe(c.OUT/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows);f.flush()
        import os;os.fsync(f.fileno())

def formal():
    c.setup();cfg=c.read(c.OUT/'config.json');verify(cfg);m=c.read(c.OUT/'metrics.json')
    if (c.DATA/'evaluation_started.json').exists():raise RuntimeError('Frozen TEST/HOLDOUT evaluation already started; no reevaluation, even after a phase reset')
    if m.get('status')=='complete':raise RuntimeError('H2 already complete; final TEST is not reevaluated')
    if m.get('status')!='ready_formal':raise RuntimeError(f'Formal stage requires passed calibration/budget gate, got {m.get("status")}')
    if (c.DATA/'formal_attempts.jsonl').exists():raise RuntimeError('Partial formal output exists; no row-by-row recovery or automatic restart')
    budget=c.Budget('formal');status='failed';real=[];synthetic=[];selected={'TEST':[],'POOL':[]};attempts=0
    reached=[];counters=Counter();completed_ids=set()
    try:
        with c.Awake():
            from h2_forecast import build_forecast
            from h2_evaluation import evaluate_h2,write_readme
            from pymatgen.analysis.structure_matcher import StructureMatcher
            from p1_data import reduced_composition
            plan=c.gz_read(c.DATA/'frozen_plan.json.gz');calibration=c.gz_read(c.DATA/'calibration_results.json.gz')
            timing=c.read(c.DATA/'calibration_timing.json')
            raw_benchmark={str(w):c.gz_read(c.DATA/f'benchmark_results_workers{w}.json.gz') for w in (1,2,4)}
            forecast=build_forecast(plan,cfg['generation'],calibration,m['benchmark'],cfg['selected_workers'],budget.remaining(),calibration_wall_seconds=timing['wall_seconds'],benchmark_results=raw_benchmark)
            c.metrics_update(status='running',phase='formal_real',formal_started_utc=c.now(),forecast=forecast,progress_percent=.6,progress_milestones_reached=[])
            if not forecast['fits_budget']:raise BudgetStop('Updated forecast exceeds remaining24h at formal launch; formal labels not started')
            for r in calibration:real.append(r);completed_ids.add(r['row']['id'])
            c.append_jsonl(c.DATA/'formal_attempts.jsonl',real);attempts=len(real)
            initial_real_remaining=4000-len(real)
            matcher=StructureMatcher(ltol=.2,stol=.3,angle_tol=5,primitive_cell=True,scale=True,attempt_supercell=False,allow_subset=False)
            accepted_by_composition=defaultdict(list)
            real_batch_seconds=0.;real_new_n=0;synthetic_batch_seconds=0.;synthetic_new_n=0

            def remaining_forecast():
                nr=4000-len(real);ns=6000-sum(map(len,selected.values()))
                def stratum_key(row):
                    from h2_forecast import _size_bin
                    return row['split'],_size_bin(row['n_sites'])
                remaining_counts=Counter(stratum_key(x) for x in plan['real_candidates'] if x['id'] not in completed_ids)
                initial_work=sum(x['remaining_n']*x['mean_worker_wall_seconds'] for x in forecast['strata'])
                current_work=sum(remaining_counts[(x['split'],x['size_bin'])]*x['mean_worker_wall_seconds'] for x in forecast['strata'])
                r0=forecast['remaining_real_seconds_upper']*current_work/max(1e-12,initial_work)
                if real_new_n>=100:r0=max(r0,1.10*real_batch_seconds/real_new_n*nr)
                s0=forecast['remaining_synthetic_seconds_upper']*ns/6000
                n_accepted=sum(map(len,selected.values()))
                if n_accepted>=100:s0=max(s0,1.10*synthetic_batch_seconds/n_accepted*ns)
                return 1.05*(r0+s0+forecast['evaluation_reserve_seconds'])

            def checkpoint(phase):
                progress=(len(real)+sum(map(len,selected.values())))/10000*100
                for milestone in (25,50,75):
                    if progress>=milestone and milestone not in reached:
                        reached.append(milestone);c.log(f'H2 formal progress {milestone}%: real={len(real)}/4000, TEST={len(selected["TEST"])}/1000, POOL={len(selected["POOL"])}/5000')
                estimate=remaining_forecast();budget.heartbeat()
                c.metrics_update(status='running',phase=phase,progress_percent=progress,progress_milestones_reached=reached,
                    n_real_completed=len(real),retained_counts={k:len(v) for k,v in selected.items()},
                    formal_label_attempts=attempts,filter_counts=dict(counters),remaining_forecast_seconds_upper=estimate,
                    active_spent_seconds=budget.previous+budget.elapsed(),budget_remaining_seconds=budget.remaining())
                if estimate>budget.remaining():raise BudgetStop(f'Updated remaining forecast {estimate/3600:.2f}h exceeds remaining budget {budget.remaining()/3600:.2f}h; progress saved')

            pending=[x for x in plan['real_candidates'] if x['id'] not in completed_ids]
            for start in range(0,len(pending),100):
                checkpoint('formal_real');t=time.perf_counter();results=admitted_batch(pending[start:start+100],cfg['selected_workers'],budget,'real')
                real_batch_seconds+=time.perf_counter()-t;real_new_n+=len(results)
                c.append_jsonl(c.DATA/'formal_attempts.jsonl',results);real.extend(results);attempts+=len(results)
                completed_ids.update(r['row']['id'] for r in results)
                counters['real_resource_oom']+=sum('OutOfMemoryError:' in r['row']['error'] for r in results)
                if any('CUDA error' in r['row']['error'] for r in results):
                    raise RuntimeError('CUDA failure in formal real batch; saved progress, no automatic worker/route change')
            checkpoint('formal_synthetic')
            for split,target in [('TEST',1000),('POOL',5000)]:
                cases=plan['synthetic_candidates'][split]
                for start in range(0,len(cases),100):
                    if len(selected[split])>=target:break
                    checkpoint(f'formal_synthetic_{split}');t=time.perf_counter();results=admitted_batch(cases[start:start+100],cfg['selected_workers'],budget,split)
                    synthetic_batch_seconds+=time.perf_counter()-t;synthetic_new_n+=len(results)
                    # Candidate prefix order is frozen independently of parallel completion.
                    for r in results:
                        row=r['row'];row['accepted']=False;row['rejection_reason']=''
                        if 'OutOfMemoryError:' in row['error']:
                            row['rejection_reason']='resource_oom'
                        elif not row['accepted_physics']:
                            row['rejection_reason']=row['failed_checks'][0] if row['failed_checks'] else 'relaxation_error'
                        elif len(selected[split])>=target:row['rejection_reason']='quota_surplus'
                        else:
                            comp=reduced_composition(r['final_structure'])
                            assert comp==row['reduced_composition']
                            # The prefilter already enforces exact composition
                            # uniqueness across all BENCH/TEST/POOL candidates.
                            from h1_generation import clean_structure
                            fs=clean_structure(r['final_structure'])
                            duplicate=next((rid for rid,s in accepted_by_composition[comp] if matcher.fit(fs,s)),None)
                            if duplicate:row['rejection_reason']='duplicate';row['duplicate_of']=duplicate
                            else:
                                row['accepted']=True;selected[split].append(r);accepted_by_composition[comp].append((row['id'],fs))
                        counters[row['rejection_reason'] or 'accepted']+=1
                    c.append_jsonl(c.DATA/'formal_attempts.jsonl',results);synthetic.extend(results);attempts+=len(results)
                    if any('CUDA error' in r['row']['error'] for r in results):
                        raise RuntimeError('CUDA failure in formal synthetic batch; saved progress, no automatic worker/route change')
                if len(selected[split])!=target:raise RuntimeError(f'Frozen {split} reserve exhausted before retained quota; all progress saved, no data route change')
            checkpoint('final_evaluation')
            if budget.remaining()<forecast['evaluation_reserve_seconds']:raise BudgetStop('Insufficient reserved time for once-only RF evaluation; labels saved')
            final_synthetic=selected['TEST']+selected['POOL']
            c.gz_write(c.DATA/'synthetic_records_private.json.gz',final_synthetic)
            c.gz_write(c.DATA/'real_records_private.json.gz',real)
            # Irreversible within this frozen run: no model/feature selection
            # follows the first TEST/HOLDOUT evaluation.
            import os
            with c.safe(c.DATA/'evaluation_started.json').open('x',encoding='utf-8') as marker:
                json.dump({'started_utc':c.now(),'n_TEST':1000,'n_HOLDOUT':2000},marker)
                marker.flush();os.fsync(marker.fileno())
            evaluated=evaluate_h2(final_synthetic,real,cfg,log=c.log)
            c.write(c.DATA/'split_manifest.json',evaluated['split_manifest'])
            write_predictions(evaluated['prediction_rows'])
            metrics=evaluated['metrics'];metrics.update(
                phase='complete',completed_utc=c.now(),benchmark=m['benchmark'],benchmark_complete=True,calibration_complete=True,selected_workers=cfg['selected_workers'],
                forecast=forecast,formal_label_attempts=attempts,filter_counts=dict(counters),
                n_real_completed=len(real),retained_counts={k:len(v) for k,v in selected.items()},
                progress_percent=100.,progress_milestones_reached=[25,50,75,100],
                scientific_llm_calls=0,old_artifacts_unchanged=c.snapshot_old()==cfg['old_artifact_hashes'],
                active_seconds_before_formal=budget.previous,formal_elapsed_seconds=budget.elapsed(),
                total_active_seconds=budget.previous+budget.elapsed(),budget_seconds=c.BUDGET_SECONDS,
                input_plan_sha256=cfg['frozen_plan_sha256'],
                private_dataset_hashes={n:c.sha(c.DATA/n) for n in ['synthetic_records_private.json.gz','real_records_private.json.gz','split_manifest.json']})
            metrics['resource_revision']=cfg.get('resource_revision')
            metrics['gpu_admission_policy']=cfg['gpu_admission_policy']
            assert metrics['old_artifacts_unchanged'];assert metrics['total_active_seconds']<=c.BUDGET_SECONDS
            full_coverage=metrics['status']=='complete'
            if not full_coverage:
                metrics['phase']='complete_with_partial_real_labels'
                metrics['stop_reason']='All4000 real rows attempted; strict convergence failed for some labels. Full2000x2000 CHGNet comparison is unavailable; coverage and conditional metrics saved. Stop for human review before H3.'
            c.write(c.OUT/'metrics.json',metrics);write_readme(metrics,cfg,c.OUT/'README.md')
            assert len(list(c.OUT.iterdir()))==5
            status='complete' if full_coverage else 'partial_real_label_coverage'
            c.log(f'H2 labeling attempts100%: TEST1000,POOL5000,real4000 attempted,RFs evaluated once; result status={status}. STOP for review before H3.')
    except BudgetStop as exc:
        status='stopped_budget';m=c.metrics_update(status=status,phase='budget_stop',stop_reason=str(exc),
            n_real_completed=len(real),retained_counts={k:len(v) for k,v in selected.items()},formal_label_attempts=attempts)
        stop_readme(str(exc),m);c.log(str(exc))
    except BaseException as exc:
        c.metrics_update(status='failed',phase='formal',error=f'{type(exc).__name__}: {exc}',
                         n_real_completed=len(real),retained_counts={k:len(v) for k,v in selected.items()},formal_label_attempts=attempts)
        c.log(traceback.format_exc());raise
    finally:budget.close(status)

if __name__=='__main__':
    mp.freeze_support();p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','calibrate','formal']);a=p.parse_args()
    if a.command=='freeze':freeze_formal()
    elif a.command=='calibrate':calibrate()
    else:formal()
