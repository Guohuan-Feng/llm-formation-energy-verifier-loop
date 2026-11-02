"""Prepare H2 and compare the SAME40 new structures at1/2/4 GPU workers."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import math
import os
import multiprocessing as mp
import statistics
import subprocess
import threading
import time
import traceback
import h2_common as c

def bench_task(candidate):
    from h2_relax import relax_candidate,runtime_metadata
    t=time.perf_counter();r=relax_candidate(candidate)
    r['row']['task_started_perf_counter']=t;r['row']['task_finished_perf_counter']=time.perf_counter()
    return {'result':r,'runtime':runtime_metadata()}

class GpuSampler:
    def __init__(self):self.samples=[];self.stop=threading.Event()
    def sample(self):
        while not self.stop.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,creationflags=0x08000000)
                if p.returncode==0:
                    mem,util=map(float,p.stdout.strip().splitlines()[0].split(','));self.samples.append({'elapsed_seconds':time.perf_counter()-self.started,'memory_used_MiB':mem,'utilization_percent':util})
            except Exception:pass
            self.stop.wait(1.)
    def __enter__(self):self.started=time.perf_counter();self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=6)

def run_batch(candidates,workers,on_result=None,hard_deadline=None):
    from h2_relax import init_worker
    results={};runtimes={};started=time.perf_counter()
    os.environ['H2_GPU_WORKERS']=str(workers)
    with GpuSampler() as sampler:
        executor=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker)
        try:
            future_map={executor.submit(bench_task,x):x['id'] for x in candidates}
            timeout=max(1.,hard_deadline-time.perf_counter()) if hard_deadline is not None else None
            for f in as_completed(future_map,timeout=timeout):
                payload=f.result();r=payload['result'];results[r['row']['id']]=r;runtimes[r['row']['worker_pid']]=payload['runtime']
                if on_result:on_result(len(results),len(candidates),r)
        except BaseException:
            c.gz_write(c.DATA/'interrupted_batch.json.gz',{'workers':workers,'results':list(results.values()),'completed_utc':c.now()})
            # Cancel this finite phase only; never wait past its scientific
            # budget for already queued structures to finish.
            for process in list((executor._processes or {}).values()):
                if process.is_alive():process.terminate()
            executor.shutdown(wait=True,cancel_futures=True)
            raise
        else:executor.shutdown(wait=True)
    wall=time.perf_counter()-started
    ordered=[results[x['id']] for x in candidates]
    first=min(r['row']['task_started_perf_counter'] for r in ordered);last=max(r['row']['task_finished_perf_counter'] for r in ordered)
    summary={'workers':workers,'n':len(candidates),'wall_seconds_including_startup_shutdown':wall,
             'throughput_per_hour':len(candidates)*3600/wall,'compute_span_seconds':last-first,
             'compute_throughput_per_hour':len(candidates)*3600/(last-first),
             'peak_device_memory_used_MiB':max((x['memory_used_MiB'] for x in sampler.samples),default=None),
             'memory_sampling_interval_seconds':1,'memory_samples':len(sampler.samples),
             'mean_gpu_utilization_percent':statistics.fmean(x['utilization_percent'] for x in sampler.samples) if sampler.samples else None,
             'peak_allocated_MiB_per_worker':{str(pid):max(r['row']['torch_cuda_peak_allocated_MiB'] or 0 for r in ordered if r['row']['worker_pid']==pid) for pid in runtimes},
             'n_errors':sum(bool(r['row']['error']) for r in ordered),'n_converged':sum(r['row']['converged'] for r in ordered),
             'n_accepted_physics':sum(r['row']['accepted_physics'] for r in ordered),'runtime':list(runtimes.values())}
    c.gz_write(c.DATA/f'gpu_memory_workers{workers}.json.gz',sampler.samples)
    return ordered,summary

def prepare():
    c.setup()
    if (c.OUT/'config.json').exists():raise RuntimeError('H2 already prepared; never silently overwrite frozen inputs')
    budget=c.Budget('prepare');status='failed'
    try:
        from h2_generation import prepare_plan
        info,plan=prepare_plan(c.log)
        c.gz_write(c.DATA/'frozen_plan.json.gz',plan)
        cfg={'experiment':'H2_formal_dataset','created_utc':c.now(),'root':str(c.ROOT),'output_dir':str(c.OUT),'hdata_dir':str(c.DATA),
             'stage':2,'scope':'H2 only. No H3/LLM calls; stop after final report or if forecast exceeds total24h.',
             'target_retained_sizes':{'TEST':1000,'POOL':5000},'real_sizes':{'TRAIN':2000,'HOLDOUT':2000},
             'budget_seconds':c.BUDGET_SECONDS,'frozen_plan_sha256':c.sha(c.DATA/'frozen_plan.json.gz'),
             'old_artifact_hashes':c.snapshot_old(),'generation':info,'scientific_llm_calls':0,
             'benchmark_source_hashes':c.source_hashes(['h2_common.py','h2_benchmark.py','h2_generation.py','h2_relax.py','h1_generation.py','p1_data.py','p1_verifier.py']),
             'convergence':{'atomic_fmax_eV_A':.1,'cell_fmax_eV_A':.1,'full_native_filter_fmax_eV_A':.1,'inclusive':True,'max_steps':300},
             'pool_label_policy':'Stored separately under work/h-data/H2; public split manifest contains IDs only. Only S0 and frozen2000-label RF subset are released for their authorized roles.',
             'progress_policy':'Notify25/50/75/100 percent or errors; no per-minute reports.',
             'restart_policy':'No scheduler restart, no per-row recovery; frozen completed phases may feed subsequent phases. Interrupted phase requires explicit full-phase recomputation under remaining total budget.'}
        c.write(c.OUT/'config.json',cfg);c.metrics_update(status='prepared',phase='benchmark_pending',scientific_llm_calls=0)
        c.safe(c.OUT/'predictions.csv').write_text('dataset,id,model,y_true_eV_per_atom,prediction_eV_per_atom\n',encoding='utf-8')
        c.safe(c.OUT/'README.md').write_text('# H2 正式数据集\n已冻结新颖性预筛输入；先比较40个新候选在1/2/4个GPU worker下的吞吐与一致性。\n目标：TEST1000、POOL5000（均为过滤后保留）；真实TRAIN2000和HOLDOUT2000。\n总预算24小时；原子力及晶胞力均≤0.1 eV/Å；H2结束即停。\n',encoding='utf-8')
        status='complete';c.log('H2 inputs frozen; no known-Matbench composition will be relaxed as synthetic data.')
    finally:budget.close(status)

def benchmark():
    c.setup();cfg=c.read(c.OUT/'config.json');c.verify_config(cfg)
    if c.read(c.OUT/'metrics.json').get('benchmark_complete'):raise RuntimeError('Benchmark already complete; no silent rerun')
    budget=c.Budget('benchmark');status='failed';all_results={};summaries=[]
    try:
        with c.Awake():
            plan=c.gz_read(c.DATA/'frozen_plan.json.gz');cases=plan['benchmark_candidates'];assert len(cases)==40
            total_done=0
            for workers in (1,2,4):
                conservative_seconds=(summaries[0]['wall_seconds_including_startup_shutdown']*1.5/workers+120) if summaries else 40*14.2332*2+120
                if conservative_seconds>budget.remaining():raise TimeoutError('Benchmark forecast exceeds remaining total24h budget; progress saved')
                c.metrics_update(status='running',phase=f'benchmark_{workers}_workers',benchmark_completed_relaxations=total_done)
                def progress(n,total,result):
                    nonlocal total_done
                    done=total_done+n
                    if done%30==0:c.log(f'H2 parallel benchmark {done}/120 ({done/120:.0%}); current workers={workers}')
                    if result['row']['error']:c.log(f'ERROR benchmark {result["row"]["id"]}: {result["row"]["error"]}')
                    if n%10==0:
                        budget.heartbeat();c.metrics_update(benchmark_completed_relaxations=done)
                results,summary=run_batch(cases,workers,progress,hard_deadline=time.perf_counter()+budget.remaining()-30)
                all_results[str(workers)]=results;total_done+=40
                c.gz_write(c.DATA/f'benchmark_results_workers{workers}.json.gz',results)
                summary['energy_max_abs_difference_eV_per_atom']=0.;summary['matches_serial']=not summary['n_errors'] and summary['memory_samples']>=2
                summary['convergence_flags_match_serial']=True;summary['physics_flags_match_serial']=True
                if workers>1:
                    serial=all_results['1'];differences=[]
                    for a,b in zip(serial,results):
                        ea,eb=a['row']['formation_energy_eV_per_atom'],b['row']['formation_energy_eV_per_atom']
                        differences.append(abs(ea-eb) if ea is not None and eb is not None else math.inf)
                    maximum=max(differences)
                    summary['energy_max_abs_difference_eV_per_atom']=maximum if math.isfinite(maximum) else None
                    summary['convergence_flags_match_serial']=all(a['row']['converged']==b['row']['converged'] for a,b in zip(serial,results))
                    summary['physics_flags_match_serial']=all(a['row']['accepted_physics']==b['row']['accepted_physics'] for a,b in zip(serial,results))
                    summary['matches_serial']=(not summary['n_errors'] and summary['memory_samples']>=2 and maximum<.001 and summary['convergence_flags_match_serial'] and summary['physics_flags_match_serial'])
                summaries.append(summary)
                c.metrics_update(benchmark=summaries,benchmark_completed_relaxations=total_done)
                c.log(f'Benchmark workers={workers}: {summary["throughput_per_hour"]:.2f}/h, device peak={summary["peak_device_memory_used_MiB"]}MiB, max energy difference={summary["energy_max_abs_difference_eV_per_atom"]}, eligible={summary["matches_serial"]}')
            eligible=[s for s in summaries if s['matches_serial']]
            if not eligible:raise RuntimeError('No reliable GPU setting; stop H2 with benchmark progress saved')
            selected=max(eligible,key=lambda x:x['throughput_per_hour'])
            c.metrics_update(status='benchmark_complete',phase='real_calibration_pending',benchmark_complete=True,selected_workers=selected['workers'],benchmark=summaries)
            cfg['selected_workers']=selected['workers'];cfg['benchmark_selection']='Highest measured end-to-end throughput among error-free settings with every energy difference<0.001eV/atom and identical convergence/physics flags';c.write(c.OUT/'config.json',cfg)
            status='complete';c.log(f'H2 benchmark complete; selected{selected["workers"]}workers. Next:real-size calibration and24h forecast gate.')
    except TimeoutError as e:
        status='stopped_budget';c.metrics_update(status=status,phase='benchmark',error=str(e));c.log(str(e));raise SystemExit(75)
    except BaseException as e:
        c.metrics_update(status='failed',phase='benchmark',error=f'{type(e).__name__}: {e}');c.log(traceback.format_exc());raise
    finally:budget.close(status)

if __name__=='__main__':
    mp.freeze_support();p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','benchmark']);a=p.parse_args()
    if a.command=='prepare':prepare()
    else:benchmark()
