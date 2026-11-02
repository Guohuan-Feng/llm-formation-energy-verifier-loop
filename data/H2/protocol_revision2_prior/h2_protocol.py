"""Finite, user-authorized H2 protocol revision and same40 serial diagnostic."""
from __future__ import annotations
import argparse
import math
import multiprocessing as mp
import statistics
import time
import traceback
import h2_common as c

PRIOR=c.DATA/'protocol_revision1_prior'
FLAGS=('distance_pass','volume_pass','energy_pass','accepted_physics','failed_checks')

def compare(reference,other):
    a={x['row']['id']:x['row'] for x in reference};b={x['row']['id']:x['row'] for x in other}
    assert set(a)==set(b) and len(a)==len(reference)==len(other)==40
    details=[]
    for rid in sorted(a):
        x,y=a[rid],b[rid];ex=x['formation_energy_eV_per_atom'];ey=y['formation_energy_eV_per_atom']
        delta=ey-ex if ex is not None and ey is not None else None
        keys=['steps','atomic_fmax_eV_A','cell_fmax_eV_A','filtered_fmax_eV_A','converged','error',*FLAGS]
        details.append({'id':rid,'formula':x['formula'],'signed_energy_difference_eV_per_atom':delta,
            'absolute_energy_difference_eV_per_atom':abs(delta) if delta is not None else None,
            'original_serial':{k:x.get(k) for k in keys},'comparison':{k:y.get(k) for k in keys}})
    values=sorted(d['absolute_energy_difference_eV_per_atom'] for d in details if d['absolute_energy_difference_eV_per_atom'] is not None)
    import numpy as np
    summary={'n':40,'n_paired_energies':len(values),'n_missing_energy_pairs':40-len(values),
        'energy_max_abs_difference_eV_per_atom':max(values) if values else None,
        'energy_mean_abs_difference_eV_per_atom':statistics.fmean(values) if values else None,
        'energy_median_abs_difference_eV_per_atom':statistics.median(values) if values else None,
        'energy_p95_abs_difference_eV_per_atom':float(np.percentile(values,95)) if values else None,
        'n_energy_differences_above_1_meV':sum(v>.001 for v in values),
        'n_energy_differences_above_5_meV':sum(v>.005 for v in values),
        'convergence_flags_match_serial':all(a[k]['converged']==b[k]['converged'] for k in a),
        'physics_flags_match_serial':all(all(a[k][flag]==b[k][flag] for flag in FLAGS) for k in a),
        'individual_physical_flags_compared':list(FLAGS),'n_changed_step_counts':sum(a[k]['steps']!=b[k]['steps'] for k in a),
        'n_errors':sum(bool(x['error']) for x in b.values()),
        'comparison_quantity':'Formation energies at relaxed endpoints from identical initial structures; FIRE trajectories/endpoints may differ. This is not a fixed-final-geometry single-point comparison.'}
    return summary,details

def repeat_serial():
    c.setup();old=c.read(PRIOR/'experiment_config.json')
    assert c.snapshot_old()==old['old_artifact_hashes']
    for name in ('h2_common.py','h2_relax.py','h1_generation.py','p1_data.py','p1_verifier.py'):
        assert c.sha(c.ROOT/name)==old['benchmark_source_hashes'][name],name
    if (c.DATA/'serial_repeat_results.json.gz').exists():raise RuntimeError('Completed same40 serial diagnostic already exists; no extra diagnostic reruns')
    cases=c.gz_read(PRIOR/'frozen_plan.json.gz')['benchmark_candidates']
    assert cases==c.gz_read(c.DATA/'frozen_plan.json.gz')['benchmark_candidates']
    c.write(c.DATA/'serial_repeat_sources.json',c.source_hashes(['h2_protocol.py','h2_benchmark.py','h2_common.py','h2_relax.py','launch_h2_repeat.ps1']))
    budget=c.Budget('serial_repeat_protocol1');status='failed'
    try:
        with c.Awake():
            from h2_benchmark import run_batch
            c.metrics_update(status='running',phase='serial_repeat',serial_repeat_completed=0)
            def progress(n,total,result):
                if result['row']['error']:c.log(f'ERROR serial_repeat {result["row"]["id"]}: {result["row"]["error"]}')
                if n%10==0:
                    budget.heartbeat();c.metrics_update(serial_repeat_completed=n)
                    c.log(f'H2 same40 repeat-serial diagnostic {n}/{total} ({n/total:.0%}); descriptive only')
            result,summary=run_batch(cases,1,progress,hard_deadline=time.perf_counter()+budget.remaining()-30,memory_suffix='serial_repeat')
            comparison,details=compare(c.gz_read(c.DATA/'benchmark_results_workers1.json.gz'),result)
            summary.update(comparison);summary['diagnostic_only']=True;summary['used_as_gate']=False
            summary['interpretation']='Within-setting run-to-run differences are consistent with numerical nondeterminism when present; one repeat cannot establish that GPU floating point is the sole cause. Different FIRE endpoints may amplify small perturbations.'
            c.gz_write(c.DATA/'serial_repeat_results.json.gz',result);c.gz_write(c.DATA/'serial_repeat_comparison.json.gz',details)
            c.write(c.DATA/'serial_repeat_summary.json',summary)
            c.metrics_update(status='serial_repeat_complete',phase='protocol_revision_pending',serial_repeat=summary,serial_repeat_completed=40)
            c.log(f'Same40 serial repeat complete: max delta={1000*comparison["energy_max_abs_difference_eV_per_atom"]:.6f}meV/atom; diagnostic only, not a gate.')
            status='complete'
    except BaseException as exc:
        c.metrics_update(status='failed',phase='serial_repeat',error=f'{type(exc).__name__}: {exc}');c.log(traceback.format_exc());raise
    finally:budget.close(status)

def activate():
    c.setup();cfg=c.read(c.OUT/'config.json');m=c.read(c.OUT/'metrics.json');ledger=c.read(c.DATA/'budget.json')
    assert not ledger.get('active_phase') and m['status']=='serial_repeat_complete'
    assert not (c.DATA/'formal_attempts.jsonl').exists() and not (c.DATA/'evaluation_started.json').exists()
    old=c.gz_read(PRIOR/'frozen_plan.json.gz');plan=c.gz_read(c.DATA/'staged_plan_protocol1.json.gz');info=c.read(c.DATA/'staged_generation_info_protocol1.json')
    for key in ('benchmark_candidates','real_candidates','real_calibration_ids','template_manifest'):assert old[key]==plan[key],key
    assert c.snapshot_old()==cfg['old_artifact_hashes']
    summaries=[];reference=c.gz_read(c.DATA/'benchmark_results_workers1.json.gz')
    for s in m['benchmark']:
        s=dict(s);comparison,details=compare(reference,c.gz_read(c.DATA/f'benchmark_results_workers{s["workers"]}.json.gz'))
        s.update(comparison)
        s['energy_tolerance_eV_per_atom']=.005
        s['matches_serial']=(s['n_paired_energies']==40 and not s['n_errors'] and s['memory_samples']>=2
            and s['energy_max_abs_difference_eV_per_atom']<=.005 and s['convergence_flags_match_serial'] and s['physics_flags_match_serial'])
        c.gz_write(c.DATA/f'benchmark_comparison_protocol1_workers{s["workers"]}.json.gz',details);summaries.append(s)
    selected=max((s for s in summaries if s['matches_serial']),key=lambda s:s['throughput_per_hour'])
    assert selected['workers']==4
    revision={'id':1,'authorized_utc':c.now(),'energy_tolerance_eV_per_atom':.005,'inclusive':True,
        'previous_tolerance_eV_per_atom':.001,'require_identical_convergence_and_all_physical_filters':True,
        'engineering_reason':'User relaxes the engineering consistency criterion from1 to5meV/atom, approximately one tenth of prior P1 CHGNet-vs-DFT58meV/atom MAE; this is not a scientific accuracy guarantee.',
        'serial_repeat_diagnostic_only':True,'excluded_replacement_targets':['He','Ne','Ar','Kr','Xe','Rn'],
        'reject_initial_volume_per_atom_below_A3':4.,'reject_initial_angle_pairs_3A_above':20000,
        'filter_scope':'Formal synthetic generation only; original same40 benchmark and fixed4000 real records unchanged.',
        'prior_archive':str(PRIOR),'label_blind':'No formal synthetic relaxation or label exists when the revised generation is frozen.'}
    c.gz_write(c.DATA/'frozen_plan.json.gz',plan)
    cfg.update(protocol_revision=revision,generation=info,selected_workers=4,
        frozen_plan_sha256=c.sha(c.DATA/'frozen_plan.json.gz'),benchmark_selection='Highest measured throughput among <=.005eV/atom and identical convergence plus every individual physical filter. User-authorized engineering criterion.')
    names=list(cfg['benchmark_source_hashes'])+['h2_protocol.py','launch_h2_repeat.ps1']
    cfg['benchmark_source_hashes']=c.source_hashes(list(dict.fromkeys(names)))
    cfg.pop('formal_source_hashes',None);cfg.pop('source_hashes',None)
    c.write(c.OUT/'config.json',cfg)
    preparation=float(info.get('generation_wall_seconds',0))
    if preparation:
        ledger['spent_seconds']+=preparation
        ledger['phases'].append({'name':'protocol1_generation','elapsed_seconds':preparation,'status':'complete','completed_utc':c.now(),'accounting':'CPU preparation charged conservatively, including any overlap with serial diagnostic'})
        c.write(c.DATA/'budget.json',ledger)
    c.metrics_update(status='benchmark_complete',phase='real_calibration_pending',selected_workers=4,benchmark=summaries,
        protocol_revision=revision,generation_counts=info['generation_counts'],prefilter_counts=info['prefilter_counts'],
        forecast={},stop_reason=None,error=None,calibration_complete=False,formal_started=False)
    c.safe(c.OUT/'README.md').write_text('# H2 修订后继续\n用户将工程一致性门槛从1改为≤5 meV/atom（约58 meV CHGNet/DFT误差的1/10），并要求全部收敛及物理过滤标志一致；选4 workers。\n同40候选串行复测仅作诊断，完整差异见metrics.json。\n正式候选在标签前排惰性替换目标、vpa<4 Å³或3 Å角对>20000；剔除计数见metrics.json。\n真实4000与原40基准不变；下一步为60个真实结构的分层耗时校准与24h预算门。\n',encoding='utf-8')
    c.log(f'User protocol revision1 frozen before formal labels: <=5meV and all filter flags; selected4workers; prefilter counts={info["prefilter_counts"]}. Next real60 calibration and remaining-budget gate.')

if __name__=='__main__':
    mp.freeze_support();p=argparse.ArgumentParser();p.add_argument('command',choices=['repeat','activate']);args=p.parse_args()
    if args.command=='repeat':repeat_serial()
    else:activate()
