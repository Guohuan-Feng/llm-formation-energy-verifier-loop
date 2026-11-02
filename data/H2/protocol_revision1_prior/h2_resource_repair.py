"""One diagnosed hardware-cohort revision; formal candidates stay byte-identical.

The original NHe3 calibration candidate committed25.31GB GPU memory on16GB
hardware. Its original frozen inputs, code and nine completed results are kept
under resource_revision0. No original material is declared a physics failure.
"""
from collections import Counter
import copy
import gzip
import json
import shutil
import numpy as np
import h2_common as c

def main():
    c.setup();cfg=c.read(c.OUT/'config.json')
    assert cfg['old_artifact_hashes']==c.snapshot_old()
    assert not c.read(c.DATA/'budget.json').get('active_phase')
    archive=c.safe(c.DATA/'resource_revision0');assert archive.exists()
    for name in ['interrupted_batch.json.gz','benchmark_stdout.log','benchmark_stderr.log','budget.json']:
        if (c.DATA/name).exists():shutil.copy2(c.DATA/name,c.safe(archive/name))
    shutil.copy2(c.OUT/'metrics.json',c.safe(archive/'failure_metrics.json'))
    shutil.copy2(c.OUT/'run.log',c.safe(archive/'failure_run.log'))
    plan=c.gz_read(c.DATA/'frozen_plan.json.gz');old=copy.deepcopy(plan['benchmark_candidates'][9])
    assert old['id']=='H2-B-000009' and old['formula']=='NHe3'
    import h1_generation as h1
    import h2_generation as g
    import p1_data
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
    from p1_verifier import elemental_references
    raw=c.gz_read(p1_data.PILOT_PATH)
    template=next(r for r in raw['records'] if r['id']==old['template_id'])
    structure=h1.clean_structure(template['structure']);available=set(elemental_references())&set(h1.CovalentRadius.radius)
    options,_=g._options(template,structure,SubstitutionProbability(),available,Counter())
    known=set(c.gz_read(c.ROOT/'work/h-data/matbench_compositions.json.gz'))
    h1prior=c.gz_read(c.ROOT/'work/h-data/h1_frozen_inputs.json.gz')['candidates']
    prohibited=known|{p1_data.reduced_composition(x['initial_structure']) for x in h1prior}
    for xs in [plan['benchmark_candidates'],*plan['synthetic_candidates'].values()]:prohibited.update(x['reduced_composition'] for x in xs)
    def angle_count(s):
        i,j,img,d=s.get_neighbor_list(3.,exclude_self=True)
        degree=np.bincount(i,minlength=len(s));return int(np.sum(degree*(degree-1)))
    other_max=max(angle_count(h1.clean_structure(x['initial_structure'])) for n,x in enumerate(plan['benchmark_candidates']) if n!=9)
    replacement=None
    for option in options[old['substitution_method']]:
        candidate=h1._candidate(template,structure,option,old['substitution_method'],9)
        group=p1_data.reduced_composition(candidate['initial_structure'])
        if group in prohibited:continue
        angle=angle_count(h1.clean_structure(candidate['initial_structure']))
        if angle>other_max:continue
        candidate.update(id=old['id'],dataset_kind='synthetic',split='BENCH',reduced_composition=group)
        replacement=candidate;break
    if replacement is None:raise RuntimeError('No same-template, same-route numerical calibration replacement; human review needed')
    plan['benchmark_candidates'][9]=replacement
    for split in ['TEST','POOL']:
        assert h1._json_hash(plan['synthetic_candidates'][split])==cfg['generation']['synthetic_candidates_sha256'][split]
    assert h1._json_hash(plan['real_candidates'])==cfg['generation']['real_candidates_sha256']
    revision={'reason':'GPU shared-memory spill:25.31GB committed versus16GB physical; incomplete numerical calibration, not a physical rejection',
        'old_candidate':old,'replacement_candidate':replacement,'other39_max_initial_angle_pairs_3A':other_max,
        'replacement_initial_angle_pairs_3A':angle_count(h1.clean_structure(replacement['initial_structure'])),
        'replacement_selection':'First frozen H2 option order in same template and replacement route, novel/exclusive composition, initial3A angle count no larger than the other39 benchmark maximum; no formation-energy selection.',
        'formal_candidate_lists_unchanged':True,'real4000_unchanged':True,
        'whole_benchmark_recomputed':True,'completed_partial_rows_not_reused':True,
        'numerical_resource_guard':'Torch CUDA allocator limited to70percent of physical GPU VRAM divided by concurrent workers; resource OOM is reported, never relabeled as a valid energy.',
        'prior_failed_budget_seconds':c.read(c.DATA/'budget.json')['spent_seconds']}
    c.write(c.DATA/'resource_revision.json',revision)
    c.gz_write(c.DATA/'frozen_plan.json.gz',plan)
    cfg['frozen_plan_sha256']=c.sha(c.DATA/'frozen_plan.json.gz')
    cfg['generation']['benchmark_candidates_sha256']=h1._json_hash(plan['benchmark_candidates'])
    cfg['resource_revision']={k:v for k,v in revision.items() if k not in ['old_candidate','replacement_candidate']}
    cfg['resource_revision']['old_candidate_formula']=old['formula'];cfg['resource_revision']['replacement_candidate_formula']=replacement['formula']
    cfg['resource_revision']['record_path']=str(c.DATA/'resource_revision.json')
    names=list(cfg['benchmark_source_hashes'])+['h2_resource_repair.py']
    cfg['benchmark_source_hashes']=c.source_hashes(names)
    c.write(c.OUT/'config.json',cfg)
    m=c.read(c.OUT/'metrics.json');m.pop('error',None)
    m.update(status='prepared',phase='benchmark_pending_resource_revision1',benchmark_completed_relaxations=0,
        resource_revision=cfg['resource_revision'],benchmark_complete=False)
    c.write(c.OUT/'metrics.json',m);c.verify_config(cfg)
    c.log(f'Resource failure saved; benchmark replacement NHe3->{replacement["formula"]}, same template/method, angles={revision["replacement_initial_angle_pairs_3A"]}; all40 re-evaluated at1/2/4, formal inputs unchanged.')
    print(json.dumps({'replacement':replacement['formula'],'angles':revision['replacement_initial_angle_pairs_3A'],'prior_completed_partial':len(c.gz_read(archive/'interrupted_batch.json.gz')['results'])},ensure_ascii=False))

if __name__=='__main__':main()
