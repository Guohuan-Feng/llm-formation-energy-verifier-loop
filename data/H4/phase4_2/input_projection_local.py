"""User-ordered step1: local input-size arithmetic, no model or target access."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'work/h-data/H4/src'))
import json
import math
from fractions import Fraction
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import h4_l0_common as c
import h4_l0_data as data
import h4_l0_prompts as p
import h4_l0_queries as q
import h4_data as old_data

def forbidden(*args,**kwargs):raise RuntimeError('Projection may not access/reveal targets')
data.reveal=forbidden;data.initial_labels=forbidden;data.score_truth=forbidden
old_data.reveal_training=forbidden;old_data.score_truth=forbidden

def ledger_count(path):
    rows=[json.loads(line) for line in Path(path).read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
    return sum(x.get('record_type')=='reservation' for x in rows)

def fit_trend(rows):
    n=len(rows);sx=sum(x for x,y in rows);sy=sum(y for x,y in rows)
    slope=Fraction(n*sum(x*y for x,y in rows)-sx*sy,n*sum(x*x for x,y in rows)-sx*sx)
    intercept=(Fraction(sy)-slope*sx)/n;prediction=intercept+20*slope
    return {'slope_bytes_per_round':float(slope),'intercept_bytes':float(intercept),
        'round20_projection_bytes':float(prediction),'round20_exact_fraction':str(prediction)}

def main():
    cfg=c.read(c.PHASE/'config.json');c.verify(cfg)
    if cfg['prompts']['input_limit']!=20000:raise RuntimeError('Projection step must precede amendment')
    before_count=ledger_count(c.OUT/'llm_calls.jsonl');before_sha=c.sha(c.OUT/'llm_calls.jsonl')
    histories=[];trends=[]
    for seed in (0,1,2):
        values=[];observations=[]
        for rnd in range(1,7):
            path=c.PHASE/'requests'/f'L0_seed{seed}_round{rnd:02d}.json';request=c.read(path)
            ordinary=p.audit_input_bound(request['prompt'],request['schema'])
            repair=p.audit_input_bound(request['prompt'],request['schema'],include_repair=True)
            if ordinary!=request['input_budget'] or repair!=request['format_repair_input_budget']:
                raise RuntimeError('Recorded budget differs from exact frozen method')
            y=max(ordinary['input_token_upper_bound'],repair['input_token_upper_bound'])
            values.append((rnd,y));observations.append({'round':rnd,'ordinary_bytes':ordinary['input_token_upper_bound'],
                'with_format_repair_bytes':repair['input_token_upper_bound'],'request_sha256':c.sha(path)})
        result=fit_trend(values);trends.append({'seed':seed,**result});histories.append({'seed':seed,'observations':observations})
    a_exact=max(Fraction(t['round20_exact_fraction']) for t in trends);a_integer=math.ceil(a_exact)
    new_limit=1000*math.ceil(a_exact*Fraction(11,10)/1000)
    # Label-free fixed POOL metadata supplies finite formula/key domains only.
    bundle=data.load_bundle()
    formula_max=max(len(bundle['by_id'][rid]['formula'].encode('utf-8')) for rid in bundle['pool_ids'])
    elements=q.allowed_elements(bundle)
    n_elements=sorted({bundle['by_id'][rid]['n_elements'] for rid in bundle['pool_ids']})
    sg_keys=sorted({bundle['by_id'][rid]['template_spacegroup'] for rid in bundle['pool_ids']})
    fs=cfg['prompts']['feature_schema']['properties']['features'];item=fs['items']['properties']
    count=fs['maxItems'];name_chars=item['name']['maxLength'];desc_chars=item['description']['maxLength'];source_chars=item['source']['maxLength']
    # Per Unicode scalar: raw UTF8<=4 bytes; JSON string escaping<=6 bytes.
    source_bytes=4*source_chars;name_json_bytes=2+6*name_chars;description_json_bytes=2+6*desc_chars
    code_bound=0
    for index in range(count):
        code_bound+=len(('@'+str(index)+' n=').encode())+name_json_bytes+len(' d=')+description_json_bytes
        code_bound+=len(' B=')+len(str(source_bytes))+len('\n')+source_bytes+len('\n@end')
    code_bound+=count-1
    width=24 # Covers .6g/.9g rendering of every accepted finite binary64 number.
    numeric='X'*width
    energy={k:numeric for k in ('min','mean','std_ddof0')};energy['quantiles_0_25_50_75_100']=[numeric]*5
    summary={'labeled_n':2000,'energy_eV_per_atom':energy,'element_material_counts':{e:2000 for e in elements},
        'n_elements_distribution':{str(k):2000 for k in n_elements},'n_sites_quantiles_0_25_50_75_100':[numeric]*5}
    summary_bound=len(p._summary_text(summary).encode())
    catalogue={'unlabeled_POOL_n':5000,'element_material_counts':{e:5000 for e in elements},
        'n_elements_distribution':{str(k):5000 for k in n_elements},'n_sites_quantiles_0_25_50_75_100':[1.0]*5,
        'template_spacegroup_distribution':{str(k):5000 for k in sg_keys}}
    catalogue_bound=len(p._catalogue_text(catalogue).encode())+5*(width-len(p._number(1.0,p.NUMERIC_FORMAT['summary'])))
    worst_header='formula\tlabel_eV_per_atom\tprediction_eV_per_atom\tabs_error_eV_per_atom'
    worst_bound=len(worst_header.encode())+20*(1+formula_max+3+3*width)
    sections=(p.PREFIX+p.DICTIONARY_CONTRACT+q.grammar_text(1500)+'\n'+
        'Active code: @i n=name d=description B=source_UTF8_bytes; full original source follows each header; @end closes:\n'+'\n'+
        'Current validation rowMAE eV/atom: '+'\n'+
        'Current worst validation rows (TSV; largest absolute errors first):\n'+'\n'+
        'Current labeled-set summary:\n'+'\n'+
        'Unlabeled POOL catalogue (no target values):\n'+'\n'+
        'Return only final JSON with features,queries,rationale.\n')
    static_bytes=len(sections.encode())
    canonical_bound=static_bytes+code_bound+width+worst_bound+summary_bound+catalogue_bound
    schema=p.response_schema(bundle,1500);wire=p.wire_schema(schema);wire_note=q.WIRE_NOTE if p._json(wire)!=p._json(schema) else ''
    schema_bytes=len(p.canonical_schema_bytes(wire));global_bytes=p.feature_contract._global_instruction_bytes()
    instruction_bytes=len(p.INSTRUCTIONS.encode());repair_bytes=len(p.FORMAT_REPAIR.encode());note_bytes=len(wire_note.encode())
    b_ordinary=canonical_bound+note_bytes+global_bytes+instruction_bytes+schema_bytes+1500
    b_repair=b_ordinary+repair_bytes
    after_count=ledger_count(c.OUT/'llm_calls.jsonl')
    if after_count!=before_count or c.sha(c.OUT/'llm_calls.jsonl')!=before_sha:raise RuntimeError('Ledger changed during local-only projection')
    result={'step':1,'created_utc':datetime.now(timezone.utc).isoformat(),
        'created_client_time':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'config_sha256_before_amendment':c.sha(c.PHASE/'config.json'),
        'method':cfg['prompts']['input_budget_method'],'unit':'conservative UTF8-serialized byte upper bound; not measured model tokens',
        'a':{'method':'Per seed ordinary least-squares input size versus request round1..6; extrapolate to20; take largest seed; include format-repair suffix',
            'observations':histories,'per_seed_trends':trends,'round20_projection_exact_fraction':str(a_exact),
            'round20_projection_bytes':float(a_exact),'round20_projection_ceil_bytes':a_integer,
            'gate_threshold_bytes':40000,'exceeds_twice_original_limit':a_exact>40000,
            'authorized_10pct_ceiling_rounded_to_1000':new_limit,'not_a_future_size_guarantee':True},
        'b':{'theoretical_serialized_upper_bound_bytes':b_repair,'ordinary_upper_bound_bytes':b_ordinary,
            'interpretation':'A conservative schema/data-domain upper bound BEFORE applying the input guard; component maxima need not be simultaneously attainable; not a claim of a tight achievable maximum.',
            'caps':{'max_features':count,'source_maxLength_characters_each':source_chars,
                'name_maxLength_characters_each':name_chars,'description_maxLength_characters_each':desc_chars,
                'source_UTF8_max_bytes_per_character':4,'JSON_name_description_max_bytes_per_character':6,
                'feedback_rows':20,'finite_number_rendering_width_bound':width,'max_formula_UTF8_bytes_in_fixed_POOL':formula_max,
                'query_capacity_max':1500,'max_labeled_count_display_digits':4,'max_catalogue_count_display_digits':4},
            'components':{'complete_active_code_block_bytes':code_bound,'fixed_template_sections_bytes':static_bytes,
                'validation_MAE_display_bytes':width,'worst20_feedback_bytes':worst_bound,'labeled_summary_bytes':summary_bound,
                'unlabeled_catalogue_bytes':catalogue_bound,'strict_wire_schema_bytes':schema_bytes,'wire_note_bytes':note_bytes,
                'global_instructions_bytes':global_bytes,'model_instructions_bytes':instruction_bytes,
                'framing_bytes':1500,'format_repair_suffix_bytes':repair_bytes},
            'post_guard_call_limit_is_separate':20000},
        'checks':{'scientific_model_calls':0,'model_fit_predict_calls':0,'generated_feature_executions':0,
            'new_POOL_labels_revealed':0,'TEST_targets_read':0,'L0_scoring_started':(c.PHASE/'evaluation_started.json').exists(),
            'global_CLI_calls_before':before_count,'global_CLI_calls_after':after_count,'ledger_sha256_unchanged':before_sha,
            'configuration_amended':False,'frozen_source_and_old_stage_verification':'PASS'}}
    c.write(c.PHASE/'input_limit_projection.json',result,exclusive=True)
    print(json.dumps({'a':float(a_exact),'a_ceil':a_integer,'b_upper_bound':b_repair,'a_gt40000':a_exact>40000,
        'new_limit_if_gate_passes':new_limit,'projection_sha256':c.sha(c.PHASE/'input_limit_projection.json')}))
if __name__=='__main__':main()
