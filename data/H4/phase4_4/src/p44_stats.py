"""Frozen-algorithm H4.4 controls/transfer statistics, each domain read once.

No training, prediction, feature execution, LLM request or target access occurs
on import/configuration. Only evaluate_synthetic/evaluate_real call their new
independent domain truth accessor after twelve complete predictions and an
exclusive caller-owned marker. Old O/L0/L1 results and caches remain read-only.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / 'work/h-data/H4'
PHASE = DATA / 'phase4_4'
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from collections import Counter
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import os
import numpy as np
import h3b_stats as frozen

SYNTHETIC_GROUPS = ('HGB_Magpie132_n2000', 'HGB_Magpie132_n5000', 'CGCNN_n2000', 'CGCNN_n5000')
REAL_GROUPS = ('O', 'L0', 'L1', 'Magpie_HGB')
SEEDS = (0, 1, 2)
N_BOOT = 2000
BOOT_SEEDS = {'synthetic': 20261030, 'real': 20261030 + 110}
N_ROWS = {'synthetic': 1000, 'real': 2000}
NEW_CACHE_PROTOCOL = 'H4.4-domain-paired-bootstrap-draws-v1'
OLD_CACHE_PROTOCOL = 'H4-private-paired-bootstrap-draws-v1'
USED = set()
LAST_CACHE = {}
OLD_ROW_REFERENCES = {
    'L1_minus_O': {'point': '-0.005275', 'ci95': ['-0.013613', '0.002602']},
    'L1_minus_L0': {'point': '-0.002149', 'ci95': ['-0.008738', '0.004175']},
    'L0_minus_O': {'point': '-0.003126', 'ci95': ['-0.010755', '0.004133']}}
OLD_L0_O_REGISTERED = {
    'row_weighted': {'point': '-0.003126', 'ci95': ['-0.010755', '0.004133']},
    'group_weighted': {'point': '-0.002522', 'ci95': ['-0.010568', '0.005686']}}


class ScientificStop(RuntimeError):
    pass


def _read_path(path):
    path = Path(path).resolve(strict=True)
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(ROOT.resolve()):
        raise ScientificStop('Statistics inputs must remain in the non-OneDrive research root')
    return path


def _read(path):
    return json.loads(_read_path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    h = hashlib.sha256()
    with _read_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _paths(domain):
    if domain not in ('synthetic', 'real'): raise ScientificStop('Unknown evaluation domain')
    return {'marker': PHASE / ('evaluation_started.json' if domain == 'synthetic' else 'holdout_evaluation_started.json'),
        'terminal': PHASE / f'{domain}_terminal_results.json', 'cache': PHASE / f'{domain}_pairing_cache.npz'}


def configuration():
    if (frozen.N_BOOT != N_BOOT or tuple(frozen.METRICS) != ('mae', 'rmse', 'r2')
        or tuple(frozen.MODES) != ('row_weighted', 'group_weighted')):
        raise ScientificStop('The frozen H3b statistical algorithm identity differs')
    prior = {condition: {'marker_path': str(DATA / phase / 'evaluation_started.json'),
        'marker_sha256': _sha(DATA / phase / 'evaluation_started.json')}
        for condition, phase in (('O', 'phase4_1'), ('L0', 'phase4_2'), ('L1', 'phase4_3'))}
    common = {'stage': '4.4', 'scoring_pass': 1, 'targets_previously_read_for_this_phase_domain': False}
    synthetic = {**common, 'domain': 'synthetic', 'condition': 'controls',
        'prior_H4_synthetic_TEST_scoring_passes': 3, 'current_H4_synthetic_TEST_scoring_pass': 4,
        **{f'prior_{condition}_marker_sha256': value['marker_sha256'] for condition, value in prior.items()}}
    real = {**common, 'domain': 'real', 'condition': 'transfer',
        'current_H4_real_transfer_HOLDOUT_scoring_pass': 1, 'historical_HOLDOUT_pass_count_known': False}
    return {'schema_version': 'H4-stage4.4-joint-domain-group-statistics-v1',
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES), 'seeds': list(SEEDS), 'bootstrap_n': N_BOOT,
        'bootstrap_algorithm': 'Frozen h3b_stats._bootstrap_context/_score/_paired; shared template/composition multinomial group draws; percentile95 intervals',
        'seed_aggregation': 'arithmetic mean of three seed-specific metrics and their shared draws, never ensemble predictions or best seed',
        'primary_seed_mean_requires_all3': True, 'domain_gate_requires_all12_complete': True,
        'domains': {
            'synthetic': {'groups': list(SYNTHETIC_GROUPS), 'n_arms': 12, 'n_rows': 1000,
                'paths': {key: str(path) for key, path in _paths('synthetic').items()},
                'grouping': 'fixed original synthetic TEST template identity', 'group_weights': 'inverse full TEST1000 group count',
                'bootstrap_seed': BOOT_SEEDS['synthetic'], 'marker_contract': synthetic,
                'truth_accessor': 'p44_data.synthetic_truth', 'prior_results_not_rescored': ['O', 'L0', 'L1']},
            'real': {'groups': list(REAL_GROUPS), 'n_arms': 12, 'n_rows': 2000,
                'paths': {key: str(path) for key, path in _paths('real').items()},
                'grouping': 'frozen reduced-composition identity', 'group_weights': 'inverse full HOLDOUT2000 group count',
                'bootstrap_seed': BOOT_SEEDS['real'], 'marker_contract': real,
                'truth_accessor': 'p44_data.real_holdout_truth', 'requires_synthetic_terminal_cache_committed_first': True,
                'historical_HOLDOUT_access_count': None}},
        'runtime_marker_additions': {'all_domains': ['owner_pid', 'config_sha256', 'committed_predictions_sha256'],
            'real': ['synthetic_evaluation_marker_sha256', 'synthetic_terminal_sha256', 'synthetic_pairing_cache_sha256']},
        'synthetic_control_pairs': [{'A': 'L1', 'B': group} for group in SYNTHETIC_GROUPS],
        'real_transfer_pairs': [{'A': 'L1', 'B': 'Magpie_HGB'}],
        'old_feature_pairs_reuse': {'path': str(DATA / 'phase4_3/paired_comparisons.json'),
            'row_six_decimal_reference': deepcopy(OLD_ROW_REFERENCES), 'new_bootstrap_or_subtraction_for_old_pairs': False},
        'cache_protocol': NEW_CACHE_PROTOCOL, 'cache_target_values_present': False,
        'selection_uses_TEST_HOLDOUT': False, 'new_LLM_calls': 0,
        'numerical_sources': {str((ROOT / 'h3b_stats.py').resolve()): _sha(ROOT / 'h3b_stats.py'),
            str(Path(__file__).resolve()): _sha(__file__)}, 'prior_H4_scoring_markers': prior,
        'numpy_version': np.__version__}


def _marker(domain, marker, *, consume=False):
    path = _read_path(marker)
    if path != _read_path(_paths(domain)['marker']): raise ScientificStop('Wrong canonical H4.4 domain gate')
    contract = configuration()['domains'][domain]['marker_contract']; value = _read(path)
    if any(type(value.get(key)) is not type(expected) or value.get(key) != expected for key, expected in contract.items()):
        raise ScientificStop('Domain marker does not bind the frozen first-pass condition and prior history')
    if consume and (type(value.get('owner_pid')) is not int or value['owner_pid'] != os.getpid()):
        raise ScientificStop('Only the process which exclusively created this scoring gate may consume it')
    if consume:
        config_sha = _sha(PHASE / 'config.json')
        if (value.get('config_sha256') != config_sha
            or _read(PHASE / 'frozen_config.json').get('sha256') != config_sha):
            raise ScientificStop('The consuming marker must bind the unchanged canonical frozen config')
    if domain == 'real':
        required = _paths('synthetic'); terminal = _read(required['terminal'])
        if (terminal.get('domain') != 'synthetic' or terminal.get('target_accessor_calls') != 1
            or terminal.get('evaluation_marker', {}).get('sha256') != _sha(required['marker'])
            or value.get('synthetic_evaluation_marker_sha256') != _sha(required['marker'])
            or value.get('synthetic_terminal_sha256') != _sha(required['terminal'])
            or value.get('synthetic_pairing_cache_sha256') != _sha(required['cache'])):
            raise ScientificStop('Real HOLDOUT may be read only after the committed synthetic terminal and cache')
    return path


def _predictions(arm, ids):
    rows = arm.get('prediction_rows')
    if not isinstance(rows, list) or len(rows) != len(ids): raise ScientificStop('Complete fixed-domain prediction rows are required')
    values = {}
    for row in rows:
        if (not isinstance(row, dict) or type(row.get('id')) is not str or row['id'] in values
            or type(row.get('prediction_eV_per_atom')) not in (int, float) or not math.isfinite(row['prediction_eV_per_atom'])):
            raise ScientificStop('Predictions must be finite, numeric, uniquely identified and label-free')
        values[row['id']] = float(row['prediction_eV_per_atom'])
    if set(values) != set(ids): raise ScientificStop('Every arm must predict the full frozen domain, without subset scoring')
    return [values[rid] for rid in ids]


def _normalise(arms, domain, ids):
    groups = SYNTHETIC_GROUPS if domain == 'synthetic' else REAL_GROUPS
    if not isinstance(arms, list) or len(arms) != 12: raise ScientificStop('Exactly four groups by three seed arms are required')
    expected = {(group, seed) for group in groups for seed in SEEDS}; by_identity = {}
    for arm in arms:
        if not isinstance(arm, dict): raise ScientificStop('Malformed evaluation arm')
        group, seed = arm.get('group'), arm.get('seed')
        if (type(seed) is not int or (group, seed) not in expected or (group, seed) in by_identity
            or arm.get('arm_id') != f'{group}_seed{seed}'
            or arm.get('status') not in ('complete', 'ok', 'reused_complete')):
            raise ScientificStop('All twelve preregistered complete group/seed identities must be present before truth access')
        by_identity[(group, seed)] = {'arm_id': arm['arm_id'], 'group': group, 'seed': seed,
            'status': arm['status'], 'method': arm.get('method'), 'predictions': _predictions(arm, ids)}
    if set(by_identity) != expected: raise ScientificStop('Incomplete domain arm identities')
    return [by_identity[(group, seed)] for group in groups for seed in SEEDS]


def _verify_committed_predictions(domain, marker, normal, ids):
    saved_path = PHASE / 'committed_predictions.json'
    if _read(marker).get('committed_predictions_sha256') != _sha(saved_path):
        raise ScientificStop('The evaluation gate no longer binds the already committed prediction payload')
    saved = _read(saved_path)
    if saved.get('evaluation_targets_read_before_commit') is not False:
        raise ScientificStop('Prediction commitment must explicitly precede all new evaluation target reads')
    committed = _normalise(saved.get('synthetic_arms' if domain == 'synthetic' else 'real_arms'), domain, ids)
    for actual, expected in zip(normal, committed):
        identity_keys = ('arm_id', 'group', 'seed', 'status', 'method')
        if (any(actual[key] != expected[key] for key in identity_keys)
            or len(actual['predictions']) != len(expected['predictions'])
            or any(a.hex() != b.hex() for a, b in zip(actual['predictions'], expected['predictions']))):
            raise ScientificStop('Passed domain predictions differ by identity or float bits from the pre-target commitment')


def _evaluate(domain, arms, bundle, marker):
    paths = _paths(domain); marker = _marker(domain, marker, consume=True)
    if paths['terminal'].exists() or paths['cache'].exists() or str(marker) in USED:
        raise ScientificStop('This domain scoring pass has already been consumed; do not reread labels')
    ids = list(bundle['synthetic_test_ids' if domain == 'synthetic' else 'real_holdout_ids'])
    grouping = bundle['synthetic_groups' if domain == 'synthetic' else 'real_groups']
    if len(ids) != N_ROWS[domain] or len(set(ids)) != N_ROWS[domain] or not set(ids) <= set(grouping):
        raise ScientificStop('The full ordered evaluation identities and frozen label-free groups must be present')
    normal = _normalise(arms, domain, ids)
    if domain == 'synthetic' and set(ids) & set(bundle.get('synthetic_pool_ids', [])):
        raise ScientificStop('Synthetic TEST and POOL may not overlap')
    if domain == 'real' and set(ids) & set(bundle.get('real_train_ids', [])):
        raise ScientificStop('DFT training and HOLDOUT may not overlap')
    _verify_committed_predictions(domain, marker, normal, ids)
    USED.add(str(marker)); LAST_CACHE.pop(domain, None)
    import p44_data
    accessor = p44_data.synthetic_truth if domain == 'synthetic' else p44_data.real_holdout_truth
    truth, provenance = accessor(ids, evaluation_marker=marker, bundle=bundle)
    if (not isinstance(truth, dict) or set(truth) != set(ids)
        or any(type(truth[rid]) not in (float, int) or not math.isfinite(truth[rid]) for rid in ids)):
        raise ScientificStop('The one domain accessor must return aligned finite complete targets')
    groups = [str(grouping[rid]) for rid in ids]; full = dict(Counter(groups))
    predictions = {arm['arm_id']: arm['predictions'] for arm in normal}
    context = frozen._bootstrap_context([truth[rid] for rid in ids], predictions, groups, full, BOOT_SEEDS[domain])
    per_arm = {arm['arm_id']: {'arm_id': arm['arm_id'], 'group': arm['group'], 'seed': arm['seed'],
        'status': arm['status'], 'method': arm['method'], 'scores': frozen._score(context, [arm['arm_id']])} for arm in normal}
    names = SYNTHETIC_GROUPS if domain == 'synthetic' else REAL_GROUPS
    per_group = {group: {'group': group,
        'per_seed': {str(seed): deepcopy(per_arm[f'{group}_seed{seed}']) for seed in SEEDS},
        'arithmetic_seed_mean': frozen._score(context, [f'{group}_seed{seed}' for seed in SEEDS])} for group in names}
    LAST_CACHE[domain] = {'protocol': np.asarray(NEW_CACHE_PROTOCOL), 'stage': np.asarray('4.4'),
        'domain': np.asarray(domain), 'arm_names': np.asarray(context['names']),
        'point': context['point'].copy(), 'draws': context['draws'].copy(),
        'evaluation_ids': np.asarray(ids), 'groups': np.asarray(groups),
        'bootstrap_seed': np.asarray(BOOT_SEEDS[domain]), 'n_boot': np.asarray(N_BOOT),
        'full_group_sizes_json': np.asarray(json.dumps(full, sort_keys=True, separators=(',', ':'))),
        'target_values_in_cache': np.asarray(False), 'evaluation_marker_sha256': np.asarray(_sha(marker))}
    return {'schema_version': 'H4.4-complete-domain-metrics-v1', 'stage': '4.4', 'domain': domain,
        'unit': 'eV/atom', 'n_test' if domain == 'synthetic' else 'n_holdout': len(ids),
        'n_groups': context['n_groups'], 'n_arms': 12, 'n_boot': N_BOOT, 'bootstrap_seed': BOOT_SEEDS[domain],
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES),
        'grouping': configuration()['domains'][domain]['grouping'],
        'target_accessor_calls': 1, 'TEST_target_accessor_calls' if domain == 'synthetic' else 'HOLDOUT_target_accessor_calls': 1,
        'target_source': provenance, 'evaluation_marker': {'path': str(marker), 'sha256': _sha(marker)},
        'evaluation_ids_sha256': _json_hash(ids), 'groups_sha256': _json_hash(groups),
        'full_group_sizes_sha256': _json_hash(full), 'per_arm': per_arm, 'per_group': per_group,
        'seed_aggregation': 'arithmetic mean of metrics and shared draws, never ensemble predictions',
        'old_O_L0_L1_results_recomputed': False, 'model_fit_predict_or_selection_calls': 0,
        'H4_synthetic_TEST_passes_after_domain': 4 if domain == 'synthetic' else None,
        'H4_real_transfer_HOLDOUT_passes_after_domain': 1 if domain == 'real' else None,
        'historical_all_experiments_HOLDOUT_pass_count': None,
        'numerical_sources': configuration()['numerical_sources'],
        'interpretation_limits': ['All12 predictions precede this domain target read; no TEST/HOLDOUT model choice.',
            'Weights use the full evaluation-domain group counts, not a completed subset.',
            'Seed means average metrics; identical baseline predictions are not independent randomized repeats.',
            'Prior model-scoring exposures in other experiments are separate from these H4 domain-pass counters.']}


def evaluate_synthetic(arms, bundle, *, evaluation_marker):
    return _evaluate('synthetic', arms, bundle, evaluation_marker)


def evaluate_real(arms, bundle, *, evaluation_marker):
    return _evaluate('real', arms, bundle, evaluation_marker)


def private_pairing_cache(domain):
    if domain not in LAST_CACHE: raise ScientificStop('No successfully completed new domain cache is available')
    return {key: value.copy() for key, value in LAST_CACHE[domain].items()}


def _arrays(cache):
    if isinstance(cache, (str, Path)):
        path = _read_path(cache)
        with np.load(path, allow_pickle=False) as archive: values = {key: archive[key].copy() for key in archive.files}
        return values, {'path': str(path), 'sha256': _sha(path)}
    if isinstance(cache, dict): return {key: np.asarray(value).copy() for key, value in cache.items()}, {'source': 'explicit in-memory cached metric draws'}
    raise ScientificStop('Expected an already committed domain cache path or array mapping')


def _validate_arrays(values, *, names, ids_key, sizes_key, n_rows, seed):
    if values['arm_names'].tolist() != names or values['n_boot'].shape != () or values['n_boot'].item() != N_BOOT:
        raise ScientificStop('Cached arm order or2000-draw identity differs')
    if values['bootstrap_seed'].shape != () or values['bootstrap_seed'].item() != seed:
        raise ScientificStop('Cached fixed bootstrap seed differs')
    ids, groups = values[ids_key].tolist(), values['groups'].tolist()
    if (len(ids) != n_rows or len(set(ids)) != n_rows or len(groups) != n_rows
        or any(type(x) is not str for x in ids + groups)
        or json.loads(values[sizes_key].item()) != dict(Counter(groups))):
        raise ScientificStop('Cached complete domain identities/groups/weights differ')
    for key, shape in (('point', (2, len(names), 3)), ('draws', (N_BOOT, 2, len(names), 3))):
        a = values[key]
        if a.shape != shape or a.dtype != np.dtype('float64') or np.isinf(a).any() or not np.isfinite(a[..., :2]).all() or (a[..., :2] < 0).any():
            raise ScientificStop('Cached metric shape/dtype/finite values differ')
    return ids, groups


def _new_cache(cache, domain):
    values, source = _arrays(cache)
    required = {'protocol', 'stage', 'domain', 'arm_names', 'point', 'draws', 'evaluation_ids', 'groups',
        'bootstrap_seed', 'n_boot', 'full_group_sizes_json', 'target_values_in_cache', 'evaluation_marker_sha256'}
    if set(values) != required:
        raise ScientificStop('Unknown/missing cache arrays; raw target arrays may not enter a paired cache')
    for key, expected in (('protocol', NEW_CACHE_PROTOCOL), ('stage', '4.4'), ('domain', domain), ('target_values_in_cache', False)):
        if values[key].shape != () or values[key].item() != expected: raise ScientificStop('New cache domain/protocol/target identity differs')
    groups = SYNTHETIC_GROUPS if domain == 'synthetic' else REAL_GROUPS
    names = [f'{group}_seed{seed}' for group in groups for seed in SEEDS]
    _validate_arrays(values, names=names, ids_key='evaluation_ids', sizes_key='full_group_sizes_json', n_rows=N_ROWS[domain], seed=BOOT_SEEDS[domain])
    return values, source


def _old_cache(cache, condition):
    values, source = _arrays(cache)
    required = {'protocol', 'stage', 'arm_names', 'point', 'draws', 'test_ids', 'groups', 'bootstrap_seed',
        'n_boot', 'full_test_group_sizes_json', 'TEST_target_values_in_cache'}
    if not required <= set(values) or set(values) - required - {'condition', 'evaluation_marker_sha256'}:
        raise ScientificStop('Unknown/missing old cache arrays or accidental raw-target content')
    for key, expected in (('protocol', OLD_CACHE_PROTOCOL), ('stage', {'O': '4.1', 'L0': '4.2', 'L1': '4.3'}[condition]), ('TEST_target_values_in_cache', False)):
        if values[key].shape != () or values[key].item() != expected: raise ScientificStop('Old cache scientific identity differs')
    if 'condition' in values and (values['condition'].shape != () or values['condition'].item() != condition):
        raise ScientificStop('Old cache condition namespace differs')
    _validate_arrays(values, names=[f'{condition}_seed{seed}' for seed in SEEDS], ids_key='test_ids',
        sizes_key='full_test_group_sizes_json', n_rows=1000, seed=BOOT_SEEDS['synthetic'])
    return values, source


def _after_terminal(domain, marker, cache):
    marker = _marker(domain, marker); terminal = _read(_paths(domain)['terminal'])
    if (terminal.get('domain') != domain or terminal.get('target_accessor_calls') != 1
        or terminal.get('evaluation_marker', {}).get('sha256') != _sha(marker)
        or cache['evaluation_marker_sha256'].item() != _sha(marker)
        or terminal.get('evaluation_ids_sha256') != _json_hash(cache['evaluation_ids'].tolist())
        or terminal.get('groups_sha256') != _json_hash(cache['groups'].tolist())
        or terminal.get('full_group_sizes_sha256') != _json_hash(json.loads(cache['full_group_sizes_json'].item()))):
        raise ScientificStop('Pairing is forbidden before its matching domain terminal/cache commit')
    return marker, terminal


def _context(names, point, draws, ids, groups, seed):
    return {'names': names, 'index': {name: i for i, name in enumerate(names)}, 'point': point, 'draws': draws,
        'n': len(ids), 'n_groups': len(set(groups)), 'n_boot': N_BOOT, 'seed': seed}


def _contrasts(context, pairs):
    return {f'{a}_minus_{b}': {'A': a, 'B': b,
        'per_seed': {str(seed): frozen._paired(context, [f'{a}_seed{seed}'], [f'{b}_seed{seed}']) for seed in SEEDS},
        'arithmetic_seed_mean': frozen._paired(context, [f'{a}_seed{seed}' for seed in SEEDS], [f'{b}_seed{seed}' for seed in SEEDS])}
        for a, b in pairs}


def paired_controls_with_prior(caches, *, synthetic_marker):
    if not isinstance(caches, dict) or set(caches) != {'O', 'L0', 'L1', 'controls'}:
        raise ScientificStop('The three unchanged prior caches and one new controls cache are required')
    current, current_source = _new_cache(caches['controls'], 'synthetic')
    marker, terminal = _after_terminal('synthetic', synthetic_marker, current)
    values = []; sources = {}; names = []
    for condition in ('O', 'L0', 'L1'):
        old, source = _old_cache(caches[condition], condition)
        if (not np.array_equal(old['test_ids'], current['evaluation_ids']) or not np.array_equal(old['groups'], current['groups'])
            or json.loads(old['full_test_group_sizes_json'].item()) != json.loads(current['full_group_sizes_json'].item())):
            raise ScientificStop('Prior and new controls cache identities/group draws/weights cannot be paired')
        values.append(old); sources[condition] = source; names.extend(old['arm_names'].tolist())
    names.extend(current['arm_names'].tolist()); sources['controls'] = current_source
    context = _context(names, np.concatenate([v['point'] for v in values] + [current['point']], axis=1),
        np.concatenate([v['draws'] for v in values] + [current['draws']], axis=2),
        current['evaluation_ids'].tolist(), current['groups'].tolist(), BOOT_SEEDS['synthetic'])
    return {'schema_version': 'H4.4-existing-shared-draw-controls-pairs-v1', 'domain': 'synthetic',
        'target_accessor_calls': 0, 'new_bootstrap_resamples': 0, 'n_boot': N_BOOT, 'bootstrap_seed': BOOT_SEEDS['synthetic'],
        'comparisons': _contrasts(context, [('L1', group) for group in SYNTHETIC_GROUPS]),
        'cache_sources': sources, 'synthetic_marker_sha256': _sha(marker),
        'old_feature_pairs': readonly_existing_feature_pairs(),
        'method': 'Frozen _paired on existing aligned metric points/shared group draws only; no old truth read or bootstrap rerun'}


def paired_real_transfer(cache, *, real_marker):
    values, source = _new_cache(cache, 'real'); marker, terminal = _after_terminal('real', real_marker, values)
    context = _context(values['arm_names'].tolist(), values['point'], values['draws'],
        values['evaluation_ids'].tolist(), values['groups'].tolist(), BOOT_SEEDS['real'])
    return {'schema_version': 'H4.4-existing-shared-draw-transfer-pairs-v1', 'domain': 'real',
        'target_accessor_calls': 0, 'new_bootstrap_resamples': 0, 'n_boot': N_BOOT, 'bootstrap_seed': BOOT_SEEDS['real'],
        'comparisons': _contrasts(context, [('L1', 'Magpie_HGB')]),
        'cache_source': source, 'real_marker_sha256': _sha(marker),
        'method': 'Frozen _paired on already committed real-domain metric draws only; no HOLDOUT reread or new resampling'}


def _rounded(value):
    if type(value) not in (float, int) or not math.isfinite(value): raise ScientificStop('A registered old contrast is undefined')
    return format(Decimal.from_float(float(value)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN), '.6f')


def readonly_existing_feature_pairs():
    path = DATA / 'phase4_3/paired_comparisons.json'; saved = _read(path)
    if (saved.get('registered_L0_minus_O_MAE_check', {}).get('status') != 'PASS'
        or saved['registered_L0_minus_O_MAE_check'].get('six_decimal_values') != OLD_L0_O_REGISTERED
        or set(saved.get('comparisons', {})) != set(OLD_ROW_REFERENCES)):
        raise ScientificStop('The saved original three feature contrasts or registered reference differ')
    for name, expected in OLD_ROW_REFERENCES.items():
        metric = saved['comparisons'][name]['arithmetic_seed_mean']['row_weighted']['mae']
        if {'point': _rounded(metric['point']), 'ci95': [_rounded(x) for x in metric['ci95']]} != expected:
            raise ScientificStop('A saved old feature row-MAE contrast differs at six decimals: ' + name)
    return {'source_path': str(_read_path(path)), 'source_sha256': _sha(path),
        'status': 'PASS', 'target_accessor_calls': 0, 'new_bootstrap_resamples': 0, 'new_paired_subtractions': 0,
        'comparisons': deepcopy(saved['comparisons']), 'registered_reference': deepcopy(saved['registered_L0_minus_O_MAE_check']),
        'reuse_policy': 'Verbatim existing numeric results; six-decimal reference verification only, no rescoring'}
