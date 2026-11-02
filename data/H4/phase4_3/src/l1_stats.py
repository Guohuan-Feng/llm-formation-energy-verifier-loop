"""One L1 terminal evaluation and subsequently authorized cached-draw pairs.

Import/configuration do not read TEST targets, fit, execute features or write.
The new L1 truth accessor is called once, after its separate exclusive marker.
Pair comparisons use existing shared draws only after L1 terminal_results exists;
they never call any truth accessor or generate a new bootstrap resample.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / 'work/h-data/H4'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import numpy as np
import h3b_stats as frozen

STAGE = '4.3'
CONDITION = 'L1'
BOOTSTRAP_SEED = 20261030
N_BOOT = 2000
SCHEMA_VERSION = 'H4-stage4.3-L1-terminal-group-statistics-v1'
CACHE_PROTOCOL = 'H4-private-paired-bootstrap-draws-v1'
_USED_MARKERS = set()
_LAST_PRIVATE_CACHE = None
_L0_O_REFERENCE = {
    'row_weighted': {'point': '-0.003126', 'ci95': ['-0.010755', '0.004133']},
    'group_weighted': {'point': '-0.002522', 'ci95': ['-0.010568', '0.005686']},
}


class ScientificStop(RuntimeError):
    """An immutable scientific identity or registered pair reference changed."""


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _read(path):
    path = Path(path).resolve(strict=True)
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve()):
        raise ValueError('L1 statistical metadata must remain within the non-OneDrive H4 directory')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _prior_markers():
    result = {}
    for phase, condition in (('4.1', 'O'), ('4.2', 'L0')):
        path = DATA / ('phase' + phase.replace('.', '_')) / 'evaluation_started.json'
        value = _read(path)
        if (not isinstance(value, dict) or value.get('stage') != phase
            or value.get('condition', 'O') != condition or value.get('scoring_pass') != 1):
            raise ValueError('The completed prior O/L0 scoring markers are required')
        result[phase] = {'path': str(path.resolve()), 'sha256': _sha(path)}
    return result


def configuration():
    if (frozen.N_BOOT != N_BOOT or tuple(frozen.METRICS) != ('mae', 'rmse', 'r2')
        or tuple(frozen.MODES) != ('row_weighted', 'group_weighted')):
        raise ValueError('The frozen H3b numerical statistics policy changed')
    prior = _prior_markers()
    return {
        'schema_version': SCHEMA_VERSION, 'stage': STAGE, 'condition': CONDITION,
        'seeds': [0, 1, 2], 'test_n': 1000, 'metrics': list(frozen.METRICS),
        'weightings': list(frozen.MODES), 'grouping': 'original synthetic template identity',
        'group_weight': 'inverse count of this template in complete TEST1000',
        'bootstrap_n': N_BOOT, 'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap': 'template-group multinomial resampling; shared draws across seeds and stages; percentile95 intervals',
        'seed_aggregation': 'arithmetic mean of seed-specific metrics and draws; never a prediction ensemble',
        'selection_uses_TEST': False,
        'marker_contract': {'stage': STAGE, 'condition': CONDITION, 'scoring_pass': 1,
            'current_L1_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 2,
            'TEST_targets_previously_read_for_this_condition': False,
            'prior_stage4_1_marker_sha256': prior['4.1']['sha256'],
            'prior_stage4_2_marker_sha256': prior['4.2']['sha256']},
        'prior_stage_markers': prior,
        'terminal_policy': 'one L1 accessor read supplies all three L1 seeds; O/L0 are never rescored',
        'pairing_cache_protocol': CACHE_PROTOCOL,
        'pairing_policy': 'Only after L1 terminal_results exists, subtract existing shared bootstrap draws; no TEST read or fresh resampling',
        'paired_comparisons': ['L1_minus_L0', 'L1_minus_O', 'L0_minus_O'],
        'registered_L0_minus_O_MAE_to_six_decimal_places': _L0_O_REFERENCE,
        'pair_reference_rounding': 'Decimal.from_float quantize(0.000001), ROUND_HALF_EVEN',
        'numerical_sources': {str((ROOT / 'h3b_stats.py').resolve()): _sha(ROOT / 'h3b_stats.py'),
                              str(Path(__file__).resolve()): _sha(__file__)},
        'numpy_version': np.__version__,
    }


def _marker(path):
    path = Path(path).resolve(strict=True)
    expected = (DATA / 'phase4_3/evaluation_started.json').resolve(strict=True)
    if path != expected or 'onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve()):
        raise ValueError('L1 requires the exclusive phase4_3/evaluation_started.json gate')
    value = _read(path)
    fields = configuration()['marker_contract']
    if not isinstance(value, dict) or any(
        type(value.get(key)) is not type(wanted) or value.get(key) != wanted
        for key, wanted in fields.items()):
        raise ValueError('L1 marker must bind both prior passes and exactly its first new L1 pass')
    return path


def _prediction_map(arm):
    rows = arm.get('prediction_rows', arm.get('predictions', []))
    if isinstance(rows, dict):
        pairs = list(rows.items())
    elif isinstance(rows, list):
        pairs = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError('Malformed L1 TEST prediction row')
            pairs.append((row['id'], row.get('prediction_eV_per_atom', row.get('prediction'))))
    else:
        raise ValueError('L1 predictions must be rows or an ID mapping')
    result = {}
    for rid, value in pairs:
        if not isinstance(rid, str) or rid in result:
            raise ValueError('Duplicate or malformed L1 TEST prediction identity')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Every L1 TEST prediction must be a finite numeric scalar')
        result[rid] = float(value)
    return result


def _normalise(arms, test_ids):
    if isinstance(arms, dict):
        arms = arms.get('arms', list(arms.values()))
    if not isinstance(arms, list) or len(arms) != 3:
        raise ValueError('L1 terminal scoring requires exactly three final seed arms')
    expected = set(test_ids)
    by_seed = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError('Malformed L1 arm')
        seed = arm.get('seed')
        if type(seed) is not int or seed not in (0, 1, 2) or seed in by_seed:
            raise ValueError('L1 requires seed0/1/2 exactly once')
        if arm.get('condition') != CONDITION or arm.get('arm_id', f'L1_seed{seed}') != f'L1_seed{seed}':
            raise ValueError('Only canonical L1 seed arms may be scored; O/L0 must not be rescored')
        if arm.get('status', 'complete') not in ('complete', 'ok', 'reused_complete'):
            raise ValueError('All three L1 arms must finish before the sole target read')
        predictions = _prediction_map(arm)
        if len(predictions) != 1000 or set(predictions) != expected:
            raise ValueError('Every L1 seed requires all fixed1000 TEST predictions')
        by_seed[seed] = {'arm_id': f'L1_seed{seed}', 'seed': seed, 'condition': CONDITION,
            'method': arm.get('method', 'HGB_Magpie132_LLM_memory_features'),
            'status': arm.get('status', 'complete'), 'prediction_map': predictions}
    return [by_seed[seed] for seed in (0, 1, 2)]


def evaluate(arms, bundle, *, evaluation_marker):
    global _LAST_PRIVATE_CACHE
    policy = configuration()
    marker = _marker(evaluation_marker)
    ids = list(bundle['test_ids'])
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('L1 must retain complete fixed TEST1000')
    if set(ids) & set(bundle.get('pool_ids', [])) or not set(ids) <= set(bundle['groups']):
        raise ValueError('Invalid TEST/POOL separation or incomplete template mapping')
    normal = _normalise(arms, ids)
    key = str(marker)
    if key in _USED_MARKERS:
        raise ScientificStop('This L1 marker already consumed its TEST target read')
    _USED_MARKERS.add(key)
    _LAST_PRIVATE_CACHE = None
    import l1_data
    truth, provenance = l1_data.score_truth(ids, evaluation_marker=marker, bundle=bundle)
    if (not isinstance(truth, dict) or set(truth) != set(ids)
        or any(type(truth[rid]) not in (int, float) or not math.isfinite(truth[rid]) for rid in ids)):
        raise ValueError('L1 TEST truth must be aligned, finite and complete')
    groups = [str(bundle['groups'][rid]) for rid in ids]
    sizes = dict(Counter(groups))
    predictions = {arm['arm_id']: [arm['prediction_map'][rid] for rid in ids] for arm in normal}
    context = frozen._bootstrap_context([truth[rid] for rid in ids], predictions, groups, sizes, BOOTSTRAP_SEED)
    names = [arm['arm_id'] for arm in normal]
    per_seed = {str(arm['seed']): {'arm_id': arm['arm_id'], 'seed': arm['seed'],
        'condition': CONDITION, 'method': arm['method'], 'status': arm['status'], 'n_test': 1000,
        'scores': frozen._score(context, [arm['arm_id']])} for arm in normal}
    _LAST_PRIVATE_CACHE = {
        'protocol': np.asarray(CACHE_PROTOCOL), 'stage': np.asarray(STAGE),
        'condition': np.asarray(CONDITION), 'arm_names': np.asarray(names),
        'point': context['point'].copy(), 'draws': context['draws'].copy(),
        'test_ids': np.asarray(ids), 'groups': np.asarray(groups),
        'bootstrap_seed': np.asarray(BOOTSTRAP_SEED), 'n_boot': np.asarray(N_BOOT),
        'full_test_group_sizes_json': np.asarray(json.dumps(sizes, sort_keys=True, separators=(',', ':'))),
        'TEST_target_values_in_cache': np.asarray(False),
        'evaluation_marker_sha256': np.asarray(_sha(marker)),
    }
    return {
        'schema_version': SCHEMA_VERSION, 'stage': STAGE, 'condition': CONDITION,
        'unit': 'eV/atom', 'n_test': 1000, 'n_groups': context['n_groups'],
        'n_boot': N_BOOT, 'bootstrap_seed': BOOTSTRAP_SEED,
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES),
        'grouping': policy['grouping'], 'bootstrap': policy['bootstrap'],
        'fixed_test_ids_sha256': _json_hash(ids), 'template_groups_sha256': _json_hash(groups),
        'full_test_group_sizes_sha256': _json_hash(sizes), 'target_source': provenance,
        'TEST_target_accessor_calls': 1, 'TEST_targets_returned_to_label_free_bundle': False,
        'prior_H4_TEST_scoring_passes': 2, 'current_L1_scoring_pass': 1,
        'H4_TEST_scoring_passes_after_this_stage': 3,
        'prior_stage_markers': policy['prior_stage_markers'],
        'evaluation_marker': {'path': str(marker), 'sha256': _sha(marker), 'scoring_pass': 1},
        'model_or_selection_calls_in_statistics': 0, 'prior_O_L0_results_recomputed': False,
        'per_seed': per_seed, 'arithmetic_seed_mean': frozen._score(context, names),
        'paired_other_conditions': {}, 'numerical_sources': policy['numerical_sources'],
        'interpretation_limits': [
            'L1 has one independent terminal TEST1000 read for all three seeds; prior O/L0 results are not rescored.',
            'Three-seed results are arithmetic means of metrics, not a prediction ensemble.',
            'Internal validation feedback/memory uses only this arm\'s revealed POOL labels; it is not an untouched-label holdout.',
            'Validation membership grows under fixed template roles; saved round MAEs need not use identical members.',
            'L1-L0 compares the full feedback, selection and memory policies, not a pure causal effect of notes alone.',
        ],
    }


def private_pairing_cache():
    if _LAST_PRIVATE_CACHE is None:
        raise ScientificStop('No completed L1 terminal draw cache exists')
    return {key: value.copy() for key, value in _LAST_PRIVATE_CACHE.items()}


def _load_cache(cache, condition):
    if isinstance(cache, (str, Path)):
        path = Path(cache).resolve(strict=True)
        if 'onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve()):
            raise ScientificStop('Paired caches must resolve within local H4 storage')
        with np.load(path, allow_pickle=False) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        source = {'path': str(path), 'sha256': _sha(path)}
    elif isinstance(cache, dict):
        values = {key: np.asarray(value).copy() for key, value in cache.items()}
        source = {'path': None, 'source': 'explicit in-memory loss-derived cache'}
    else:
        raise ScientificStop('A paired cache must be an NPZ path or array mapping')
    required = {'protocol', 'stage', 'arm_names', 'point', 'draws', 'test_ids', 'groups',
        'bootstrap_seed', 'n_boot', 'full_test_group_sizes_json', 'TEST_target_values_in_cache'}
    if not required <= set(values) or set(values) - required - {'condition', 'evaluation_marker_sha256'}:
        raise ScientificStop('Unknown/missing paired cache fields; raw targets must not enter the cache')
    stage = {'O': '4.1', 'L0': '4.2', 'L1': '4.3'}[condition]
    for key, expected in (('protocol', CACHE_PROTOCOL), ('stage', stage),
                          ('bootstrap_seed', BOOTSTRAP_SEED), ('n_boot', N_BOOT),
                          ('TEST_target_values_in_cache', False)):
        if values[key].shape != () or values[key].item() != expected:
            raise ScientificStop('Paired cache protocol/stage/bootstrap identity mismatch: ' + key)
    if 'condition' in values and (values['condition'].shape != () or values['condition'].item() != condition):
        raise ScientificStop('Paired cache condition mismatch')
    names = values['arm_names'].tolist()
    ids = values['test_ids'].tolist()
    groups = values['groups'].tolist()
    if names != [f'{condition}_seed{s}' for s in (0, 1, 2)]:
        raise ScientificStop('Paired cache seed order or arm namespace changed')
    if (len(ids) != 1000 or len(set(ids)) != 1000 or len(groups) != 1000
        or any(type(x) is not str for x in ids + groups)):
        raise ScientificStop('Paired caches require fixed1000 ordered identities and groups')
    if json.loads(values['full_test_group_sizes_json'].item()) != dict(Counter(groups)):
        raise ScientificStop('Paired cache complete TEST group weights changed')
    for key, shape in (('point', (2, 3, 3)), ('draws', (N_BOOT, 2, 3, 3))):
        array = values[key]
        if (array.shape != shape or array.dtype != np.dtype('float64') or np.isinf(array).any()
            or not np.isfinite(array[..., :2]).all() or (array[..., :2] < 0).any()):
            raise ScientificStop('Paired numerical cache shape/dtype/finite identity changed')
    return values, source


def inspect_cache(cache, condition):
    """Read only cache metadata/format. No subtraction, CI or target access."""
    values, source = _load_cache(cache, condition)
    return {**source, 'condition': condition, 'stage': values['stage'].item(),
        'arm_names': values['arm_names'].tolist(), 'n_test': len(values['test_ids']),
        'n_groups': len(set(values['groups'].tolist())), 'point_shape': list(values['point'].shape),
        'draws_shape': list(values['draws'].shape), 'bootstrap_seed': BOOTSTRAP_SEED,
        'n_boot': N_BOOT, 'raw_TEST_targets_present': False}


def _after_L1_terminal(marker):
    marker = _marker(marker)
    terminal_path = DATA / 'phase4_3/terminal_results.json'
    if not terminal_path.is_file():
        raise ScientificStop('Paired differences are forbidden before L1 terminal_results is committed')
    result = _read(terminal_path)
    if (result.get('stage') != STAGE or result.get('condition') != CONDITION
        or result.get('TEST_target_accessor_calls') != 1 or result.get('current_L1_scoring_pass') != 1
        or result.get('prior_H4_TEST_scoring_passes') != 2
        or result.get('evaluation_marker', {}).get('sha256') != _sha(marker)):
        raise ScientificStop('Committed L1 terminal identity/pass disclosure changed')
    return marker, result


def _rounded(value):
    if value is None or not math.isfinite(value):
        raise ScientificStop('Registered L0-O comparison is undefined')
    return format(Decimal.from_float(float(value)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN), '.6f')


def _check_L0_O_reference(scores):
    checked = {}
    for mode, expected in _L0_O_REFERENCE.items():
        mae = scores[mode]['mae']
        actual = {'point': _rounded(mae['point']), 'ci95': [_rounded(x) for x in mae['ci95']]}
        if actual != expected:
            raise ScientificStop('L0-O registered MAE/CI differs at six decimal places: ' + mode)
        checked[mode] = actual
    return checked


def paired_comparisons(caches, *, l1_evaluation_marker):
    """After L1 terminal, subtract existing stored group draws only.

    ``caches`` maps exactly O/L0/L1 to their private NPZ paths or explicit array
    mappings. No O/L0 difference is formed until the completed L1 gate passes.
    All original bootstrap axes, template groups, weights and seeds must agree.
    """
    marker, terminal = _after_L1_terminal(l1_evaluation_marker)
    if not isinstance(caches, dict) or set(caches) != {'O', 'L0', 'L1'}:
        raise ScientificStop('Exactly O, L0 and L1 caches are required')
    loaded = {}; sources = {}
    for condition in ('O', 'L0', 'L1'):
        loaded[condition], sources[condition] = _load_cache(caches[condition], condition)
    reference = loaded['L1']
    for condition in ('O', 'L0'):
        for key in ('test_ids', 'groups', 'bootstrap_seed', 'n_boot'):
            if not np.array_equal(loaded[condition][key], reference[key]):
                raise ScientificStop('Paired fixed IDs/groups/draw identities differ: ' + condition + ':' + key)
        if json.loads(loaded[condition]['full_test_group_sizes_json'].item()) != json.loads(reference['full_test_group_sizes_json'].item()):
            raise ScientificStop('Paired full-TEST group weights differ')
    if (reference.get('evaluation_marker_sha256', np.asarray('')).item() != _sha(marker)
        or terminal['fixed_test_ids_sha256'] != _json_hash(reference['test_ids'].tolist())
        or terminal['template_groups_sha256'] != _json_hash(reference['groups'].tolist())
        or terminal['full_test_group_sizes_sha256'] != _json_hash(json.loads(reference['full_test_group_sizes_json'].item()))):
        raise ScientificStop('L1 draw cache does not match its committed terminal identities')
    for seed in (0, 1, 2):
        for mode_index, mode in enumerate(frozen.MODES):
            for metric_index, metric in enumerate(frozen.METRICS):
                point = reference['point'][mode_index, seed, metric_index]
                recorded = terminal['per_seed'][str(seed)]['scores'][mode][metric]['point']
                if not ((recorded is None and np.isnan(point)) or recorded == float(point)):
                    raise ScientificStop('L1 cache metric points differ from terminal results')
    names = [name for condition in ('O', 'L0', 'L1') for name in loaded[condition]['arm_names'].tolist()]
    context = {'names': names, 'index': {name: i for i, name in enumerate(names)},
        'point': np.concatenate([loaded[c]['point'] for c in ('O', 'L0', 'L1')], axis=1),
        'draws': np.concatenate([loaded[c]['draws'] for c in ('O', 'L0', 'L1')], axis=2),
        'n': 1000, 'n_groups': len(set(reference['groups'].tolist())), 'n_boot': N_BOOT, 'seed': BOOTSTRAP_SEED}
    comparisons = {}
    for a, b in (('L1', 'L0'), ('L1', 'O'), ('L0', 'O')):
        comparisons[a + '_minus_' + b] = {
            'per_seed': {str(seed): frozen._paired(context, [f'{a}_seed{seed}'], [f'{b}_seed{seed}']) for seed in (0, 1, 2)},
            'arithmetic_seed_mean': frozen._paired(context,
                [f'{a}_seed{seed}' for seed in (0, 1, 2)], [f'{b}_seed{seed}' for seed in (0, 1, 2)])}
    checked = _check_L0_O_reference(comparisons['L0_minus_O']['arithmetic_seed_mean'])
    return {'schema_version': 'H4-existing-group-draw-paired-comparisons-v1',
        'stage': STAGE, 'condition': CONDITION, 'n_test': 1000, 'n_groups': context['n_groups'],
        'n_boot': N_BOOT, 'bootstrap_seed': BOOTSTRAP_SEED, 'metrics': list(frozen.METRICS),
        'weightings': list(frozen.MODES), 'comparisons': comparisons,
        'registered_L0_minus_O_MAE_check': {'status': 'PASS', 'six_decimal_values': checked},
        'cache_sources': sources, 'L1_terminal_marker_sha256': _sha(marker),
        'TEST_target_accessor_calls': 0, 'new_bootstrap_resamples': 0,
        'method': 'Frozen _paired on existing aligned per-seed point metrics and shared template-group bootstrap draws',
        'interpretation_limit': 'L1-L0 is the full feedback/selection/memory policy difference, not a pure notes-only causal effect.'}
