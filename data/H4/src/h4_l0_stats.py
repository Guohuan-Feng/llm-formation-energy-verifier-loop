"""Stage4.2 L0 terminal-only statistics; original O results remain immutable.

Import/configuration never inspect TEST targets, fit or write files. A separate
stage4.2 marker permits one new L0 TEST1000 read shared by all three seeds.
Numerics are the frozen H3b group-bootstrap implementation, as in stage4.1.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / 'work/h-data/H4'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collections import Counter
import hashlib
import json
import math
import numpy as np
import h3b_stats as frozen

STAGE = '4.2'
CONDITION = 'L0'
BOOTSTRAP_SEED = 20261030
SCHEMA_VERSION = 'H4-stage4.2-L0-terminal-group-statistics-v1'
_USED_MARKERS: set[str] = set()
_LAST_PRIVATE_CACHE = None


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _prior_marker():
    path = (DATA / 'phase4_1/evaluation_started.json').resolve(strict=True)
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve()):
        raise ValueError('Prior stage4.1 marker must remain in the non-OneDrive H4 directory')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if (not isinstance(value, dict) or value.get('stage') != '4.1'
        or value.get('condition', 'O') != 'O' or value.get('scoring_pass') != 1):
        raise ValueError('An unchanged first O scoring marker is required before L0 terminal evaluation')
    return path


def configuration():
    """Describe the first separate L0 scoring pass without reading targets."""
    if frozen.N_BOOT != 2000 or tuple(frozen.METRICS) != ('mae', 'rmse', 'r2'):
        raise ValueError('Frozen H3b numerical statistics policy changed')
    prior = _prior_marker()
    paths = [ROOT / 'h3b_stats.py', Path(__file__).resolve()]
    return {
        'schema_version': SCHEMA_VERSION, 'stage': STAGE, 'condition': CONDITION,
        'seeds': [0, 1, 2], 'test_n': 1000,
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES),
        'grouping': 'original synthetic template identity',
        'group_weight': 'inverse count of this template in the complete TEST1000',
        'bootstrap_n': 2000, 'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap': 'within-domain template-group multinomial resampling; shared draws across all three seeds; percentile95 intervals',
        'seed_aggregation': 'arithmetic mean of seed-specific metrics and their paired bootstrap draws; never a prediction ensemble or best seed',
        'selection_uses_TEST': False,
        'terminal_policy': 'one separate L0 stage-global marker and one TEST1000 accessor call for its three seeds; never rescore O',
        'marker_contract': {'stage': STAGE, 'condition': CONDITION, 'scoring_pass': 1,
            'current_L0_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 1,
            'TEST_targets_previously_read_for_this_condition': False,
            'prior_stage4_1_marker_sha256': _sha(prior)},
        'prior_stage4_1_marker': {'path': str(prior), 'sha256': _sha(prior)},
        'O_results_recomputed': False,
        'new_O_L0_paired_comparison': False,
        'numerical_sources': {str(path.resolve()): _sha(path) for path in paths},
        'numpy_version': np.__version__,
    }


def _marker(path):
    path = Path(path).resolve(strict=True)
    expected = (DATA / 'phase4_2/evaluation_started.json').resolve(strict=True)
    if ('onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve())
        or path != expected):
        raise ValueError('L0 terminal scoring requires the exclusive phase4_2/evaluation_started.json marker')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    expected_fields = {'stage': STAGE, 'condition': CONDITION, 'scoring_pass': 1,
        'current_L0_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 1,
        'TEST_targets_previously_read_for_this_condition': False}
    if not isinstance(value, dict) or any(
            type(value.get(key)) is not type(expected_value) or value.get(key) != expected_value
            for key, expected_value in expected_fields.items()):
        raise ValueError('L0 marker must disclose prior O scoring and authorize exactly its first new L0 pass')
    prior = _prior_marker()
    if value.get('prior_stage4_1_marker_sha256') != _sha(prior):
        raise ValueError('The prior stage4.1 scoring marker changed or is not bound to this L0 pass')
    return path, prior


def _prediction_map(arm):
    rows = arm.get('prediction_rows', arm.get('predictions', []))
    if isinstance(rows, dict):
        pairs = list(rows.items())
    elif isinstance(rows, list):
        pairs = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError('Malformed L0 TEST prediction row')
            pairs.append((row['id'], row.get('prediction_eV_per_atom', row.get('prediction'))))
    else:
        raise ValueError('L0 TEST predictions must be rows or an id mapping')
    result = {}
    for rid, value in pairs:
        if not isinstance(rid, str) or rid in result:
            raise ValueError('Duplicate or malformed L0 TEST prediction identity')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Each L0 TEST prediction must be one finite numeric scalar')
        result[rid] = float(value)
    return result


def _normalise(arms, test_ids):
    if isinstance(arms, dict):
        arms = arms.get('arms', list(arms.values()))
    if not isinstance(arms, list) or len(arms) != 3:
        raise ValueError('Stage4.2 requires exactly three final L0 seed arms')
    expected = set(test_ids)
    by_seed = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError('Malformed L0 model arm')
        seed = arm.get('seed')
        if type(seed) is not int or seed not in (0, 1, 2) or seed in by_seed:
            raise ValueError('Stage4.2 requires each of seed0/1/2 exactly once')
        if arm.get('condition') != CONDITION:
            raise ValueError('Only L0 arms may be terminally scored in stage4.2; O must not be rescored')
        if arm.get('status', 'complete') not in ('complete', 'ok', 'reused_complete'):
            raise ValueError('All L0 arms must finish before the sole new L0 target read')
        values = _prediction_map(arm)
        if len(values) != 1000 or set(values) != expected:
            raise ValueError('Every L0 seed requires all fixed1000 TEST predictions; no subset selection')
        name = arm.get('arm_id', f'L0_seed{seed}')
        if not isinstance(name, str) or not name.startswith('L0_'):
            raise ValueError('L0 arm identities must use the L0 namespace')
        by_seed[seed] = {'arm_id': name, 'condition': CONDITION, 'seed': seed,
            'method': arm.get('method', 'HGB_Magpie_LLM'),
            'status': arm.get('status', 'complete'), 'prediction_map': values}
    if set(by_seed) != {0, 1, 2} or len({row['arm_id'] for row in by_seed.values()}) != 3:
        raise ValueError('L0 seed or arm identity coverage changed')
    return [by_seed[seed] for seed in (0, 1, 2)]


def evaluate(arms, bundle, *, evaluation_marker):
    """Score only L0 after complete predictions and its new single-use gate.

    One h4_l0_data.score_truth call supplies all three seeds. No old O target
    accessor or evaluation is invoked. No fit, selection, provider or write is
    performed here; private paired draws are only for later authorized use.
    """
    global _LAST_PRIVATE_CACHE
    policy = configuration()
    marker, prior = _marker(evaluation_marker)
    ids = list(bundle['test_ids'])
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('L0 must retain the complete frozen TEST1000')
    if 'pool_ids' in bundle and set(ids) & set(bundle['pool_ids']):
        raise ValueError('POOL/TEST identity overlap is forbidden')
    if not set(ids) <= set(bundle['groups']):
        raise ValueError('Complete TEST template-group mapping is required')
    normal = _normalise(arms, ids)
    marker_key = str(marker)
    if marker_key in _USED_MARKERS:
        raise RuntimeError('This L0 terminal marker has already consumed its TEST target read')
    _USED_MARKERS.add(marker_key)
    _LAST_PRIVATE_CACHE = None
    import h4_l0_data
    truth, provenance = h4_l0_data.score_truth(ids, evaluation_marker=marker, bundle=bundle)
    if not isinstance(truth, dict) or set(truth) != set(ids):
        raise ValueError('Terminal L0 TEST1000 truth coverage changed')
    if any(type(truth[rid]) not in (int, float) or not math.isfinite(truth[rid]) for rid in ids):
        raise ValueError('Terminal TEST targets must be finite scalars')
    groups = [str(bundle['groups'][rid]) for rid in ids]
    full_sizes = dict(Counter(groups))
    predictions = {arm['arm_id']: [arm['prediction_map'][rid] for rid in ids] for arm in normal}
    context = frozen._bootstrap_context([truth[rid] for rid in ids], predictions,
        groups, full_sizes, BOOTSTRAP_SEED)
    per_seed = {str(arm['seed']): {
        'arm_id': arm['arm_id'], 'condition': CONDITION, 'seed': arm['seed'],
        'method': arm['method'], 'status': arm['status'], 'n_test': 1000,
        'scores': frozen._score(context, [arm['arm_id']])} for arm in normal}
    names = [arm['arm_id'] for arm in normal]
    _LAST_PRIVATE_CACHE = {
        'protocol': np.asarray('H4-private-paired-bootstrap-draws-v1'),
        'stage': np.asarray(STAGE), 'condition': np.asarray(CONDITION),
        'arm_names': np.asarray(names), 'point': context['point'].copy(),
        'draws': context['draws'].copy(), 'test_ids': np.asarray(ids),
        'groups': np.asarray(groups), 'bootstrap_seed': np.asarray(BOOTSTRAP_SEED),
        'n_boot': np.asarray(2000),
        'full_test_group_sizes_json': np.asarray(json.dumps(full_sizes, sort_keys=True, separators=(',', ':'))),
        'TEST_target_values_in_cache': np.asarray(False),
    }
    return {
        'schema_version': SCHEMA_VERSION, 'stage': STAGE, 'condition': CONDITION,
        'unit': 'eV/atom', 'n_test': 1000, 'n_groups': context['n_groups'],
        'n_boot': 2000, 'bootstrap_seed': BOOTSTRAP_SEED,
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES),
        'grouping': policy['grouping'], 'bootstrap': policy['bootstrap'],
        'fixed_test_ids_sha256': _json_hash(ids), 'template_groups_sha256': _json_hash(groups),
        'full_test_group_sizes_sha256': _json_hash(full_sizes), 'target_source': provenance,
        'TEST_target_accessor_calls': 1, 'TEST_targets_returned_to_label_free_bundle': False,
        'prior_H4_TEST_scoring_passes': 1, 'current_L0_scoring_pass': 1,
        'H4_TEST_scoring_passes_after_this_stage': 2,
        'prior_stage4_1_marker': {'path': str(prior), 'sha256': _sha(prior)},
        'evaluation_marker': {'path': str(marker), 'sha256': _sha(marker), 'scoring_pass': 1},
        'model_or_selection_calls_in_statistics': 0, 'O_results_recomputed': False,
        'per_seed': per_seed, 'arithmetic_seed_mean': frozen._score(context, names),
        'paired_other_conditions': {}, 'numerical_sources': policy['numerical_sources'],
        'interpretation_limits': [
            'This is the first L0 TEST1000 scoring pass; O had a distinct prior pass and is not rescored.',
            'L0 TEST1000 targets are accessed only after candidate selection and final fullTRAIN fitting are complete.',
            'Three-seed aggregation is the arithmetic mean of metrics, not a prediction ensemble or best-seed result.',
            'The same template-group draws and complete TEST group sizes are used for both weightings and every L0 seed.',
            'L0 prompts include this arm\'s internal validation feedback by design; internal validation is not an untouched-label holdout.',
            'Validation members grow as labels are revealed under a frozen template assignment; round metrics need not have identical members.',
            'No O-L0 comparison is calculated in stage4.2; private loss-derived draws are reserved for later authorized comparisons.',
        ],
    }


def private_pairing_cache():
    """Copy L0-only loss-derived draws, without raw TEST targets or scoring."""
    if _LAST_PRIVATE_CACHE is None:
        raise RuntimeError('No successful L0 terminal evaluation cache is available')
    return {key: value.copy() for key, value in _LAST_PRIVATE_CACHE.items()}
