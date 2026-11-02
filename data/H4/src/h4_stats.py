"""H4 stage4.1 terminal-only statistics, using frozen H3b numerical code.

Import/configuration do not load evaluation targets, fit models or write files.
One explicit H4 stage4.1 marker permits the original TEST1000 target read once.
Private paired bootstrap draws can be persisted by the coordinator for stage4.4.
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

BOOTSTRAP_SEED = 20261030
SCHEMA_VERSION = 'H4-stage4.1-O-terminal-group-statistics-v1'
STAGE = '4.1'
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


def configuration():
    """Freeze the complete stage4.1 scoring policy before scientific calls."""
    if frozen.N_BOOT != 2000 or tuple(frozen.METRICS) != ('mae', 'rmse', 'r2'):
        raise ValueError('Frozen H3b numerical statistics policy changed')
    paths = [ROOT / 'h3b_stats.py', Path(__file__).resolve()]
    return {
        'schema_version': SCHEMA_VERSION,
        'stage': STAGE,
        'condition': 'O',
        'seeds': [0, 1, 2],
        'test_n': 1000,
        'metrics': list(frozen.METRICS),
        'weightings': list(frozen.MODES),
        'grouping': 'original synthetic template identity',
        'group_weight': 'inverse count of this template in the complete TEST1000',
        'bootstrap_n': 2000,
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap': 'within-domain template-group multinomial resampling; shared draws across all three seeds; percentile95 intervals',
        'seed_aggregation': 'arithmetic mean of seed-specific metrics and their paired bootstrap draws; never a prediction ensemble or best seed',
        'selection_uses_TEST': False,
        'terminal_policy': 'one stage-global marker; one TEST1000 accessor call supplies all three seeds; no partial-subset scoring',
        'marker_contract': {'stage': '4.1', 'scoring_pass': 1, 'TEST_targets_previously_read': False},
        'numerical_sources': {str(path.resolve()): _sha(path) for path in paths},
        'numpy_version': np.__version__,
    }


def _marker(path):
    path = Path(path).resolve(strict=True)
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(DATA.resolve()):
        raise ValueError('Terminal marker must resolve within the non-OneDrive H4 private directory')
    if path.name != 'evaluation_started.json':
        raise ValueError('Stage4.1 requires its exclusive evaluation_started.json marker')
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if (not isinstance(value, dict) or value.get('stage') != STAGE
        or value.get('scoring_pass') != 1 or value.get('TEST_targets_previously_read') is not False):
        raise ValueError('Terminal marker must explicitly authorize the first stage4.1 TEST scoring pass')
    if value.get('condition', 'O') != 'O':
        raise ValueError('Stage4.1 terminal marker condition must be O')
    return path


def _prediction_map(arm):
    rows = arm.get('prediction_rows', arm.get('predictions', []))
    if isinstance(rows, dict):
        pairs = list(rows.items())
    elif isinstance(rows, list):
        pairs = []
        for row in rows:
            if not isinstance(row, dict) or 'id' not in row:
                raise ValueError('Malformed H4 TEST prediction row')
            value = row.get('prediction_eV_per_atom', row.get('prediction'))
            pairs.append((row['id'], value))
    else:
        raise ValueError('H4 TEST predictions must be rows or an id mapping')
    result = {}
    for rid, value in pairs:
        if rid in result:
            raise ValueError('Duplicate H4 TEST prediction identity')
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Each H4 TEST prediction must be one finite numeric scalar')
        result[rid] = float(value)
    return result


def _normalise(arms, test_ids):
    if isinstance(arms, dict):
        arms = arms.get('arms', list(arms.values()))
    if not isinstance(arms, list) or len(arms) != 3:
        raise ValueError('Stage4.1 requires exactly three final O seed arms')
    expected = set(test_ids)
    by_seed = {}
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError('Malformed H4 model arm')
        seed = arm.get('seed')
        if type(seed) is not int or seed not in (0, 1, 2) or seed in by_seed:
            raise ValueError('Stage4.1 requires each of seed0/1/2 exactly once')
        if arm.get('condition', 'O') != 'O':
            raise ValueError('Only O arms may be terminally scored in stage4.1')
        if arm.get('status', 'complete') not in ('complete', 'ok', 'reused_complete'):
            raise ValueError('All O arms must finish before the sole terminal target read')
        values = _prediction_map(arm)
        if len(values) != 1000 or set(values) != expected:
            raise ValueError('Stage4.1 requires all fixed1000 TEST predictions for every seed; no subset selection')
        by_seed[seed] = {
            'arm_id': arm.get('arm_id', f'O_seed{seed}'),
            'condition': 'O', 'seed': seed,
            'method': arm.get('method', 'HGB_Magpie_LLM'),
            'status': arm.get('status', 'complete'),
            'prediction_map': values,
        }
    if set(by_seed) != {0, 1, 2} or len({row['arm_id'] for row in by_seed.values()}) != 3:
        raise ValueError('Stage4.1 seed or arm identity coverage changed')
    return [by_seed[seed] for seed in (0, 1, 2)]


def evaluate(arms, bundle, *, evaluation_marker):
    """Score all three O arms after one authorized TEST1000 read.

    ``bundle`` is label-free, with frozen ``test_ids`` and template ``groups``.
    ``h4_data.score_truth`` enforces the same marker and fixed membership gate.
    Prediction coverage is checked before the marker is consumed or truth read.
    This function does not write files or perform any fit/selection/provider call.
    """
    global _LAST_PRIVATE_CACHE
    _LAST_PRIVATE_CACHE = None
    policy = configuration()
    marker = _marker(evaluation_marker)
    ids = list(bundle['test_ids'])
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError('H4 stage4.1 must retain the complete frozen TEST1000')
    if 'pool_ids' in bundle and set(ids) & set(bundle['pool_ids']):
        raise ValueError('POOL/TEST identity overlap is forbidden')
    if not set(ids) <= set(bundle['groups']):
        raise ValueError('Complete TEST template-group mapping is required')
    normal = _normalise(arms, ids)
    marker_key = str(marker)
    if marker_key in _USED_MARKERS:
        raise RuntimeError('This terminal marker has already consumed its TEST target read')
    _USED_MARKERS.add(marker_key)
    import h4_data
    truth, provenance = h4_data.score_truth(ids, evaluation_marker=marker, bundle=bundle)
    if not isinstance(truth, dict) or set(truth) != set(ids):
        raise ValueError('Terminal TEST1000 truth coverage changed')
    if any(type(truth[rid]) not in (int, float) or not math.isfinite(truth[rid]) for rid in ids):
        raise ValueError('Terminal TEST targets must be finite scalars')
    groups = [str(bundle['groups'][rid]) for rid in ids]
    full_sizes = dict(Counter(groups))
    predictions = {arm['arm_id']: [arm['prediction_map'][rid] for rid in ids] for arm in normal}
    context = frozen._bootstrap_context([truth[rid] for rid in ids], predictions,
                                         groups, full_sizes, BOOTSTRAP_SEED)
    per_seed = {}
    for arm in normal:
        per_seed[str(arm['seed'])] = {
            'arm_id': arm['arm_id'], 'condition': 'O', 'seed': arm['seed'],
            'method': arm['method'], 'status': arm['status'], 'n_test': 1000,
            'scores': frozen._score(context, [arm['arm_id']]),
        }
    ordered_names = [arm['arm_id'] for arm in normal]
    _LAST_PRIVATE_CACHE = {
        'protocol': np.asarray('H4-private-paired-bootstrap-draws-v1'),
        'stage': np.asarray(STAGE),
        'arm_names': np.asarray(ordered_names),
        'point': context['point'].copy(),
        'draws': context['draws'].copy(),
        'test_ids': np.asarray(ids),
        'groups': np.asarray(groups),
        'bootstrap_seed': np.asarray(BOOTSTRAP_SEED),
        'n_boot': np.asarray(2000),
        'full_test_group_sizes_json': np.asarray(json.dumps(full_sizes, sort_keys=True, separators=(',', ':'))),
        'TEST_target_values_in_cache': np.asarray(False),
    }
    return {
        'schema_version': SCHEMA_VERSION,
        'stage': STAGE, 'condition': 'O', 'unit': 'eV/atom',
        'n_test': 1000, 'n_groups': context['n_groups'],
        'n_boot': 2000, 'bootstrap_seed': BOOTSTRAP_SEED,
        'metrics': list(frozen.METRICS), 'weightings': list(frozen.MODES),
        'grouping': policy['grouping'], 'bootstrap': policy['bootstrap'],
        'fixed_test_ids_sha256': _json_hash(ids),
        'template_groups_sha256': _json_hash(groups),
        'full_test_group_sizes_sha256': _json_hash(full_sizes),
        'target_source': provenance,
        'TEST_target_accessor_calls': 1,
        'TEST_targets_returned_to_label_free_bundle': False,
        'evaluation_marker': {'path': str(marker), 'sha256': _sha(marker), 'scoring_pass': 1},
        'model_or_selection_calls_in_statistics': 0,
        'per_seed': per_seed,
        'arithmetic_seed_mean': frozen._score(context, ordered_names),
        'paired_other_conditions': {},
        'numerical_sources': policy['numerical_sources'],
        'interpretation_limits': [
            'TEST1000 is accessed only after all candidate selection and final model fitting are complete.',
            'Three-seed aggregation is the arithmetic mean of metrics, not a prediction ensemble or best-seed result.',
            'The same template-group draws and complete TEST group sizes are used for both weightings and every seed.',
            'Candidate generation sees300 samples from all2000 labeled examples, including any internal-validation members; the candidate-selection validation is not an untouched-label holdout.',
        ],
    }


def private_pairing_cache():
    """Return copies of terminal loss-derived draws for private stage4.4 use.

    There are no raw TEST targets in this cache. The coordinator must keep it
    outside prompts, queries and candidate selection, and save it only in H4.
    No target accessor or numerical recomputation occurs in this function.
    """
    if _LAST_PRIVATE_CACHE is None:
        raise RuntimeError('No successful H4 stage4.1 terminal evaluation cache is available')
    return {key: value.copy() for key, value in _LAST_PRIVATE_CACHE.items()}
