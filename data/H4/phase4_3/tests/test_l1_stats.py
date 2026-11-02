"""Pure synthetic CPU fixtures; no real TEST access, pairing or model calls."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'work/h-data/H4/phase4_3/src'))
from collections import Counter
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import l1_stats as stats

BASE = ROOT / 'work/h-data/H4/phase4_3/tests/software_stats'


def safe(path):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(BASE.resolve()):
        raise ValueError('Synthetic writes must remain inside new L1 tests')
    return path


def write(path, value):
    path = safe(path); safe(path.parent).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class L1StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.case = safe(BASE / uuid.uuid4().hex); self.case.mkdir(parents=True)
        self.location = patch.object(stats, 'DATA', self.case)
        self.location.start(); self.addCleanup(self.location.stop)
        for phase, condition in (('phase4_1', 'O'), ('phase4_2', 'L0')):
            write(self.case / phase / 'evaluation_started.json',
                {'stage': phase[5:].replace('_', '.'), 'condition': condition,
                 'scoring_pass': 1, 'synthetic_fixture': True})
        self.prior_bytes = {phase: (self.case / phase / 'evaluation_started.json').read_bytes()
            for phase in ('phase4_1', 'phase4_2')}
        self.marker = safe(self.case / 'phase4_3/evaluation_started.json')
        self.marker_value = stats.configuration()['marker_contract']
        write(self.marker, self.marker_value)
        self.ids = [f'toy_test_{i:04d}' for i in range(1000)]
        self.truth = {rid: i / 100. - 5. for i, rid in enumerate(self.ids)}
        self.groups = ['big' if i < 900 else 'small' for i in range(1000)]
        self.bundle = {'test_ids': list(self.ids), 'pool_ids': ['toy_pool'],
            'groups': dict(zip(self.ids, self.groups))}
        self.arms = [{'arm_id': f'L1_seed{seed}', 'condition': 'L1', 'seed': seed,
            'status': 'complete', 'prediction_rows': [{'id': rid,
                'prediction_eV_per_atom': self.truth[rid] + (.1, -.2, .3)[seed]}
                for rid in self.ids]} for seed in (0, 1, 2)]
        self.calls = 0
        def truth_accessor(ids, *, evaluation_marker, bundle=None):
            self.calls += 1
            self.assertEqual(list(ids), self.ids)
            self.assertEqual(Path(evaluation_marker), self.marker)
            return dict(self.truth), {'source': 'SYNTHETIC-CPU-FIXTURE'}
        def forbidden_old(*args, **kwargs):
            raise AssertionError('O/L0 truth must never be called')
        self.providers = patch.dict(sys.modules, {'l1_data': SimpleNamespace(score_truth=truth_accessor),
            'h4_data': SimpleNamespace(score_truth=forbidden_old),
            'h4_l0_data': SimpleNamespace(score_truth=forbidden_old)})
        self.providers.start(); self.addCleanup(self.providers.stop)

    def evaluate(self):
        return stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)

    def cache(self, condition):
        point = np.empty((2, 3, 3), dtype=np.float64)
        for mode in (0, 1):
            for seed in (0, 1, 2):
                point[mode, seed] = (.25 + mode * .01 + seed * .001, .35 + seed * .001, .7 + seed * .01)
        draws = np.broadcast_to(point, (2000,) + point.shape).copy()
        if condition == 'L0':
            for mode, (delta, lo, hi) in enumerate(((-.003126, -.010755, .004133), (-.002522, -.010568, .005686))):
                line = lo + (hi - lo) * (np.arange(2000) / 1999. - .025) / .95
                for seed in (0, 1, 2):
                    offset = (seed - 1) * .0001
                    point[mode, seed, 0] += delta + offset
                    draws[:, mode, seed, 0] += line + offset
            point[..., 1] -= .002; draws[..., 1] -= .002
            point[..., 2] += .001; draws[..., 2] += .001
        elif condition == 'L1':
            point[..., 0] -= .01; draws[..., 0] -= .01
            point[..., 1] -= .008; draws[..., 1] -= .008
            point[..., 2] += .005; draws[..., 2] += .005
        result = {'protocol': np.asarray(stats.CACHE_PROTOCOL),
            'stage': np.asarray({'O': '4.1', 'L0': '4.2', 'L1': '4.3'}[condition]),
            'arm_names': np.asarray([f'{condition}_seed{s}' for s in (0, 1, 2)]),
            'point': point, 'draws': draws, 'test_ids': np.asarray(self.ids),
            'groups': np.asarray(self.groups), 'bootstrap_seed': np.asarray(20261030),
            'n_boot': np.asarray(2000),
            'full_test_group_sizes_json': np.asarray(json.dumps(dict(Counter(self.groups)))),
            'TEST_target_values_in_cache': np.asarray(False)}
        if condition != 'O': result['condition'] = np.asarray(condition)
        if condition == 'L1': result['evaluation_marker_sha256'] = np.asarray(stats._sha(self.marker))
        return result

    def commit_synthetic_terminal(self, cache):
        per_seed = {str(seed): {'scores': {mode: {metric: {'point': float(cache['point'][i, seed, j])}
            for j, metric in enumerate(stats.frozen.METRICS)} for i, mode in enumerate(stats.frozen.MODES)}}
            for seed in (0, 1, 2)}
        result = {'stage': '4.3', 'condition': 'L1', 'TEST_target_accessor_calls': 1,
            'current_L1_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 2,
            'evaluation_marker': {'sha256': stats._sha(self.marker)},
            'fixed_test_ids_sha256': stats._json_hash(self.ids),
            'template_groups_sha256': stats._json_hash(self.groups),
            'full_test_group_sizes_sha256': stats._json_hash(dict(Counter(self.groups))),
            'per_seed': per_seed}
        write(self.case / 'phase4_3/terminal_results.json', result)

    def test_configuration_three_pass_disclosure_and_fixed_numerics(self):
        cfg = stats.configuration()
        self.assertEqual(cfg['marker_contract'], self.marker_value)
        self.assertEqual(cfg['bootstrap_n'], 2000)
        self.assertEqual(cfg['bootstrap_seed'], 20261030)
        self.assertEqual(cfg['marker_contract']['prior_H4_TEST_scoring_passes'], 2)
        self.assertEqual(self.calls, 0)

    def test_single_L1_target_read_all_seeds_no_old_access_and_no_bundle_change(self):
        before = copy.deepcopy(self.bundle); result = self.evaluate()
        self.assertEqual(self.calls, 1); self.assertEqual(before, self.bundle)
        self.assertEqual(result['H4_TEST_scoring_passes_after_this_stage'], 3)
        for phase, original in self.prior_bytes.items():
            self.assertEqual((self.case / phase / 'evaluation_started.json').read_bytes(), original)
        for seed, value in enumerate((.1, .2, .3)):
            self.assertAlmostEqual(result['per_seed'][str(seed)]['scores']['row_weighted']['mae']['point'], value)
        self.assertAlmostEqual(result['arithmetic_seed_mean']['row_weighted']['mae']['point'], .2)
        self.assertEqual(result['paired_other_conditions'], {})

    def test_complete_group_weights_and_all_2000_draws(self):
        for arm in self.arms:
            arm['predictions'] = {rid: self.truth[rid] + (1. if i < 900 else 0.) for i, rid in enumerate(self.ids)}
            del arm['prediction_rows']
        scores = self.evaluate()['arithmetic_seed_mean']
        self.assertAlmostEqual(scores['row_weighted']['mae']['point'], .9)
        self.assertAlmostEqual(scores['group_weighted']['mae']['point'], .5)
        self.assertEqual(scores['row_weighted']['mae']['n_valid_boot'], 2000)

    def test_O_L0_and_noncanonical_seed_namespace_rejected(self):
        for condition in ('O', 'L0'):
            self.arms[0]['condition'] = condition
            with self.assertRaises(ValueError): self.evaluate()
        self.arms[0]['condition'] = 'L1'; self.arms[0]['arm_id'] = 'O_seed0'
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_missing_duplicate_nonfinite_or_POOL_predictions_stop_before_truth(self):
        for defect in ('missing', 'duplicate', 'nonfinite', 'boolean', 'pool'):
            with self.subTest(defect=defect):
                rows = self.arms[0]['prediction_rows']; before = copy.deepcopy(rows)
                if defect == 'missing': rows.pop()
                elif defect == 'duplicate': rows[-1] = dict(rows[0])
                elif defect == 'nonfinite': rows[0]['prediction_eV_per_atom'] = float('nan')
                elif defect == 'boolean': rows[0]['prediction_eV_per_atom'] = False
                else: rows[0]['id'] = 'toy_pool'
                with self.assertRaises(ValueError): self.evaluate()
                self.arms[0]['prediction_rows'] = before
        self.assertEqual(self.calls, 0)

    def test_failed_or_duplicate_seed_stops_before_truth(self):
        self.arms[0]['status'] = 'failed'
        with self.assertRaises(ValueError): self.evaluate()
        self.arms[0]['status'] = 'complete'; self.arms[1]['seed'] = 0
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_missing_typed_marker_fields_or_changed_prior_hash_stop_before_truth(self):
        for key in self.marker_value:
            changed = dict(self.marker_value); del changed[key]; write(self.marker, changed)
            with self.assertRaises(ValueError): self.evaluate()
        changed = dict(self.marker_value); changed['scoring_pass'] = True; write(self.marker, changed)
        with self.assertRaises(ValueError): self.evaluate()
        write(self.marker, self.marker_value)
        write(self.case / 'phase4_2/evaluation_started.json', {'stage': '4.2', 'condition': 'L0', 'scoring_pass': 1, 'changed': True})
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_wrong_marker_path_is_rejected(self):
        path = self.case / 'other/evaluation_started.json'; write(path, self.marker_value)
        with self.assertRaises(ValueError): stats.evaluate(self.arms, self.bundle, evaluation_marker=path)
        self.assertEqual(self.calls, 0)

    def test_repeated_marker_and_partial_truth_failure_never_repeat_access(self):
        self.truth.pop(self.ids[-1])
        with self.assertRaises(ValueError): self.evaluate()
        with self.assertRaises(stats.ScientificStop): self.evaluate()
        self.assertEqual(self.calls, 1)

    def test_completed_cache_copies_no_targets_and_repeated_call_preserves_cache(self):
        self.evaluate(); cache = stats.private_pairing_cache()
        self.assertEqual(cache['draws'].shape, (2000, 2, 3, 3))
        self.assertEqual(cache['arm_names'].tolist(), ['L1_seed0', 'L1_seed1', 'L1_seed2'])
        self.assertFalse(cache['TEST_target_values_in_cache'].item())
        cache['point'][:] = -123.
        self.assertFalse((stats.private_pairing_cache()['point'] == -123.).any())
        before = stats.private_pairing_cache()['point']
        with self.assertRaises(stats.ScientificStop): self.evaluate()
        self.assertTrue(np.array_equal(before, stats.private_pairing_cache()['point']))

    def test_old_O_cache_without_condition_is_format_compatible(self):
        inspected = stats.inspect_cache(self.cache('O'), 'O')
        self.assertEqual(inspected['point_shape'], [2, 3, 3])
        self.assertEqual(inspected['draws_shape'], [2000, 2, 3, 3])
        self.assertFalse(inspected['raw_TEST_targets_present'])
        self.assertEqual(self.calls, 0)

    def test_NPZ_cache_read_is_pickle_free_and_within_new_test_directory(self):
        path = safe(self.case / 'cache/O.npz'); safe(path.parent).mkdir(parents=True)
        with path.open('wb') as stream: np.savez_compressed(stream, **self.cache('O'))
        self.assertEqual(stats.inspect_cache(path, 'O')['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_pairing_is_forbidden_before_committed_L1_terminal(self):
        caches = {c: self.cache(c) for c in ('O', 'L0', 'L1')}
        with patch.object(stats.frozen, '_paired', side_effect=AssertionError('must not form any pair yet')):
            with self.assertRaises(stats.ScientificStop):
                stats.paired_comparisons(caches, l1_evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_existing_draw_pairs_all_three_comparisons_no_truth_or_resampling(self):
        caches = {c: self.cache(c) for c in ('O', 'L0', 'L1')}
        self.commit_synthetic_terminal(caches['L1'])
        with patch.object(stats.frozen, '_bootstrap_context', side_effect=AssertionError('no new resampling')):
            result = stats.paired_comparisons(caches, l1_evaluation_marker=self.marker)
        self.assertEqual(set(result['comparisons']), {'L1_minus_L0', 'L1_minus_O', 'L0_minus_O'})
        self.assertEqual(result['registered_L0_minus_O_MAE_check']['status'], 'PASS')
        self.assertEqual(result['registered_L0_minus_O_MAE_check']['six_decimal_values'], stats._L0_O_REFERENCE)
        self.assertEqual(result['new_bootstrap_resamples'], 0)
        for pair in result['comparisons'].values():
            self.assertEqual(set(pair['per_seed']), {'0', '1', '2'})
            for mode in ('row_weighted', 'group_weighted'):
                self.assertEqual(set(pair['arithmetic_seed_mean'][mode]), {'mae', 'rmse', 'r2'})
        self.assertEqual(self.calls, 0)

    def test_L0_O_reference_mismatch_causes_scientific_stop(self):
        caches = {c: self.cache(c) for c in ('O', 'L0', 'L1')}
        self.commit_synthetic_terminal(caches['L1'])
        caches['L0']['point'][0, :, 0] += .00001
        with self.assertRaisesRegex(stats.ScientificStop, 'six decimal'):
            stats.paired_comparisons(caches, l1_evaluation_marker=self.marker)

    def test_cache_seed_ID_group_dtype_or_target_field_corruption_rejected(self):
        for defect in ('seed', 'IDs', 'groups', 'dtype', 'raw_targets', 'weights'):
            with self.subTest(defect=defect):
                cache = self.cache('L0')
                if defect == 'seed': cache['bootstrap_seed'] = np.asarray(1)
                elif defect == 'IDs': cache['test_ids'][0] = cache['test_ids'][1]
                elif defect == 'groups': cache['groups'][0] = 'changed'
                elif defect == 'dtype': cache['draws'] = cache['draws'].astype(np.float32)
                elif defect == 'raw_targets': cache['truth'] = np.arange(1000)
                else: cache['full_test_group_sizes_json'] = np.asarray('{}')
                with self.assertRaises(stats.ScientificStop): stats.inspect_cache(cache, 'L0')

    def test_cross_stage_identity_or_L1_terminal_point_mismatch_stops_pairing(self):
        caches = {c: self.cache(c) for c in ('O', 'L0', 'L1')}
        self.commit_synthetic_terminal(caches['L1'])
        caches['O']['test_ids'] = caches['O']['test_ids'][::-1]
        with self.assertRaises(stats.ScientificStop): stats.paired_comparisons(caches, l1_evaluation_marker=self.marker)
        caches['O'] = self.cache('O'); caches['L1']['point'][0, 0, 0] += .001
        with self.assertRaises(stats.ScientificStop): stats.paired_comparisons(caches, l1_evaluation_marker=self.marker)


if __name__ == '__main__': unittest.main()
