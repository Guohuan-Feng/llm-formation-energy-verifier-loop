"""Pure synthetic CPU checks for stage4.1 statistics and leakage gates."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import h4_stats as stats

BASE = stats.DATA / 'software_stats'


def safe(path):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(BASE.resolve()):
        raise ValueError('Synthetic test writes must stay in the local H4 software_stats directory')
    return path


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.case = safe(BASE / uuid.uuid4().hex)
        self.case.mkdir(parents=True)
        self.marker = safe(self.case / 'evaluation_started.json')
        self.marker.write_text(json.dumps({'stage': '4.1', 'scoring_pass': 1,
                                         'TEST_targets_previously_read': False}), encoding='utf-8')
        self.ids = [f'toy_test_{i:04d}' for i in range(1000)]
        self.truth = {rid: i / 100. - 5. for i, rid in enumerate(self.ids)}
        self.bundle = {'test_ids': list(self.ids), 'pool_ids': ['toy_pool'],
                       'groups': {rid: 'large' if i < 900 else 'small' for i, rid in enumerate(self.ids)}}
        offsets = (.1, -.2, .3)
        self.arms = [{'arm_id': f'O_seed{seed}', 'condition': 'O', 'seed': seed,
                      'method': 'HGB_Magpie_LLM', 'status': 'complete',
                      'prediction_rows': [{'id': rid, 'prediction_eV_per_atom': self.truth[rid] + offsets[seed]}
                                          for rid in self.ids]} for seed in (0, 1, 2)]
        self.calls = 0
        def truth_accessor(ids, *, evaluation_marker, bundle=None):
            self.calls += 1
            self.assertEqual(list(ids), self.ids)
            self.assertEqual(Path(evaluation_marker), self.marker)
            return dict(self.truth), {'path': 'SYNTHETIC-CPU-FIXTURE', 'sha256': 'mock',
                                     'target_values_returned_to_bundle': False}
        self.provider = SimpleNamespace(score_truth=truth_accessor)
        self.stub = patch.dict(sys.modules, {'h4_data': self.provider})
        self.stub.start(); self.addCleanup(self.stub.stop)

    def test_configuration_is_terminal_fixed1000_and_frozen2000_bootstrap(self):
        cfg = stats.configuration()
        self.assertEqual(cfg['test_n'], 1000)
        self.assertEqual(cfg['bootstrap_n'], 2000)
        self.assertEqual(cfg['bootstrap_seed'], 20261030)
        self.assertFalse(cfg['selection_uses_TEST'])
        self.assertEqual(self.calls, 0)

    def test_single_truth_read_scores_all_seeds_and_arithmetic_metric_mean(self):
        original_bundle = copy.deepcopy(self.bundle)
        result = stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.bundle, original_bundle)
        self.assertEqual(result['n_test'], 1000)
        self.assertEqual(set(result['per_seed']), {'0', '1', '2'})
        self.assertAlmostEqual(result['per_seed']['0']['scores']['row_weighted']['mae']['point'], .1)
        self.assertAlmostEqual(result['per_seed']['1']['scores']['row_weighted']['mae']['point'], .2)
        self.assertAlmostEqual(result['per_seed']['2']['scores']['row_weighted']['mae']['point'], .3)
        mean = result['arithmetic_seed_mean']['row_weighted']['mae']
        self.assertAlmostEqual(mean['point'], .2)
        self.assertNotAlmostEqual(mean['point'], (.1 - .2 + .3) / 3.)
        self.assertEqual(mean['n_valid_boot'], 2000)
        self.assertEqual(result['model_or_selection_calls_in_statistics'], 0)

    def test_complete_TEST_group_sizes_define_group_weights(self):
        for arm in self.arms:
            arm['prediction_rows'] = [{'id': rid, 'prediction_eV_per_atom': self.truth[rid] + (1. if i < 900 else 0.)}
                                      for i, rid in enumerate(self.ids)]
        result = stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        scores = result['arithmetic_seed_mean']
        self.assertAlmostEqual(scores['row_weighted']['mae']['point'], .9)
        self.assertAlmostEqual(scores['group_weighted']['mae']['point'], .5)
        self.assertEqual(result['n_groups'], 2)

    def test_missing_prediction_stops_before_any_target_read(self):
        self.arms[0]['prediction_rows'].pop()
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_duplicate_prediction_stops_before_any_target_read(self):
        self.arms[0]['prediction_rows'][-1] = dict(self.arms[0]['prediction_rows'][0])
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_nonfinite_prediction_stops_before_any_target_read(self):
        self.arms[0]['prediction_rows'][0]['prediction_eV_per_atom'] = float('nan')
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_POOL_export_identity_stops_before_any_target_read(self):
        self.arms[0]['prediction_rows'][0]['id'] = 'toy_pool'
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_failed_model_stops_before_any_target_read(self):
        self.arms[0]['status'] = 'failed'
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_wrong_stage_marker_stops_before_any_target_read(self):
        self.marker.write_text(json.dumps({'stage': '4.2', 'scoring_pass': 1,
                                         'TEST_targets_previously_read': False}), encoding='utf-8')
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 0)

    def test_repeated_marker_does_not_read_targets_twice(self):
        stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        with self.assertRaises(RuntimeError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 1)

    def test_partial_truth_failure_still_consumes_terminal_read(self):
        self.truth.pop(self.ids[-1])
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 1)
        with self.assertRaises(RuntimeError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        self.assertEqual(self.calls, 1)

    def test_private_draw_cache_has_no_raw_targets_and_returns_copies(self):
        stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)
        cache = stats.private_pairing_cache()
        self.assertEqual(cache['point'].shape, (2, 3, 3))
        self.assertEqual(cache['draws'].shape, (2000, 2, 3, 3))
        self.assertFalse(cache['TEST_target_values_in_cache'].item())
        self.assertFalse(any(key in cache for key in ('truth', 'y', 'targets', 'formation_energy_eV_per_atom')))
        old = cache['point'].copy(); cache['point'][:] = -999.
        self.assertTrue(np.array_equal(stats.private_pairing_cache()['point'], old))


if __name__ == '__main__':
    unittest.main()
