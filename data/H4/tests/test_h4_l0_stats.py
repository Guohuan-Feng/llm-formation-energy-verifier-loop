"""Synthetic CPU-only L0 statistics and distinct-stage terminal gate checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'work/h-data/H4/src'))
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np
import h4_l0_stats as stats

BASE = ROOT / 'work/h-data/H4/software_l0_stats'


def safe(path):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(BASE.resolve()):
        raise ValueError('Synthetic test writes must stay in H4 software_l0_stats')
    return path


def write(path, value):
    path = safe(path)
    safe(path.parent).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class L0StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.case = safe(BASE / uuid.uuid4().hex)
        self.case.mkdir(parents=True)
        self.location = patch.object(stats, 'DATA', self.case)
        self.location.start(); self.addCleanup(self.location.stop)
        self.prior = safe(self.case / 'phase4_1/evaluation_started.json')
        write(self.prior, {'stage': '4.1', 'condition': 'O', 'scoring_pass': 1,
            'TEST_targets_previously_read': False, 'synthetic_test_fixture': True})
        self.prior_bytes = self.prior.read_bytes()
        self.marker = safe(self.case / 'phase4_2/evaluation_started.json')
        self.marker_value = {'stage': '4.2', 'condition': 'L0', 'scoring_pass': 1,
            'current_L0_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 1,
            'TEST_targets_previously_read_for_this_condition': False,
            'prior_stage4_1_marker_sha256': hashlib.sha256(self.prior_bytes).hexdigest()}
        write(self.marker, self.marker_value)
        self.ids = [f'toy_test_{i:04d}' for i in range(1000)]
        self.truth = {rid: i / 100. - 5. for i, rid in enumerate(self.ids)}
        self.bundle = {'test_ids': list(self.ids), 'pool_ids': ['toy_pool'],
            'groups': {rid: 'large' if i < 900 else 'small' for i, rid in enumerate(self.ids)}}
        self.arms = [{'arm_id': f'L0_seed{seed}', 'condition': 'L0', 'seed': seed,
            'method': 'HGB_Magpie_LLM', 'status': 'complete',
            'prediction_rows': [{'id': rid,
                'prediction_eV_per_atom': self.truth[rid] + (.1, -.2, .3)[seed]}
                for rid in self.ids]} for seed in (0, 1, 2)]
        self.calls = 0
        def truth_accessor(ids, *, evaluation_marker, bundle=None):
            self.calls += 1
            self.assertEqual(list(ids), self.ids)
            self.assertEqual(Path(evaluation_marker), self.marker)
            return dict(self.truth), {'path': 'SYNTHETIC-CPU-FIXTURE',
                'target_values_returned_to_bundle': False}
        def old_accessor(*args, **kwargs):
            raise AssertionError('Old O target accessor must not be called')
        self.provider = patch.dict(sys.modules, {
            'h4_l0_data': SimpleNamespace(score_truth=truth_accessor),
            'h4_data': SimpleNamespace(score_truth=old_accessor)})
        self.provider.start(); self.addCleanup(self.provider.stop)

    def evaluate(self):
        return stats.evaluate(self.arms, self.bundle, evaluation_marker=self.marker)

    def test_configuration_freezes_distinct_L0_pass_and_2000_bootstrap(self):
        cfg = stats.configuration()
        self.assertEqual(cfg['stage'], '4.2')
        self.assertEqual(cfg['condition'], 'L0')
        self.assertEqual(cfg['bootstrap_n'], 2000)
        self.assertEqual(cfg['bootstrap_seed'], 20261030)
        self.assertEqual(cfg['marker_contract'], self.marker_value)
        self.assertFalse(cfg['O_results_recomputed'])
        self.assertFalse(cfg['selection_uses_TEST'])
        self.assertEqual(self.calls, 0)

    def test_single_L0_read_scores_three_seeds_and_preserves_O_marker(self):
        original_bundle = copy.deepcopy(self.bundle)
        result = self.evaluate()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.bundle, original_bundle)
        self.assertEqual(self.prior.read_bytes(), self.prior_bytes)
        self.assertEqual(set(result['per_seed']), {'0', '1', '2'})
        self.assertEqual(result['n_test'], 1000)
        self.assertEqual(result['H4_TEST_scoring_passes_after_this_stage'], 2)
        self.assertFalse(result['O_results_recomputed'])
        self.assertEqual(result['paired_other_conditions'], {})
        for seed, value in enumerate((.1, .2, .3)):
            self.assertAlmostEqual(result['per_seed'][str(seed)]['scores']['row_weighted']['mae']['point'], value)
        mean = result['arithmetic_seed_mean']['row_weighted']['mae']
        self.assertAlmostEqual(mean['point'], .2)
        self.assertNotAlmostEqual(mean['point'], (.1 - .2 + .3) / 3.)
        self.assertEqual(mean['n_valid_boot'], 2000)

    def test_group_weighting_uses_complete_TEST_group_counts(self):
        for arm in self.arms:
            arm['predictions'] = {rid: self.truth[rid] + (1. if i < 900 else 0.)
                for i, rid in enumerate(self.ids)}
            del arm['prediction_rows']
        result = self.evaluate()['arithmetic_seed_mean']
        self.assertAlmostEqual(result['row_weighted']['mae']['point'], .9)
        self.assertAlmostEqual(result['group_weighted']['mae']['point'], .5)

    def test_O_arm_is_rejected_before_truth_access(self):
        self.arms[0]['condition'] = 'O'; self.arms[0]['arm_id'] = 'O_seed0'
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_old_O_namespace_is_rejected_even_if_condition_says_L0(self):
        self.arms[0]['arm_id'] = 'O_seed0'
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_missing_duplicate_and_nonfinite_predictions_stop_before_truth(self):
        for defect in ('missing', 'duplicate', 'nonfinite', 'boolean', 'pool'):
            with self.subTest(defect=defect):
                rows = self.arms[0]['prediction_rows']; original = copy.deepcopy(rows)
                if defect == 'missing': rows.pop()
                elif defect == 'duplicate': rows[-1] = dict(rows[0])
                elif defect == 'nonfinite': rows[0]['prediction_eV_per_atom'] = float('nan')
                elif defect == 'boolean': rows[0]['prediction_eV_per_atom'] = True
                else: rows[0]['id'] = 'toy_pool'
                with self.assertRaises(ValueError): self.evaluate()
                self.arms[0]['prediction_rows'] = original
        self.assertEqual(self.calls, 0)

    def test_failed_or_duplicate_seed_stops_before_truth(self):
        self.arms[0]['status'] = 'failed'
        with self.assertRaises(ValueError): self.evaluate()
        self.arms[0]['status'] = 'complete'; self.arms[1]['seed'] = 0
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_all_marker_disclosures_are_required_and_typed(self):
        for key in self.marker_value:
            with self.subTest(key=key):
                changed = dict(self.marker_value); del changed[key]; write(self.marker, changed)
                with self.assertRaises(ValueError): self.evaluate()
        for key in ('scoring_pass', 'current_L0_scoring_pass', 'prior_H4_TEST_scoring_passes'):
            changed = dict(self.marker_value); changed[key] = True; write(self.marker, changed)
            with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_prior_marker_hash_mismatch_stops_before_truth(self):
        changed = dict(self.marker_value); changed['prior_stage4_1_marker_sha256'] = '0' * 64
        write(self.marker, changed)
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_changed_prior_O_marker_stops_before_truth(self):
        write(self.prior, {'stage': '4.1', 'condition': 'O', 'scoring_pass': 1, 'modified': True})
        with self.assertRaises(ValueError): self.evaluate()
        self.assertEqual(self.calls, 0)

    def test_wrong_stage_or_marker_path_stops_before_truth(self):
        changed = dict(self.marker_value); changed['stage'] = '4.1'; write(self.marker, changed)
        with self.assertRaises(ValueError): self.evaluate()
        write(self.marker, self.marker_value)
        other = safe(self.case / 'other/evaluation_started.json'); write(other, self.marker_value)
        with self.assertRaises(ValueError):
            stats.evaluate(self.arms, self.bundle, evaluation_marker=other)
        self.assertEqual(self.calls, 0)

    def test_repeated_marker_never_reads_truth_twice_or_discards_cache(self):
        self.evaluate(); before = stats.private_pairing_cache()
        with self.assertRaises(RuntimeError): self.evaluate()
        self.assertEqual(self.calls, 1)
        self.assertTrue(np.array_equal(before['point'], stats.private_pairing_cache()['point']))

    def test_partial_truth_failure_consumes_read(self):
        self.truth.pop(self.ids[-1])
        with self.assertRaises(ValueError): self.evaluate()
        with self.assertRaises(RuntimeError): self.evaluate()
        self.assertEqual(self.calls, 1)

    def test_private_cache_is_L0_only_with_no_raw_targets_and_returns_copies(self):
        self.evaluate(); cache = stats.private_pairing_cache()
        self.assertEqual(cache['arm_names'].tolist(), ['L0_seed0', 'L0_seed1', 'L0_seed2'])
        self.assertEqual(cache['stage'].item(), '4.2')
        self.assertEqual(cache['point'].shape, (2, 3, 3))
        self.assertEqual(cache['draws'].shape, (2000, 2, 3, 3))
        self.assertFalse(cache['TEST_target_values_in_cache'].item())
        self.assertFalse(any(key in cache for key in ('truth', 'y', 'targets', 'formation_energy_eV_per_atom')))
        old = cache['point'].copy(); cache['point'][:] = -999.
        self.assertTrue(np.array_equal(old, stats.private_pairing_cache()['point']))


if __name__ == '__main__': unittest.main()
