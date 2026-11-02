"""Meaningful CPU mocks; no real model, generated function, CLI or TEST access."""
from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import l1_models as m
import l1_data as d


class Sandbox:
    calls = []
    def __init__(self, sources, **kwargs):
        self.names = list(sources); self.settings = kwargs
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def evaluate_many(self, records, **kwargs):
        ids = [row['id'] for row in records]
        self.calls.append((ids, self.settings, kwargs))
        values = np.tile(np.array([10., 20., np.nan]), (len(ids), 1))
        status = np.tile(np.array(['ok', 'ok', 'exception']), (len(ids), 1))
        return {'ids': np.asarray(ids), 'feature_names': np.asarray(self.names), 'values': values,
            'status': status, 'errors': [{'row': i, 'feature': 2, 'status': 'exception', 'error': 'ValueError: CPU mock'} for i in range(len(ids))],
            'validation': [], 'performance': {'CPU_mock': True}}


class Fitter:
    def __init__(self): self.calls = []
    def __call__(self, train, y, predict):
        self.calls.append((train.copy(), y.copy(), predict.copy()))
        if train.shape[1] == 134: value = .4
        elif train.shape[1] == 133: value = .2 if np.nanmean(train[:, 132]) == 20. else .8
        else: value = .5
        return np.full(len(predict), value), {'parameters': m.HGB_PARAMETERS,
            'training_weights': 'unweighted', 'imputation': 'TRAIN-only median', 'fit_predict_seconds': 0.}


class Tests(unittest.TestCase):
    def setUp(self):
        top = m.safe(m.DATA / 'projection_data/software_model_tests')
        top.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=top); self.addCleanup(self.tmp.cleanup)
        self.out = m.safe(self.tmp.name)
        pool = ['P' + str(i).zfill(4) for i in range(2500)]
        test = ['T' + str(i).zfill(4) for i in range(1000)]
        ids = pool + test
        groups = {rid: 'g' + str(i) for i, rid in enumerate(ids)}
        mapping = {groups[rid]: 'val' if i % 5 == 0 else 'fit' for i, rid in enumerate(pool)}
        self.bundle = {'ids': ids, 'pool_ids': pool, 'test_ids': test,
            'index': {rid: i for i, rid in enumerate(ids)}, 'magpie132': np.zeros((len(ids), 132)),
            'structures_by_id': {rid: {} for rid in ids}, 'groups': groups, 'source_hashes': {},
            'perseed': {0: {'S0_ids': pool[:500], 'fit_group_mapping': mapping, 'cutoff': {'sha256': 'mock', 'group': 'mock'}}}}
        self.labels = {rid: 1. for rid in pool[:575]}
        self.features = [{'name': 'f' + str(i), 'description': 'CPU mock', 'source': 'def feature(s):\n    return ' + str(i) + '.0\n'} for i in range(3)]
        self.fitter = Fitter(); Sandbox.calls = []

    def baseline(self):
        return m._evaluate_round_baseline(self.bundle, 0, 1, self.features, self.labels,
            self.out / 'round1', sandbox_factory=Sandbox, model_fitter=self.fitter)

    def test_LOFO_sign_exact_deleted_columns_and_failed_raw_none_without_execution(self):
        baseline = self.baseline()
        self.assertEqual(baseline['stage'], '4.3'); self.assertEqual(baseline['condition'], 'L1')
        with patch.object(m.base, '_execute_features', side_effect=AssertionError('LOFO cannot execute code')):
            result = m.analyze_lofo(self.bundle, baseline, self.labels, self.out / 'lofo', model_fitter=self.fitter)
        values = [row['delta_without_minus_full'] for row in result['contributions']]
        self.assertAlmostEqual(values[0], .2); self.assertAlmostEqual(values[1], -.4)
        self.assertIsNone(values[2]); self.assertEqual(result['LOFO_fit_count'], 2)
        self.assertEqual([x[0].shape[1] for x in self.fitter.calls], [134, 133, 133])
        self.assertTrue(np.all(self.fitter.calls[1][0][:, 132] == 20.))
        self.assertTrue(np.all(self.fitter.calls[2][0][:, 132] == 10.))
        self.assertEqual(result['failures'][0]['feature_index'], 2)
        self.assertEqual(result['failures'][0]['failed_count'], len(baseline['identity']['fit_ids']))
        self.assertIn('exception|ValueError: CPU mock', result['failures'][0]['reasons'])
        self.assertEqual(len(Sandbox.calls), 1)
        self.assertFalse(result['TEST_targets_or_feedback_used'])

    def test_LOFO_completed_cache_skips_refits_and_detects_nested_prediction_tamper(self):
        baseline = self.baseline()
        first = m.analyze_lofo(self.bundle, baseline, self.labels, self.out / 'lofo', model_fitter=self.fitter)
        again = m.analyze_lofo(self.bundle, baseline, self.labels, self.out / 'lofo', model_fitter=self.fitter)
        self.assertEqual(first['record_sha256'], again['record_sha256']); self.assertEqual(len(self.fitter.calls), 3)
        ref = first['artifacts']['feature0_validation_predictions']
        m.safe(ref['path']).write_bytes(b'CPU test tamper')
        with self.assertRaisesRegex(RuntimeError, 'artifact hash'):
            m.analyze_lofo(self.bundle, baseline, self.labels, self.out / 'lofo', model_fitter=self.fitter)

    def test_LOFO_rejects_changed_saved_labels_before_any_refit(self):
        baseline = self.baseline(); changed = dict(self.labels); changed['P0000'] = 2.
        with self.assertRaisesRegex(ValueError, 'target identity changed'):
            m.analyze_lofo(self.bundle, baseline, changed, self.out / 'lofo', model_fitter=self.fitter)
        self.assertEqual(len(self.fitter.calls), 1)

    def test_empty_query_history_keeps_capacity_carry_and_all_raw_definitions(self):
        baseline = self.baseline()
        result = m.analyze_lofo(self.bundle, baseline, self.labels, self.out / 'lofo', model_fitter=self.fitter)
        query = {'query_payload': {'queries': []}, 'query_selection': {'queries': [], 'capacity': 75,
            'carry_in': 0, 'carry_out': 75, 'selected_n': 0}}
        history = m.history_record(baseline, result, query, 'CPU notes')
        self.assertEqual(history['features'], self.features)
        self.assertEqual(history['queries'], []); self.assertEqual(history['query_summary']['carry_out'], 75)
        self.assertEqual(history['query_summary']['requested_count'], 0)
        self.assertEqual(history['notes'], 'CPU notes')
        with self.assertRaisesRegex(ValueError, '300'):
            m.history_record(baseline, result, query, 'x' * 301)

    def test_initial_reuses_identical_S0_cache_never_fits_and_is_not_a_round_candidate(self):
        labels = {rid: 1. for rid in self.bundle['pool_ids'][:500]}
        split = m._split(self.bundle, 0, labels, initial=True)
        previous = m._validate(self.bundle, 0, split, [], self.out / 'old_initial', 'initial', None,
                               None, None, self.fitter)
        count = len(self.fitter.calls)
        with patch.object(m, '_original_initial_path', return_value=Path(previous['metadata_path'])):
            result = m.initial_state(self.bundle, 0, labels, self.out / 'L1_initial',
                model_fitter=lambda *a: (_ for _ in ()).throw(AssertionError('No initial refit')))
        self.assertEqual(len(self.fitter.calls), count)
        self.assertTrue(result['reused_initial_baseline']); self.assertFalse(result['winner_eligible'])
        self.assertEqual(result['stage'], '4.3'); self.assertEqual(result['condition'], 'L1')
        self.assertEqual(result['validation_mae'], previous['validation_mae'])
        self.assertEqual(result['artifacts'], previous['artifacts'])

    def test_output_guard_rejects_writes_to_old_phase(self):
        with self.assertRaisesRegex(ValueError, 'phase4_3'):
            m.safe(m.ROOT / 'work/h-data/H4/phase4_2/would_not_write')

    def test_L1_oracle_is_independent_and_never_looks_up_TEST_field(self):
        class Guard(dict):
            def __getitem__(self, key):
                if key == 'formation_energy_eV_per_atom': raise AssertionError('TEST target was looked up')
                return super().__getitem__(key)
        mixed = [{'row': {'id': rid, 'split': 'POOL', 'formation_energy_eV_per_atom': 1.}}
                 for rid in self.bundle['pool_ids'][:500]]
        mixed.append({'row': Guard(id='T0000', split='TEST')})
        source = d.ROOT / 'work/h-data/H2/synthetic_records_private.json.gz'
        self.bundle['source_hashes'][str(source)] = 'mock'
        old = deepcopy(d.frozen._REVEALED)
        with patch.object(d, '_REVEALED', {}), patch.object(d, 'sha', return_value='mock'), \
             patch.object(d.gzip, 'open', return_value=nullcontext(io.StringIO('CPU mock'))), \
             patch.object(d.json, 'load', return_value=mixed):
            labels = d.initial_labels(0, bundle=self.bundle)
            self.assertEqual(len(labels), 500); self.assertEqual(len(d._REVEALED[0]), 500)
        self.assertEqual(d.frozen._REVEALED, old)

    def test_L0_gate_cannot_authorize_L1_TEST_before_opening_target_source(self):
        gate = self.out / 'evaluation_started.json'
        gate.write_text(json.dumps({'stage': '4.2', 'condition': 'L0'}), encoding='utf-8')
        with patch.object(d, 'PHASE', self.out), patch.object(d, '_TRUTH_READ', False), \
             patch.object(d, 'sha', return_value='mock'), \
             patch.object(d.gzip, 'open', side_effect=AssertionError('No target source before correct gate')):
            with self.assertRaisesRegex(ValueError, 'gate identity'):
                d.score_truth(self.bundle['test_ids'], evaluation_marker=gate, bundle=self.bundle)


if __name__ == '__main__': unittest.main(verbosity=2)
