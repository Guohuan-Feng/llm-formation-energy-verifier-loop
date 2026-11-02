"""Saved synthetic metadata/report fixtures; never run a scientific operation."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / 'work/h-data/H4/phase4_3/src'))
import json
import hashlib
import unittest
from unittest.mock import patch
import uuid
import l1_report as report

BASE = ROOT / 'work/h-data/H4/phase4_3/tests/software_report'


def safe(path):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(BASE.resolve()):
        raise ValueError('Mock reporting writes must remain in new L1 software_report')
    return path


def sealed(payload):
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    return {**payload, 'record_sha256': digest}


def write(path, payload, *, seal=False):
    path = safe(path); safe(path.parent).mkdir(parents=True, exist_ok=True)
    value = sealed(payload) if seal else payload
    path.write_text(json.dumps(value), encoding='utf-8')
    return value


def scores(offset=0.):
    return {mode: {metric: {'point': value + offset,
        'ci95': [value + offset - .01, value + offset + .01],
        'n_valid_boot': 2000, 'n_undefined_boot': 0}
        for metric, value in zip(('mae', 'rmse', 'r2'), (.2, .3, .8))}
        for mode in ('row_weighted', 'group_weighted')}


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.case = safe(BASE / uuid.uuid4().hex); self.case.mkdir(parents=True)
        self.location = patch.object(report, 'PHASE', self.case)
        self.location.start(); self.addCleanup(self.location.stop)
        self.old_location = patch.object(report, 'OLD_PHASE', self.case / 'old_L0')
        self.old_location.start(); self.addCleanup(self.old_location.stop)
        self.features = [
            {'name': 'sites', 'description': 'Site count', 'source': "def feature(s):\n    return float(s['n_sites'])\n"},
            {'name': 'volume', 'description': 'Atomic volume', 'source': "def feature(s):\n    return float(s['volume_per_atom_A3'])\n"},
            {'name': 'dropped', 'description': 'Unretained', 'source': "def feature(s):\n    return float(s['n_elements'])\n"}]
        self.states = {}; self.finals = []
        for seed in (0, 1, 2):
            oldpaths = []; paths = []; winner_index = 17 if seed == 2 else 20
            for condition in ('L0', 'L1'):
                for rnd in range(1, 21):
                    base = self.case / 'old_L0' if condition == 'L0' else self.case
                    path = base / 'rounds' / f'{condition}_seed{seed}' / f'round{rnd:02d}' / 'round.json'
                    n = 500 + 75 * rnd - (50 if seed == 1 and rnd == 20 else 0)
                    mae = .22 - .0002 * rnd + .003 * seed if condition == 'L0' else .2 + .003 * seed + abs(rnd - winner_index) * .0003
                    record = write(path, {'status': 'complete', 'condition': condition,
                        'seed': seed, 'round_index': rnd, 'labelled_count': n,
                        'fit_count': int(n * .8), 'validation_count': n - int(n * .8),
                        'validation_mae': mae, 'validation_group_mae': mae + .01,
                        'features': self.features, 'metadata_path': str(path.resolve())}, seal=True)
                    if condition == 'L0': oldpaths.append(str(path.resolve()))
                    else:
                        paths.append(str(path.resolve()))
                        if rnd == winner_index:
                            self.winner = record
                            lofo_path = path.parent / 'lofo/lofo.json'
                            write(lofo_path, {'status': 'complete', 'seed': seed, 'round_index': rnd,
                                'identity': {'source_round_record_sha256': record['record_sha256']},
                                'raw_features': self.features, 'baseline_validation_mae': mae,
                                'contributions': [{'feature_index': 0, 'delta_without_minus_full': .002},
                                    {'feature_index': 1, 'delta_without_minus_full': -.001},
                                    {'feature_index': 2, 'delta_without_minus_full': .1}]}, seal=True)
            write(self.case / 'old_L0/states' / f'seed{seed}.json', {'seed': seed, 'round_model_paths': oldpaths})
            audits = [{'request_id': f'L1_seed{seed}_round{rnd:02d}', 'round_index': rnd,
                'memory_token_count': 100 + 10 * rnd, 'memory_UTF8_bytes': 800 + 40 * rnd,
                'memory_trimmed': rnd in (8, 16),
                'input_budget': {'input_limit': 70000, 'input_token_upper_bound': 20000 + 10 * rnd, 'within_budget': True},
                'format_repair_input_budget': {'input_limit': 70000, 'input_token_upper_bound': 20155 + 10 * rnd, 'within_budget': True}}
                for rnd in range(1, 21)]
            self.states[seed] = {'seed': seed, 'round_completed': 20, 'round_model_paths': paths,
                'history': [{'round_index': rnd} for rnd in range(1, 21)],
                'memory_audits': audits, 'final_fill': {'selected_n': 50 if seed == 1 else 0}}
            winner = json.loads(Path(paths[winner_index - 1]).read_text())
            self.finals.append({'seed': seed, 'train_count': 2000, 'test_count': 1000,
                'selected_round_index': winner_index, 'selected_round_record_sha256': winner['record_sha256'],
                'full_train_features': {'kept_names': ['sites', 'volume'], 'failure_rates': [0., .01, 1.],
                    'status_counts': {'ok': 3980, 'exception': 20, 'ast_rejected': 2000}},
                'test_features': {'failure_rates': [.005, .02, 1.],
                    'status_counts': {'ok': 1975, 'exception': 25, 'ast_rejected': 1000}}})
        self.terminal = {'stage': '4.3', 'condition': 'L1', 'n_test': 1000, 'n_groups': 333, 'n_boot': 2000,
            'per_seed': {str(seed): {'scores': scores(.001 * seed)} for seed in (0, 1, 2)},
            'arithmetic_seed_mean': scores(.001)}
        self.paired = {'registered_L0_minus_O_MAE_check': {'status': 'PASS'},
            'comparisons': {name: {'per_seed': {str(seed): scores(-.002 - seed * .0001) for seed in (0, 1, 2)},
                'arithmetic_seed_mean': scores(-.0021)} for name in report.PAIRS}}
        self.summary = {'cli_attempt_count': 180, 'per_kind': {
            'features': {'input_tokens': 1000, 'output_tokens': 2000, 'cached_input_tokens': 0},
            'features_query_memory': {'input_tokens': 30000, 'output_tokens': 40000, 'cached_input_tokens': 100}}}
        self.cfg = {'prompts': {'input_limit': 70000, 'input_budget_method': 'SYNTHETIC immutable UTF8 accounting'},
            'input_limit_basis': {'measurement': 'SYNTHETIC fixture only', 'estimated_bytes': 63000,
                'formula': 'recorded estimate plus margin, rounded before freeze'}}

    def make(self):
        return report.make_report(self.states, self.terminal, self.finals, self.paired,
            self.summary, 1234.5, self.cfg)

    def test_complete_mock_report_plot_and_ledgers_without_scientific_operations(self):
        paths = self.make()
        self.assertTrue(all(Path(path).is_file() for path in paths.values()))
        self.assertEqual(Path(paths['validation_curve']).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        text = Path(paths['report']).read_text(encoding='utf-8')
        self.assertIn('0.201000 [0.191000, 0.211000]', text)
        self.assertIn('total is three', text)
        self.assertIn('does not isolate a causal effect of notes alone', text)
        self.assertIn('L1 attempts:60; global attempts:180/300', text)
        self.assertIn('70000 bytes', text)
        self.assertIn('SYNTHETIC fixture only', text)
        with Path(paths['curve_csv']).open(encoding='utf-8') as stream:
            self.assertEqual(sum(1 for line in stream) - 1, 120)
        with Path(paths['request_input_ledger']).open(encoding='utf-8') as stream:
            self.assertEqual(sum(1 for line in stream) - 1, 60)
        top = json.loads(Path(paths['top_feature_sources']).read_text())
        for seed in ('0', '1', '2'):
            self.assertEqual([f['name'] for f in top[seed]['features']], ['sites', 'volume'])
            self.assertEqual(top[seed]['features'][0]['source'], self.features[0]['source'])
            self.assertEqual(top[seed]['features'][1]['delta_without_minus_full'], -.001)

    def test_dropped_high_LOFO_feature_is_not_promoted_by_report(self):
        paths = self.make(); top = json.loads(Path(paths['top_feature_sources']).read_text())
        self.assertFalse(any(f['name'] == 'dropped' for item in top.values() for f in item['features']))

    def test_invalid_reference_or_memory_bound_refuses_report(self):
        self.paired['registered_L0_minus_O_MAE_check']['status'] = 'FAIL'
        with self.assertRaises(report.ReportIntegrityError): self.make()
        self.paired['registered_L0_minus_O_MAE_check']['status'] = 'PASS'
        self.states[0]['memory_audits'][0]['memory_token_count'] = 6001
        with self.assertRaises(report.ReportIntegrityError): self.make()
        self.assertFalse((self.case / 'report/H4_stage4_3_report.md').exists())

    def test_changed_winner_LOFO_binding_refuses_report(self):
        path = Path(self.states[0]['round_model_paths'][-1]).parent / 'lofo/lofo.json'
        value = json.loads(path.read_text()); value.pop('record_sha256')
        value['identity']['source_round_record_sha256'] = 'wrong'
        write(path, value, seal=True)
        with self.assertRaises(report.ReportIntegrityError): self.make()

    def test_undefined_intervals_display_as_undefined_without_new_statistics(self):
        self.assertEqual(report._metric({'point': None, 'ci95': [None, None]}),
            'undefined [undefined, undefined]')


if __name__ == '__main__': unittest.main()
