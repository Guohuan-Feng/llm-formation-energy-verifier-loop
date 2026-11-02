"""Saved synthetic metadata/report fixtures; never run a scientific operation."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'work/h-data/H4/phase4_3/memory_20000_revision_20261031/src'))
import json
import hashlib
import unittest
from unittest.mock import patch
import uuid
import l1_revision_report as report

BASE = ROOT / 'work/h-data/H4/phase4_3/memory_20000_revision_20261031/tests/software_report'


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
        self.ledger_location = patch.object(report, 'LEDGER', self.case / 'llm_calls.jsonl')
        self.ledger_location.start(); self.addCleanup(self.ledger_location.stop)
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
            audits = [{'request_id': f'L1R2_seed{seed}_round{rnd:02d}', 'round_index': rnd,
                'memory_token_count': 1000 + 700 * rnd, 'memory_UTF8_bytes': 800 + 40 * rnd,
                'memory_trimmed': rnd in (8, 16),
                'memory_policy_audit': {'name_description_source_or_source_summary_retained': False,
                    'older_round_notes_queries_retained_in_any_counted_view': False,
                    'recent_rounds_preserved': list(range(max(1, rnd - 3), rnd))},
                'input_budget': {'input_limit': 189000, 'canonical_prompt_bytes': 14000 + 10 * rnd, 'wire_prompt_bytes': 14100 + 10 * rnd, 'known_fixed_context_bytes': 4400, 'input_token_upper_bound': 20000 + 10 * rnd, 'within_budget': True},
                'format_repair_input_budget': {'input_limit': 189000, 'canonical_prompt_bytes': 14000 + 10 * rnd, 'wire_prompt_bytes': 14255 + 10 * rnd, 'known_fixed_context_bytes': 4400, 'input_token_upper_bound': 20155 + 10 * rnd, 'within_budget': True}}
                for rnd in range(1, 21)]
            self.states[seed] = {'seed': seed, 'round_completed': 20, 'round_model_paths': paths,
                'history': [{'round_index': rnd, 'notes': '科学note' + str(rnd)} for rnd in range(1, 21)],
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
        self.summary = {'cli_attempt_count': 183, 'per_kind': {
            'features': {'input_tokens': 1000, 'output_tokens': 2000, 'cached_input_tokens': 0},
            'features_query_memory': {'input_tokens': 30000, 'output_tokens': 40000, 'cached_input_tokens': 100}}}
        self.cfg = {'prompts': {'input_limit': 189000, 'input_budget_method': 'SYNTHETIC immutable UTF8 accounting'},
            'input_limit_basis': {'measurement': 'SYNTHETIC fixture only', 'estimated_bytes': 63000,
                'formula': 'recorded estimate plus margin, rounded before revision freeze'},
            'memory': {'limit_tokens': 20000},
            'token_counter': {'limit_tokens': 6000, 'token_counter_sha256': 'f' * 64},
            'revision_provenance': {'original_config_sha256': 'a' * 64,
                'original_frozen_utc': '2026-10-31T19:13:44+00:00',
                'revision_utc': '2026-10-31T22:00:00+00:00', 'reason': 'SYNTHETIC explicit memory revision',
                'voided_calls': [], 'new_request_namespace': 'L1R2',
                'archive_manifest_path': str(self.case / 'mock_archive_manifest.json')}}
        self.records = []
        for index in range(1, 184):
            if index <= 60:
                offset = index - 1; request = f'O_seed{offset // 20}_candidate{offset % 20:02d}'
            elif index <= 120:
                offset = index - 61; request = f'L0_seed{offset // 20}_round{offset % 20 + 1:02d}'
            elif index <= 123:
                request = f'L1_seed{index - 121}_round01'
            else:
                offset = index - 124; request = f'L1R2_seed{offset // 20}_round{offset % 20 + 1:02d}'
            metadata = {'call_index': index, 'request_id': request, 'prompt_sha256': hashlib.sha256(request.encode()).hexdigest()}
            self.records.append({**metadata, 'record_type': 'reservation'})
            result = {**metadata, 'record_type': 'result', 'status': 'ok',
                'canonical_prompt_sha256': hashlib.sha256((request + ':canonical').encode()).hexdigest(),
                'validated_response_sha256': hashlib.sha256((request + ':response').encode()).hexdigest(),
                'input_tokens': 1000, 'output_tokens': 2000, 'cached_input_tokens': 100}
            self.records.append(result)
            if 121 <= index <= 123:
                self.cfg['revision_provenance']['voided_calls'].append({k: result[k] for k in
                    ('request_id', 'call_index', 'prompt_sha256', 'canonical_prompt_sha256', 'validated_response_sha256')})
        self.save_ledger()

    def save_ledger(self):
        path = safe(self.case / 'llm_calls.jsonl')
        path.write_text(''.join(json.dumps(row) + '\n' for row in self.records), encoding='utf-8')

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
        self.assertIn('valid L1R2 logical requests:60', text)
        self.assertIn('global physical attempts:183/300', text)
        self.assertIn('all L1 physical attempts:63', text)
        self.assertIn('Original L1 freeze UTC:', text)
        self.assertIn('Authorized revision UTC:', text)
        self.assertIn('legacy provenance', text)
        self.assertNotIn('set before the first L1 call', text)
        audit = json.loads(Path(paths['revision_audit']).read_text())
        self.assertEqual(audit['call_accounting']['formal_valid_logical_requests'], 180)
        self.assertEqual(audit['call_accounting']['tokens_by_scope']['formal_L1']['input_tokens'], 60000)
        self.assertEqual(audit['call_accounting']['tokens_by_scope']['voided_L1']['input_tokens'], 3000)
        self.assertIn('189000 bytes', text)
        self.assertIn('Active revised memory cap:20000 tokens', text)
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
        self.states[0]['memory_audits'][0]['memory_token_count'] = 20001
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



    def test_voided_prompt_mismatch_stops_before_report_write(self):
        self.cfg['revision_provenance']['voided_calls'][0]['canonical_prompt_sha256'] = '0' * 64
        with self.assertRaises(report.ReportIntegrityError): self.make()
        self.assertFalse((self.case / 'report/H4_stage4_3_report.md').exists())

    def test_physical_retries_and_unknown_usage_are_not_hidden(self):
        request = 'L1R2_seed0_round01'
        old = next(row for row in self.records if row.get('record_type') == 'result' and row['request_id'] == request)
        old['status'] = 'rate_limited'; old['validated_response_sha256'] = None; old['input_tokens'] = None
        meta = {k: old[k] for k in ('request_id', 'prompt_sha256')}; meta['call_index'] = 184
        self.records.extend([{**meta, 'record_type': 'reservation'}, {**meta, 'record_type': 'result',
            'status': 'ok', 'validated_response_sha256': 'b' * 64, 'input_tokens': 500, 'output_tokens': 1000, 'cached_input_tokens': 0}])
        self.summary['cli_attempt_count'] = 184; self.save_ledger()
        accounting = report._call_accounting(self.summary, report._revision_provenance(self.cfg))
        self.assertEqual(accounting['global_physical_cli_attempts'], 184)
        self.assertEqual(accounting['formal_valid_logical_requests'], 180)
        self.assertEqual(accounting['formal_L1_cli_attempts'], 61)
        self.assertEqual(accounting['tokens_by_scope']['formal_L1']['unknown_token_results'], 1)
        self.assertEqual(accounting['tokens_by_scope']['formal_L1']['input_tokens'], 59500)

    def test_wrong_actual_byte_accounting_and_old_namespace_refused(self):
        audit = self.states[0]['memory_audits'][0]
        audit['input_budget']['wire_prompt_bytes'] += 1
        with self.assertRaises(report.ReportIntegrityError): report._memory_rows(self.states, 189000)
        audit['input_budget']['wire_prompt_bytes'] -= 1
        audit['request_id'] = 'L1_seed0_round01'
        with self.assertRaises(report.ReportIntegrityError): report._memory_rows(self.states, 189000)

    def test_per_round_actual_note_and_byte_values(self):
        rows = report._memory_rows(self.states, 189000)
        current = next(row for row in rows if row['seed'] == 0 and row['round'] == 5)
        self.assertEqual(current['current_response_note_characters'], len('科学note5'))
        self.assertEqual(current['retained_prior_notes_characters'], sum(len('科学note' + str(n)) for n in (2, 3, 4)))
        self.assertEqual(current['canonical_prompt_UTF8_bytes'], 14050)
        self.assertEqual(current['ordinary_wire_prompt_UTF8_bytes'], 14150)
        self.assertEqual(current['ordinary_complete_UTF8_bound_bytes'], 20050)
        self.assertEqual(current['memory_limit_tokens'], 20000)

    def test_incomplete_ledger_and_changed_stage_bound_refused(self):
        path = safe(self.case / 'llm_calls.jsonl'); path.write_bytes(path.read_bytes().rstrip(b'\n'))
        with self.assertRaises(report.ReportIntegrityError): report._call_accounting(self.summary, self.cfg['revision_provenance'])
        self.save_ledger(); self.cfg['memory']['limit_tokens'] = 6000
        with self.assertRaises(report.ReportIntegrityError): report._revision_provenance(self.cfg)
        with self.assertRaises(report.ReportIntegrityError): report._memory_rows(self.states, 188000)

    def test_inherited_numeric_display_and_saved_LOFO_helpers_unchanged(self):
        import ast
        original = ast.parse((ROOT / 'work/h-data/H4/phase4_3/src/l1_report.py').read_text(encoding='utf-8'))
        revised = ast.parse(Path(report.__file__).read_text(encoding='utf-8'))
        for name in ('_number', '_metric', '_score_rows', '_failure', '_top_features', '_plot'):
            left = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
            right = next(node for node in revised.body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False), name)


    def test_admissible_query_only_fallback_is_valid_round_not_provider_success(self):
        request = 'L1R2_seed0_round01'
        original = next(row for row in self.records if row.get('record_type') == 'result' and row['request_id'] == request)
        original.update(status='format_invalid', validated_response_sha256=None, format_round=0)
        identity = {k: original[k] for k in ('request_id', 'prompt_sha256')}; identity['call_index'] = 184
        self.records.extend([{**identity, 'record_type': 'reservation'}, {**identity, 'record_type': 'result',
            'status': 'format_invalid', 'format_round': 1, 'input_tokens': 1000, 'output_tokens': 2000, 'cached_input_tokens': 0}])
        self.summary['cli_attempt_count'] = 184; self.save_ledger()
        write(self.case / 'logical_results' / (request + '.json'), {'request_id': request, 'last_call_index': 184,
            'last_status': 'format_invalid', 'admissible_format_failure': True})
        query_path = self.case / 'query_records' / (request + '.json')
        query = {'request_id': request, 'query_payload': {'queries': [], 'rationale': ''},
            'response_audit': {'query_fallback': True, 'features_preserved': True,
                'notes_preserved': True, 'admissible_format_failure': True}}
        write(query_path, query)
        result = report._call_accounting(self.summary, self.cfg['revision_provenance'])
        self.assertEqual(result['formal_L1_completed_valid_rounds'], 60)
        self.assertEqual(result['formal_L1_successful_requests'], 59)
        self.assertEqual(result['formal_L1_accepted_query_fallback_requests'], 1)
        query['response_audit']['features_preserved'] = False; write(query_path, query)
        with self.assertRaises(report.ReportIntegrityError):
            report._call_accounting(self.summary, self.cfg['revision_provenance'])

if __name__ == '__main__': unittest.main()
