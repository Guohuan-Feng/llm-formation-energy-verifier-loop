"""Synthetic saved-artifact flow tests; no real TEST target or scientific operation."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / 'work/h-data/H4/phase4_3/manual_terminal_resume_20261031/src'))
import score_l1_terminal as terminal
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np

BASE = ROOT / 'work/h-data/H4/phase4_3/manual_terminal_resume_20261031/tests/software_terminal'


def safe(path):
    result = Path(path).resolve()
    if 'onedrive' in str(result).casefold() or not result.is_relative_to(BASE.resolve()):
        raise ValueError('Synthetic fixtures must remain in the new manual-terminal tests directory')
    return result


def raw(path, value):
    p = safe(path); safe(p.parent).mkdir(parents=True, exist_ok=True)
    p.write_bytes(value.encode('utf-8') if isinstance(value, str) else value)
    return p


def write(path, value, *, seal=False):
    if seal:
        value = {**value, 'record_sha256': hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()}
    raw(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.case = safe(BASE / uuid.uuid4().hex); self.phase = self.case / 'phase4_3'
        self.manual = self.phase / 'manual_terminal_resume_20261031'; self.public = self.case / 'public'
        self.snapshot = self.phase / 'stage4_2_public_snapshot'; self.oconfig = self.case / 'phase4_2/stage4_1_public_snapshot/config.json'
        self.post_metadata = {self.phase / 'memory_content_amendment_L1.md', self.phase / 'prior_stages_preservation_current.json'}
        for name, value in [('PHASE', self.phase), ('MANUAL', self.manual), ('PUBLIC', self.public),
            ('REVISION', self.phase / 'revision'), ('SNAPSHOT', self.snapshot), ('O_CONFIG', self.oconfig), ('POST_METADATA', self.post_metadata)]:
            p = patch.object(terminal, name, value); p.start(); self.addCleanup(p.stop)
        self.ids = [f'MOCK_TEST{i:04d}' for i in range(1000)]
        self.pool = [f'MOCK_POOL{i:04d}' for i in range(2000)]
        source = raw(self.case / 'mock_frozen_numerical_source.py', '# synthetic frozen source marker, never imported\n')
        self.cfg = {'data': {'test_ids': self.ids, 'perseed': {str(seed): {'S0_ids': self.pool[:500]} for seed in range(3)}},
            'source_hashes': {str(source.resolve()): sha(source)},
            'statistics': {'marker_contract': {'stage': '4.3', 'condition': 'L1', 'current_L1_scoring_pass': 1}},
            'revision_provenance': {'revision_utc': 'SYNTHETIC'}}
        write(self.phase / 'config.json', self.cfg)
        write(self.phase / 'frozen_config.json', {'sha256': sha(self.phase / 'config.json')})
        oldO = {'stage': '4.1', 'condition': 'O', 'SYNTHETIC_saved_value': 0.21}
        oldL0 = {'stage': '4.2', 'condition': 'L0', 'SYNTHETIC_saved_value': 0.20}
        write(self.phase.parent / 'phase4_1/terminal_results.json', oldO)
        write(self.phase.parent / 'phase4_2/terminal_results.json', oldL0)
        write(self.snapshot / 'metrics.json', {'prior_stage4_1_results': str(self.oconfig.with_name('metrics.json').resolve()), 'results': oldL0})
        ocfg = write(self.oconfig, {'stage': '4.1', 'synthetic': 'preserved O'})
        lcfg = write(self.snapshot / 'config.json', {'stage': '4.2', 'synthetic': 'preserved L0'})
        old_O_metrics = self.oconfig.with_name('metrics.json')
        write(old_O_metrics, {'results': oldO})
        manifest = self.phase / 'prior_stages_preservation.json'
        write(manifest, {'phase4_2': {str(old_O_metrics.resolve()): sha(old_O_metrics)}})
        self.cfg['input_hashes'] = {str(manifest.resolve()): sha(manifest)}
        write(self.phase / 'config.json', self.cfg)
        write(self.phase / 'frozen_config.json', {'sha256': sha(self.phase / 'config.json')})
        write(self.public / 'config.json', {**self.cfg, 'preserved_stage_configs': {'O': ocfg, 'L0': lcfg}})
        write(self.public / 'metrics.json', {'status': 'prepared', 'prior_stage4_1_results': str(old_O_metrics.resolve()),
            'prior_stage4_2_results': oldL0, 'preserved_condition_results': {'O': oldO, 'L0': oldL0}})
        lines = ['# Synthetic preserved revision README', 'Status: prepared; stage4.4 waits for explicit continue.']
        lines += [f'Preserved original explanation{i}: top30/latest3, notes300, memory20000, voided3.' for i in range(14)]
        lines += ['Actual TEST scoring passes O=1,L0=1,L1=0; totalH4=2; oneL1 accessor covers all3seeds if reached.', 'Original rule18.', 'Original forecast19.']
        self.original_readme = ('\n'.join(lines) + '\n').encode()
        raw(self.public / 'README.md', self.original_readme)
        for name in ('predictions.csv', 'llm_calls.jsonl', 'run.log'):
            raw(self.snapshot / name, 'SYNTHETIC old ' + name + '\n')
            raw(self.public / name, (self.snapshot / name).read_bytes() + b'SYNTHETIC already-committed appended bytes\n')
        raw(self.snapshot / 'README.md', 'SYNTHETIC previous-stage README\n')
        self.summary = {'cli_attempt_count': 183, 'completed_cli_attempt_count': 183,
            'revision_accounting': {'revised_successes': 60}}
        write(self.phase / 'run_stop.json', {'status': 'failed', 'elapsed_seconds': 9554.234, 'transport': self.summary})
        write(self.phase / 'selected_rounds.json', ['SYNTHETIC'] * 3)
        write(self.phase / 'selection_pre_fill.json', {'SYNTHETIC': True})
        for condition in ('phase4_1', 'phase4_2'):
            write(self.phase.parent / condition / 'evaluation_started.json', {'SYNTHETIC': True})
            raw(self.phase.parent / condition / 'private_pairing_cache.npz', b'SYNTHETIC cached draws, never scored')
        for path in self.post_metadata:
            raw(path, 'SYNTHETIC pre-completion preservation metadata\n')
        self.arms = []
        for seed in range(3):
            state = {'seed': seed, 'round_completed': 20, 'labels': {rid: .0 for rid in self.pool},
                'history': [{}] * 20, 'round_model_paths': ['SYNTHETIC'] * 20,
                'query_history': [{}] * 20, 'memory_audits': [{}] * 20}
            write(self.phase / f'states/seed{seed}.json', state)
            prediction = raw(self.phase / f'final/L1_seed{seed}/predictions.npz', b'SYNTHETIC already-computed predictions')
            write(self.phase / f'final/L1_seed{seed}/final_predictions.json', {'status': 'complete', 'seed': seed,
                'train_count': 2000, 'test_count': 1000, 'TEST_target_values_read': False, 'TEST_scoring_performed': False,
                'artifacts': {'predictions': {'path': str(prediction.resolve()), 'sha256': sha(prediction)}}}, seal=True)
            self.arms.append({'arm_id': f'L1_seed{seed}', 'condition': 'L1', 'seed': seed, 'status': 'complete',
                'prediction_rows': [{'id': rid, 'prediction_eV_per_atom': .5} for rid in self.ids]})
        write(self.phase / 'committed_prediction_arms.json', self.arms)
        steps = {}
        for n in range(1, 5):
            receipt = self.manual / f'step{n}.json'; write(receipt, {'status': 'PASS', 'step': n, 'synthetic': True})
            steps[str(n)] = {'status': 'PASS', 'receipt_path': str(receipt.resolve())}
        self.gate_path = self.manual / 'authorization_gate.json'
        self.gate = {'status': 'PASS', 'stage': '4.3', 'action': 'terminal_score_once', 'authorized_by_user': True,
            'steps': steps, 'artifacts': {str(p.resolve()): sha(p) for p in self.case.rglob('*') if p.is_file()},
            'report_appendix': '## SYNTHETIC explicit public-update approval\nApproved gate appendix only.',
            'post_completion_updates_allowed': [str(p.resolve()) for p in self.post_metadata]}
        self.gate['artifacts'][str(Path(terminal.__file__).resolve())] = sha(terminal.__file__)
        backup = Path(terminal.__file__).with_name('score_l1_terminal_before_reference_fix.py').resolve()
        self.gate['artifacts'][str(backup)] = sha(backup)
        self.save_gate()
        self.events = []; self.evaluations = 0; self.pair_failure = False
        self.fake_data = SimpleNamespace(_TRUTH_READ=False, load_bundle=lambda: {'test_ids': self.ids})
        def normalise(arms, ids):
            self.assertEqual(ids, self.ids); self.assertEqual(len(arms), 3)
        def evaluate(arms, bundle, *, evaluation_marker):
            self.assertTrue(Path(evaluation_marker).exists()); self.events.append('evaluate'); self.evaluations += 1
            self.fake_data._TRUTH_READ = True
            return {'stage': '4.3', 'condition': 'L1', 'TEST_target_accessor_calls': 1,
                'current_L1_scoring_pass': 1, 'prior_H4_TEST_scoring_passes': 2,
                'evaluation_marker': {'sha256': sha(evaluation_marker)}, 'synthetic_no_real_scoring': True}
        def cached():
            self.events.append('cache'); return {'SYNTHETIC': np.asarray([1., 2.])}
        def pairs(caches, *, l1_evaluation_marker):
            self.events.append('pairs')
            self.assertTrue((self.phase / 'terminal_results.json').exists())
            self.assertTrue((self.phase / 'private_pairing_cache.npz').exists())
            self.assertEqual(json.loads((self.public / 'metrics.json').read_text())['current_L1_scoring_pass'], 1)
            if self.pair_failure:
                return {'registered_L0_minus_O_MAE_check': {'status': 'FAIL'}}
            return {'registered_L0_minus_O_MAE_check': {'status': 'PASS'}, 'comparisons': 'SYNTHETIC no subtraction'}
        self.fake_stats = SimpleNamespace(_normalise=normalise, evaluate=evaluate,
            private_pairing_cache=cached, paired_comparisons=pairs)
        def make_report(*args):
            self.events.append('report'); p = raw(self.phase / 'report/H4_stage4_3_report.md', '# SYNTHETIC completed report\n')
            return {'report': str(p)}
        self.fake_report = SimpleNamespace(make_report=make_report)
        self.deps_patch = patch.object(terminal, '_dependencies', return_value=(self.fake_data, self.fake_stats, self.fake_report))
        self.deps = self.deps_patch.start(); self.addCleanup(self.deps_patch.stop)

    def save_gate(self):
        write(self.gate_path, self.gate)

    def score(self):
        return terminal.score(gate_path=self.gate_path)

    def test_once_order_commits_public_results_then_pairs_and_final_hashes(self):
        unchanged = {name: (self.public / name).read_bytes() for name in ('config.json', 'predictions.csv', 'llm_calls.jsonl', 'run.log')}
        result = self.score(); self.assertEqual(result['status'], 'complete')
        self.assertEqual(self.events, ['evaluate', 'cache', 'pairs', 'report'])
        self.assertEqual(self.evaluations, 1)
        audit = json.loads((self.manual / 'report_appendix_audit.json').read_text())
        report_path = self.phase / 'report/H4_stage4_3_report.md'
        self.assertEqual(audit['final_report_sha256'], sha(report_path))
        self.assertNotEqual(audit['gate_appendix_stage_report_sha256'], audit['final_report_sha256'])
        for name, value in audit['public_file_old_new_hashes'].items():
            self.assertEqual(value['new_sha256'], sha(self.public / name))
        self.assertIn('Public file hashes after terminal completion', report_path.read_text())
        m = json.loads((self.public / 'metrics.json').read_text()); self.assertEqual(m['prior_stage4_1_results'], str(self.oconfig.with_name('metrics.json').resolve()))
        self.assertEqual(m['preserved_condition_results']['O']['SYNTHETIC_saved_value'], .21)
        self.assertEqual(m['preserved_condition_results']['L0']['SYNTHETIC_saved_value'], .20)
        self.assertEqual(m['prior_stage4_2_results']['SYNTHETIC_saved_value'], .20)
        for name, value in unchanged.items(): self.assertEqual((self.public / name).read_bytes(), value)
        lines = (self.public / 'README.md').read_bytes().splitlines(); original = self.original_readme.splitlines()
        self.assertEqual(len(lines), 20)
        for i in range(19):
            if i not in (1, 16): self.assertEqual(lines[i], original[i])
        self.assertIn(b'L1=1; totalH4=3;', lines[16])
        verification = terminal.verify(gate_path=self.gate_path)
        self.assertEqual(verification['status'], 'PASS'); self.assertEqual(self.evaluations, 1)
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 1)

    def test_failed_step_or_false_receipt_refuses_before_any_dependency_or_marker(self):
        self.gate['steps']['4']['status'] = 'FAIL'; self.save_gate()
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 0); self.assertFalse((self.phase / 'evaluation_started.json').exists())
        self.gate['steps']['4']['status'] = 'PASS'
        receipt = Path(self.gate['steps']['4']['receipt_path']); write(receipt, {'status': 'FAIL'})
        self.gate['artifacts'][str(receipt)] = sha(receipt); self.save_gate()
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.deps.assert_not_called()

    def test_changed_committed_prediction_or_existing_gate_cannot_score(self):
        raw(self.phase / 'committed_prediction_arms.json', 'changed')
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 0)

    def test_existing_marker_blocks_a_first_or_second_truth_access(self):
        write(self.phase / 'evaluation_started.json', {'SYNTHETIC_existing_gate': True})
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 0)

    def test_paired_reference_failure_preserves_once_scored_result_and_stops(self):
        self.pair_failure = True; result = self.score()
        self.assertEqual(result['status'], 'failed'); self.assertTrue(result['L1_TEST_truth_accessor_consumed'])
        self.assertTrue((self.phase / 'terminal_results.json').exists()); self.assertTrue((self.phase / 'private_pairing_cache.npz').exists())
        self.assertFalse((self.phase / 'paired_comparisons.json').exists()); self.assertEqual(self.evaluations, 1)
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 1)
        self.assertEqual(terminal.verify(gate_path=self.gate_path)['terminal_state'], 'failed')

    def test_approved_post_metadata_updates_do_not_reopen_truth_or_change_public(self):
        self.assertEqual(self.score()['status'], 'complete')
        for path in self.post_metadata: raw(path, 'SYNTHETIC approved final metadata update')
        self.assertEqual(terminal.verify(gate_path=self.gate_path)['status'], 'PASS')
        self.assertEqual(self.evaluations, 1)

    def test_public_config_archive_block_is_semantically_checked(self):
        public = json.loads((self.public / 'config.json').read_text()); public['preserved_stage_configs']['O']['scientific_change'] = True
        write(self.public / 'config.json', public)
        self.gate['artifacts'][str((self.public / 'config.json').resolve())] = sha(self.public / 'config.json'); self.save_gate()
        with self.assertRaises(terminal.TerminalContinuationError): self.score()
        self.assertEqual(self.evaluations, 0)

    def test_verify_ready_does_not_import_statistics_or_truth_accessor(self):
        self.assertEqual(terminal.verify(gate_path=self.gate_path)['terminal_state'], 'not_started')
        self.deps.assert_not_called(); self.assertEqual(self.evaluations, 0)



    def test_real_legacy_reference_schema_resolves_only_protected_snapshot(self):
        snapshot = json.loads((self.snapshot / 'metrics.json').read_text())
        self.assertIs(type(snapshot['prior_stage4_1_results']), str)
        O, L0 = terminal._old_results()
        self.assertEqual(O['condition'], 'O'); self.assertEqual(L0['condition'], 'L0')
        arbitrary = self.case / 'arbitrary_O_metrics.json'; write(arbitrary, {'results': O})
        snapshot['prior_stage4_1_results'] = str(arbitrary.resolve()); write(self.snapshot / 'metrics.json', snapshot)
        with self.assertRaises(terminal.TerminalContinuationError): terminal._old_results()
        self.assertEqual(self.evaluations, 0); self.deps.assert_not_called()

if __name__ == '__main__': unittest.main()
