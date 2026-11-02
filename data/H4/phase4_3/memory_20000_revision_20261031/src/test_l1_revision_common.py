"""Pure CPU verification tests; no scientific module/API/model/target access."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import l1_revision_common as c


def configs():
    original = {'protocol': 'same', 'runtime': {'model': 'gpt-6.1-sol', 'reasoning': 'high'},
        'data': {'S0_ids': ['POOL-only'], 'mapping': 'fixed'}, 'models': {'rate': .05},
        'statistics': {'bootstrap_n': 2000}, 'protocol_files': {'schema': 'unchanged'},
        'budget': {'H4_global_CLI_cap': 300, 'H4_global_logical_cap': 180, 'repairs': 1},
        'token_counter': {'record': 'same', 'legacy_limit': 6000},
        'memory': {'limit_tokens': 6000}, 'prompts': {'input_limit': 189000, 'prefix': 'unchanged',
            'schema': {'maxLength': 20000}, 'instructions': 'same', 'memory': {'limit_tokens': 6000},
            'memory_label': 'old', 'memory_explanation': 'old'}}
    revised = deepcopy(original)
    revised['memory'] = {'limit_tokens': 20000, 'external_policy_limit_tokens': 20000,
        'notes_limit_unicode_characters': 300, 'scope': 'TRAIN_validation_only', 'stop_scope': 'seed'}
    revised['prompts'].update(memory=deepcopy(revised['memory']), memory_label='new metadata',
        memory_explanation='new metadata policy', request_namespace='new-R2')
    revised.update(revision_provenance={'mock': True}, revision_accounting={'mock': True})
    return original, revised


class Tests(unittest.TestCase):
    def setUp(self):
        root = (c.REVISION / 'software_common').resolve()
        assert 'onedrive' not in str(root).casefold() and root.is_relative_to(c.REVISION)
        root.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=root); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def write(self, path, content, readonly=False):
        path = Path(path).resolve()
        assert 'onedrive' not in str(path).casefold() and path.is_relative_to(self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict): content = (json.dumps(content) + '\n').encode()
        path.write_bytes(content)
        if readonly: path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
        return path

    def test_exact_allowed_memory_and_wording_changes(self):
        before, after = configs()
        audit = c.verify_scientific_invariants(after, before)
        self.assertEqual(audit['scientific_invariants'], 'PASS')
        self.assertTrue(audit['old_counter_identity_unchanged'])

    def test_numeric_data_schema_instructions_and_whole_input_changes_rejected(self):
        before, after = configs()
        for section, key, value in [('models', 'rate', .1), ('data', 'mapping', 'wrong'),
                                    ('statistics', 'bootstrap_n', 10), ('prompts', 'input_limit', 190000),
                                    ('prompts', 'instructions', 'wrong'), ('runtime', 'reasoning', 'low')]:
            wrong = deepcopy(after);wrong[section][key] = value
            with self.subTest(section=section, key=key):
                with self.assertRaises(c.RevisionIntegrityError): c.verify_scientific_invariants(wrong, before)
        after['prompts']['schema']['maxLength'] = 34000
        with self.assertRaises(c.RevisionIntegrityError): c.verify_scientific_invariants(after, before)

    def test_counter_identity_and_notes_cap_remain_frozen(self):
        before, after = configs();after['token_counter']['legacy_limit'] = 20000
        with self.assertRaises(c.RevisionIntegrityError): c.verify_scientific_invariants(after, before)
        before, after = configs();after['memory']['notes_limit_unicode_characters'] = 301
        after['prompts']['memory'] = deepcopy(after['memory'])
        with self.assertRaises(c.RevisionIntegrityError): c.verify_scientific_invariants(after, before)

    def fixture_archive(self):
        canonical = self.root / 'canonical'
        archive = self.root / 'voided'
        original_artifact = canonical / 'round01/train.npz'
        original_meta = canonical / 'round01/round.json'
        blob = b'original CPU fake feature cache'
        old_sha = hashlib.sha256(blob).hexdigest()
        payload = {'status': 'complete', 'artifacts': {'train': {'path': str(original_artifact), 'sha256': old_sha}}}
        payload['record_sha256'] = hashlib.sha256(c.canonical(payload).encode()).hexdigest()
        stored = [(original_artifact, archive / 'round01/train.npz', blob),
                  (original_meta, archive / 'round01/round.json', (json.dumps(payload) + '\n').encode()),
                  (self.root / 'old_config.json', archive / 'config.json', b'{"original":true}\n')]
        mappings = []
        for original, archived, raw in stored:
            self.write(archived, raw, readonly=True)
            self.write(original, b'NEW canonical run must never be used for the old artifact')
            mappings.append({'original_path': str(original), 'archived_path': str(archived),
                'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw), 'readonly': True, 'operation': 'moved'})
        ledger = self.write(self.root / 'public/ledger.jsonl', b'original-prefix\nnew-tail\n')
        record = {'version': 'CPU test', 'archive_root': str(archive),
            'original_config_sha256': c.sha(archive / 'config.json'), 'artifact_mappings': mappings,
            'voided_calls': [{'request_id': 'void' + str(i)} for i in range(3)],
            'prefixes': {'ledger': {'path': str(ledger), 'size': len(b'original-prefix\n'),
                                   'sha256': hashlib.sha256(b'original-prefix\n').hexdigest()}}}
        return record, original_meta

    def test_old_metadata_artifacts_resolve_to_archive_not_new_canonical_files(self):
        record, original = self.fixture_archive()
        audit = c.verify_archived_metadata(original, record)
        self.assertEqual(audit['artifacts_checked'], 1)
        self.assertTrue(audit['archived_metadata'].startswith(record['archive_root']))
        self.assertEqual(c.verify_void_archive(record)['voided_requests'], 3)
        with self.assertRaises(c.RevisionIntegrityError):
            c.resolve_archived_reference(self.root / 'unmapped.npz', record)

    def test_archive_readonly_flag_and_prefix_tamper_fail_closed(self):
        record, _ = self.fixture_archive()
        item = record['artifact_mappings'][0]
        path = Path(item['archived_path']);path.chmod(path.stat().st_mode | stat.S_IWRITE)
        with self.assertRaises(c.RevisionIntegrityError): c.verify_void_archive(record)
        path.chmod(path.stat().st_mode & ~stat.S_IWRITE)
        self.write(record['prefixes']['ledger']['path'], b'Xriginal-prefix\nnew-tail\n')
        with self.assertRaises(c.RevisionIntegrityError): c.verify_void_archive(record)

    def test_physical_void_formal_repairs_and_unknown_tokens_are_separate(self):
        rows = []
        for index, (rid, status, tokens) in enumerate([
            ('old', 'ok', (10, 2, 0)), ('void', 'ok', (11, None, None)),
            ('new', 'format_invalid', (12, 3, 0)), ('new', 'ok', (13, 4, 1))], 1):
            rows.append({'record_type': 'reservation', 'call_index': index, 'request_id': rid})
            rows.append({'record_type': 'result', 'call_index': index, 'request_id': rid, 'status': status,
                         **dict(zip(c.TOKEN_FIELDS, tokens))})
        summary = c.ledger_summary(rows, void_request_ids={'void'}, formal_l1_request_ids={'new'})
        self.assertEqual(summary['global_physical_cli_attempts'], 4)
        self.assertEqual(summary['global_physical_logical_requests'], 3)
        self.assertEqual(summary['formal_valid_logical_requests'], 2)
        self.assertEqual(summary['formal_L1_cli_attempts'], 2)
        self.assertEqual(summary['formal_L1_successful_requests'], 1)
        self.assertIsNone(summary['tokens_by_scope']['voided_L1']['output_tokens'])
        self.assertEqual(summary['tokens_by_scope']['voided_L1']['unknown_token_fields']['output_tokens'], 1)
        self.assertEqual(summary['tokens_by_scope']['formal_L1']['input_tokens_known_sum'], 25)

    def test_replayed_success_and_namespace_overlap_are_rejected(self):
        with self.assertRaises(c.RevisionIntegrityError):
            c.ledger_summary([], void_request_ids={'same'}, formal_l1_request_ids={'same'})
        rows = []
        for index in (1, 2):
            rows += [{'record_type': 'reservation', 'call_index': index, 'request_id': 'same'},
                     {'record_type': 'result', 'call_index': index, 'request_id': 'same', 'status': 'ok'}]
        with self.assertRaises(c.RevisionIntegrityError):
            c.ledger_summary(rows, void_request_ids=set(), formal_l1_request_ids={'same'})


if __name__ == '__main__': unittest.main(verbosity=2)
