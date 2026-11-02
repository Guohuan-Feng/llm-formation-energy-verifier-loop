"""Read-only preservation/invariant helpers for the authorized L1 revision.

Import is standard-library-only. This module never moves files, writes config,
loads targets, fits a model, executes a feature or calls an LLM. Old scientific
modules are imported lazily only by explicit verification entry points.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True
REVISION = Path(__file__).resolve().parent.parent
PHASE = REVISION.parent
DATA = PHASE.parent
ROOT = DATA.parents[2]
OLD_SRC = PHASE / 'src'
OLD_H4_SRC = DATA / 'src'
ARCHIVE = PHASE / 'voided_original_round1_20261031'
COUNTER_SHA256 = 'f374eab336ae08f9bcb8a58cf0f8346c596ff8eb99e1b37357840f7393e6e99f'
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'cached_input_tokens')
PROMPT_REVISION_KEYS = {'protocol_version', 'memory', 'memory_label', 'memory_explanation',
    'memory_label_sha256', 'memory_explanation_sha256', 'request_namespace', 'history_check',
    'memory_overflow_action', 'full_input_overflow_action', 'memory_or_full_input_overflow_action'}
TOP_LOGISTICAL_KEYS = {'frozen_utc', 'source_hashes', 'input_hashes', 'first_round_requests',
    'prepare_elapsed_seconds', 'revision_provenance', 'revision_accounting'}


class RevisionIntegrityError(ValueError):
    pass


def read_path(path, *, must_exist=True):
    value = Path(path).resolve(strict=must_exist)
    external = {(Path.home() / '.codex' / name).resolve() for name in ('AGENTS.md', 'AGENTS.override.md')}
    if 'onedrive' in str(value).casefold() or not (value.is_relative_to(ROOT.resolve()) or value in external):
        raise RevisionIntegrityError('Read escaped non-OneDrive research inputs/global storage instructions')
    return value


def sha(path):
    digest = hashlib.sha256()
    with read_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(read_path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def differences(before, after, path=()):
    if isinstance(before, dict) and isinstance(after, dict):
        answer = []
        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                answer.append(path + (key,))
            else:
                answer.extend(differences(before[key], after[key], path + (key,)))
        return answer
    return [] if type(before) is type(after) and before == after else [path]


def verify_scientific_invariants(newcfg, originalcfg):
    """Only memory metadata/cap/wording and named logistical records may differ."""
    changes = differences(originalcfg, newcfg)
    disallowed = []
    for path in changes:
        allowed = (path[0] == 'memory' or path[0] in TOP_LOGISTICAL_KEYS
                   or path[0] == 'prompts' and len(path) > 1 and path[1] in PROMPT_REVISION_KEYS)
        if not allowed:
            disallowed.append('.'.join(path))
    if disallowed:
        raise RevisionIntegrityError('Unauthorized scientific/config change: ' + ', '.join(disallowed))
    memory = newcfg['memory']
    cap = memory.get('limit_tokens', memory.get('external_policy_limit_tokens'))
    notes = memory.get('notes_limit_unicode_characters', memory.get('notes_max_unicode_characters'))
    if cap != 20000 or notes != 300 or memory.get('scope') != 'TRAIN_validation_only':
        raise RevisionIntegrityError('Revision memory must be20000 tokens, notes300, TRAIN/validation only')
    if newcfg['prompts'].get('memory') != memory:
        raise RevisionIntegrityError('Prompt and top-level memory policy differ')
    if newcfg['token_counter'] != originalcfg['token_counter']:
        raise RevisionIntegrityError('Tokenizer identity/counter record must remain unchanged')
    if newcfg['prompts']['input_limit'] != originalcfg['prompts']['input_limit']:
        raise RevisionIntegrityError('Whole-request input guard is unchanged by this revision')
    if newcfg['budget']['H4_global_CLI_cap'] != 300 or newcfg['budget']['H4_global_logical_cap'] != 180:
        raise RevisionIntegrityError('Scientific physical300/formal-valid180 caps changed')
    return {'scientific_invariants': 'PASS', 'allowed_config_changes': ['.'.join(path) for path in changes],
            'memory_cap_tokens': 20000, 'old_counter_identity_unchanged': True,
            'whole_request_input_limit_unchanged': True}


def _manifest(value):
    return read(value) if not isinstance(value, dict) else value


def resolve_archived_reference(path, manifest, *, allow_unchanged=False):
    """Resolve old absolute artifact paths through the explicit move/copy table.

    Never silently follow an old phase4_3 canonical run path into the new run.
    Unmoved O/L0 baseline artifacts may be used only with explicit permission.
    """
    record = _manifest(manifest)
    original = read_path(path, must_exist=False)
    for item in record['artifact_mappings']:
        if original == read_path(item['original_path'], must_exist=False):
            return read_path(item['archived_path'])
    archive_root = read_path(record['archive_root'])
    if original.is_relative_to(archive_root):
        return read_path(original)
    prior_roots = (DATA / 'phase4_1', DATA / 'phase4_2')
    if allow_unchanged and any(original.is_relative_to(root.resolve()) for root in prior_roots):
        return read_path(original)
    raise RevisionIntegrityError('Old artifact has no explicit archive mapping: ' + str(original))


def verify_archived_metadata(path, manifest):
    """Check a sealed old record and all its artifact hashes using archive paths."""
    mapped = resolve_archived_reference(path, manifest)
    value = read(mapped)
    payload = dict(value)
    expected = payload.pop('record_sha256', None)
    if expected != hashlib.sha256(canonical(payload).encode('utf-8')).hexdigest():
        raise RevisionIntegrityError('Voided old metadata self-hash changed')
    for ref in value.get('artifacts', {}).values():
        actual = resolve_archived_reference(ref['path'], manifest, allow_unchanged=True)
        if sha(actual) != ref['sha256']:
            raise RevisionIntegrityError('Voided old artifact hash changed')
    return {'status': 'PASS', 'archived_metadata': str(mapped),
            'record_sha256': expected, 'artifacts_checked': len(value.get('artifacts', {}))}


def prefix_matches(ref):
    path = read_path(ref['path'])
    size = ref['size']
    if type(size) is not int or size < 0 or path.stat().st_size < size:
        return False
    h = hashlib.sha256()
    with path.open('rb') as stream:
        remaining = size
        while remaining:
            block = stream.read(min(4 * 1024 * 1024, remaining))
            if not block:
                return False
            h.update(block); remaining -= len(block)
    return h.hexdigest() == ref['sha256']


def verify_void_archive(manifest, *, full=True):
    record = _manifest(manifest)
    archive_root = read_path(record['archive_root'])
    if not archive_root.is_relative_to(PHASE.resolve()) or archive_root == PHASE.resolve():
        raise RevisionIntegrityError('Voided archive must be a separate phase4_3 subdirectory')
    originals = set()
    for item in record['artifact_mappings']:
        before = read_path(item['original_path'], must_exist=False)
        archived = read_path(item['archived_path'])
        if before in originals or not archived.is_relative_to(archive_root):
            raise RevisionIntegrityError('Duplicate original mapping or escaped archive artifact')
        originals.add(before)
        if (item['operation'] not in ('moved', 'copied') or item.get('readonly') is not True
                or archived.stat().st_size != item['size'] or sha(archived) != item['sha256']
                or archived.stat().st_mode & stat.S_IWRITE):
            raise RevisionIntegrityError('Archive bytes/size/readonly/operation changed')
        # These original source files remain canonical and immutable. Config and
        # public snapshots are deliberately replaced by the new revision config.
        if before.is_relative_to(OLD_SRC.resolve()) and (not before.is_file() or sha(before) != item['sha256']):
            raise RevisionIntegrityError('Canonical original L1 scientific source changed')
    for ref in record['prefixes'].values():
        if not prefix_matches(ref):
            raise RevisionIntegrityError('A pre-revision ledger/log/prediction prefix changed')
    original_cfg = archive_root / 'config.json'
    if sha(original_cfg) != record['original_config_sha256']:
        raise RevisionIntegrityError('Original archived config identity changed')
    if len(record['voided_calls']) != 3 or len({r['request_id'] for r in record['voided_calls']}) != 3:
        raise RevisionIntegrityError('Exactly3 distinct old L1 requests must remain explicitly voided')
    return {'void_archive': 'PASS', 'archived_files_checked': len(originals),
            'prefixes_checked': len(record['prefixes']), 'voided_requests': 3,
            'original_config_sha256': record['original_config_sha256']}


def _load_original(filename, name):
    if str(OLD_H4_SRC) not in sys.path:
        sys.path.insert(0, str(OLD_H4_SRC))
    spec = importlib.util.spec_from_file_location(name, read_path(OLD_SRC / filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_prior_stages(record_path=None, *, full=True):
    old = _load_original('l1_common.py', '_revision_readonly_original_common')
    path = PHASE / 'prior_stages_preservation.json' if record_path is None else record_path
    return old.verify_preservation(read(path), full=full)


def _read_ledger(path):
    raw = read_path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise RevisionIntegrityError('Incomplete durable ledger tail')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def ledger_summary(rows, *, void_request_ids, formal_l1_request_ids):
    void_ids, formal_ids = set(void_request_ids), set(formal_l1_request_ids)
    if void_ids & formal_ids:
        raise RevisionIntegrityError('Voided and new-formal request namespaces overlap')
    reservations = [r for r in rows if r.get('record_type') == 'reservation']
    results = [r for r in rows if r.get('record_type') == 'result']
    by_index = {r['call_index']: r for r in reservations}
    if len(by_index) != len(reservations) or len({r['call_index'] for r in results}) != len(results):
        raise RevisionIntegrityError('Duplicate physical reservation/result index')
    for r in results:
        if r['call_index'] not in by_index or by_index[r['call_index']]['request_id'] != r['request_id']:
            raise RevisionIntegrityError('Physical result/reservation identity mismatch')
    all_ids = {r['request_id'] for r in reservations}
    successes = [r for r in results if r.get('status') == 'ok']
    success_ids = {r['request_id'] for r in successes}
    if len(success_ids) != len(successes):
        raise RevisionIntegrityError('A successful request was dispatched again')
    def scope_tokens(ids):
        chosen = [r for r in results if ids is None or r['request_id'] in ids]
        answer = {'result_count': len(chosen), 'unknown_token_results': 0,
                  'unknown_token_fields': {}, 'token_totals_are_known_values_only': True}
        for field in TOKEN_FIELDS:
            known = [r[field] for r in chosen if r.get(field) is not None]
            if any(type(value) is not int or value < 0 for value in known):
                raise RevisionIntegrityError('Token counts must be nonnegative integers or unknown')
            answer[field + '_known_sum'] = sum(known)
            answer[field] = sum(known) if known or not chosen else None
            answer['unknown_token_fields'][field] = len(chosen) - len(known)
        answer['unknown_token_results'] = sum(any(r.get(field) is None for field in TOKEN_FIELDS) for r in chosen)
        return answer
    completed = {r['call_index'] for r in results}
    return {'global_physical_cli_attempts': len(reservations), 'global_completed_cli_attempts': len(results),
        'unresolved_physical_reservations': len(set(by_index) - completed),
        'global_physical_logical_requests': len(all_ids), 'formal_valid_logical_requests': len(all_ids - void_ids),
        'voided_logical_requests': len(all_ids & void_ids),
        'voided_cli_attempts': sum(r['request_id'] in void_ids for r in reservations),
        'formal_L1_logical_requests': len(all_ids & formal_ids),
        'formal_L1_successful_requests': len(success_ids & formal_ids),
        'formal_L1_cli_attempts': sum(r['request_id'] in formal_ids for r in reservations),
        'all_L1_cli_attempts': sum(r['request_id'] in void_ids | formal_ids for r in reservations),
        'tokens_by_scope': {'global': scope_tokens(None), 'voided_L1': scope_tokens(void_ids),
            'formal_L1': scope_tokens(formal_ids), 'all_L1': scope_tokens(void_ids | formal_ids)}}


def verify_revision(cfg=None, *, full=True, manifest_path=None):
    cfg = read(PHASE / 'config.json') if cfg is None else cfg
    frozen = read(PHASE / 'frozen_config.json')
    if sha(PHASE / 'config.json') != frozen['sha256']:
        raise RevisionIntegrityError('New canonical config differs from its new frozen hash')
    if read(PHASE / 'config.json') != cfg or (ROOT / 'H4_feature_loop/config.json').read_bytes() != (PHASE / 'config.json').read_bytes():
        raise RevisionIntegrityError('Private/published/config-argument revision differs')
    provenance = cfg['revision_provenance']
    manifest_path = provenance['archive_manifest_path'] if manifest_path is None else manifest_path
    record = read(manifest_path)
    original_path = provenance.get('original_config_path', str(Path(record['archive_root']) / 'config.json'))
    originalcfg = read(original_path)
    invariants = verify_scientific_invariants(cfg, originalcfg)
    amendment = provenance['amendment_path']
    if sha(amendment) != frozen['amendment_sha256']:
        raise RevisionIntegrityError('New memory amendment differs from its frozen sidecar hash')
    for section in ('source_hashes', 'input_hashes', 'global_context_hashes'):
        for path, expected in cfg[section].items():
            if sha(path) != expected:
                raise RevisionIntegrityError('Frozen revision source/input/context changed: ' + Path(path).name)
    archive = verify_void_archive(record, full=full)
    prior = verify_prior_stages(full=full)
    counter = _load_original('l1_counter.py', '_revision_readonly_original_counter').verify()
    if counter['token_counter_sha256'] != COUNTER_SHA256 or counter['limit_tokens'] != 6000:
        raise RevisionIntegrityError('Historical tokenizer record must remain unchanged')
    accounting = cfg['revision_accounting']
    ids = accounting['expected_request_ids']
    if (len(ids) != 60 or len(set(ids)) != 60 or accounting['physical_unique_request_cap'] != 183
            or accounting['global_before'] != 123 or accounting['voided'] != 3 or accounting['active_formal'] != 60):
        raise RevisionIntegrityError('Revision physical/valid/logical accounting differs')
    void_ids = {r['request_id'] for r in record['voided_calls']}
    rows = _read_ledger(ROOT / 'H4_feature_loop/llm_calls.jsonl')
    ledger = ledger_summary(rows, void_request_ids=void_ids, formal_l1_request_ids=ids)
    if (ledger['global_physical_cli_attempts'] > 300 or ledger['global_physical_logical_requests'] > 183
            or ledger['formal_valid_logical_requests'] > 180 or ledger['voided_logical_requests'] != 3):
        raise RevisionIntegrityError('Physical300/unique183/valid180 budgets were violated')
    out = ROOT / 'H4_feature_loop'
    files = {'config.json', 'predictions.csv', 'llm_calls.jsonl', 'metrics.json', 'run.log', 'README.md'}
    if {p.name for p in out.iterdir()} != files or len((out / 'README.md').read_text(encoding='utf-8').splitlines()) > 20:
        raise RevisionIntegrityError('Public six-file/README20 boundary changed')
    return {'status': 'PASS', **invariants, **archive, **prior,
            'counter_identity': counter, 'ledger': ledger, 'old_voided_science_not_reused': True}
