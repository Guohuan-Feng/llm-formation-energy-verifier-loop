"""One explicitly authorized terminal-only L1 continuation; never train or call LLMs.

Import has no scientific side effects. Only score() may invoke the unchanged
L1 statistics, once after four local receipts and exact committed-prediction
hashes pass. verify() never loads targets or recomputes any metric or interval.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[6]
PHASE = ROOT / 'work/h-data/H4/phase4_3'
MANUAL = PHASE / 'manual_terminal_resume_20261031'
PUBLIC = ROOT / 'H4_feature_loop'
REVISION = PHASE / 'memory_20000_revision_20261031'
SNAPSHOT = PHASE / 'stage4_2_public_snapshot'
O_CONFIG = PHASE.parent / 'phase4_2/stage4_1_public_snapshot/config.json'
META_OUTPUTS = {'evaluation_started.json', 'terminal_results.json', 'private_pairing_cache.npz', 'paired_comparisons.json'}
POST_METADATA = {PHASE / 'memory_content_amendment_L1.md', PHASE / 'prior_stages_preservation_current.json'}
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
import re
import time
import traceback
import uuid


class TerminalContinuationError(RuntimeError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read_path(path):
    result = Path(path).resolve(strict=True)
    global_instructions = {(Path.home() / '.codex' / name).resolve() for name in ('AGENTS.md', 'AGENTS.override.md')}
    if 'onedrive' in str(result).casefold() or not (result.is_relative_to(ROOT.resolve()) or result in global_instructions):
        raise TerminalContinuationError('Read escaped the authorized non-OneDrive research inputs')
    return result


def _output(path):
    result = Path(path).resolve()
    allowed = (result.is_relative_to(MANUAL.resolve()) or result.is_relative_to((PHASE / 'report').resolve())
        or result.parent == PHASE.resolve() and result.name in META_OUTPUTS
        or result in {(PUBLIC / 'metrics.json').resolve(), (PUBLIC / 'README.md').resolve()})
    if 'onedrive' in str(result).casefold() or not allowed:
        raise TerminalContinuationError('Terminal continuation may write only new terminal/report outputs and approved metrics/README')
    return result


def _read(path):
    return json.loads(_read_path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    result = hashlib.sha256()
    with _read_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def _write(path, content, *, exclusive=False):
    destination = _output(path)
    _output(destination.parent if destination.is_relative_to(MANUAL.resolve()) or destination.is_relative_to((PHASE / 'report').resolve()) else MANUAL).mkdir(parents=True, exist_ok=True)
    payload = content.encode('utf-8') if isinstance(content, str) else content
    if exclusive:
        with destination.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        return
    temp = _output(MANUAL / 'atomic' / (uuid.uuid4().hex + '.tmp'))
    _output(temp.parent).mkdir(parents=True, exist_ok=True)
    try:
        with temp.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, _output(destination))
    finally:
        if temp.exists():
            temp.unlink()


def _json_write(path, value, *, exclusive=False):
    return _write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n', exclusive=exclusive)


def _npz_write(path, arrays):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write(path, buffer.getvalue(), exclusive=True)


def _sealed(path):
    result = _read(path); raw = dict(result); expected = raw.pop('record_sha256', None)
    actual = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if expected != actual:
        raise TerminalContinuationError('Existing model metadata self-hash changed')
    return result


def _old_results():
    snapshot = _read(SNAPSHOT / 'metrics.json')
    O, L0 = snapshot['prior_stage4_1_results'], snapshot['results']
    if (O != _read(PHASE.parent / 'phase4_1/terminal_results.json')
        or L0 != _read(PHASE.parent / 'phase4_2/terminal_results.json')):
        raise TerminalContinuationError('The readonly O/L0 snapshot and committed terminal results differ')
    return O, L0


def _public_config(cfg):
    public = _read(PUBLIC / 'config.json'); preserved = public.pop('preserved_stage_configs', None)
    if (public != cfg or not isinstance(preserved, dict) or set(preserved) != {'O', 'L0'}
        or preserved['O'] != _read(O_CONFIG) or preserved['L0'] != _read(SNAPSHOT / 'config.json')):
        raise TerminalContinuationError('Published config must be unchanged science plus the approved complete O/L0 configuration snapshots')


def _prediction_structure(arms, cfg):
    ids = cfg['data']['test_ids']; expected = set(ids)
    if len(ids) != 1000 or len(expected) != 1000 or len(arms) != 3:
        raise TerminalContinuationError('Exactly3 already committed complete TEST1000 prediction arms are required')
    seeds = set()
    for arm in arms:
        seed = arm.get('seed'); rows = arm.get('prediction_rows', [])
        if (type(seed) is not int or seed not in (0, 1, 2) or seed in seeds
            or arm.get('condition') != 'L1' or arm.get('arm_id') != f'L1_seed{seed}' or arm.get('status') != 'complete'
            or len(rows) != 1000 or {row.get('id') for row in rows} != expected
            or any(type(row.get('prediction_eV_per_atom')) not in (int, float)
                or not math.isfinite(row['prediction_eV_per_atom']) for row in rows)):
            raise TerminalContinuationError('An existing prediction arm is incomplete, nonfinite or has changed identities')
        seeds.add(seed)


def _validate_gate(gate_path, *, post=False):
    gate_path = _read_path(gate_path)
    if gate_path != (MANUAL / 'authorization_gate.json').resolve():
        raise TerminalContinuationError('Only this explicit terminal-continuation authorization gate is accepted')
    gate = _read(gate_path)
    if (gate.get('status') != 'PASS' or gate.get('stage') != '4.3'
        or gate.get('action') != 'terminal_score_once' or gate.get('authorized_by_user') is not True
        or set(gate.get('steps', {})) != {'1', '2', '3', '4'}
        or any(gate['steps'][str(n)].get('status') != 'PASS' for n in range(1, 5))
        or type(gate.get('report_appendix')) is not str or not gate['report_appendix'].strip()):
        raise TerminalContinuationError('All four ordered local checks and the explicit user authorization must pass first')
    artifacts = {_read_path(path): digest for path, digest in gate.get('artifacts', {}).items()}
    permitted_post = {_read_path(path) for path in gate.get('post_completion_updates_allowed', [])}
    if not permitted_post <= {path.resolve() for path in POST_METADATA}:
        raise TerminalContinuationError('Only the two explicitly approved final amendment/manifest metadata updates may follow completion')
    mutable_public = {(PUBLIC / name).resolve() for name in ('metrics.json', 'README.md')}
    for path, expected in artifacts.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise TerminalContinuationError('Every gate artifact must bind its full SHA256')
        if post and path in mutable_public | permitted_post:
            continue
        if _sha(path) != expected:
            raise TerminalContinuationError('A checked input changed after the four-step gate: ' + str(path))
    cfg = _read(PHASE / 'config.json'); frozen = _read(PHASE / 'frozen_config.json')
    if frozen['sha256'] != _sha(PHASE / 'config.json'):
        raise TerminalContinuationError('The private scientific configuration differs from its original frozen record')
    _public_config(cfg)
    required = {PHASE / 'config.json', PHASE / 'frozen_config.json', PHASE / 'run_stop.json',
        PHASE / 'committed_prediction_arms.json', PHASE / 'selected_rounds.json', PHASE / 'selection_pre_fill.json',
        PUBLIC / 'config.json', PUBLIC / 'metrics.json', PUBLIC / 'README.md', PUBLIC / 'predictions.csv', PUBLIC / 'llm_calls.jsonl', PUBLIC / 'run.log',
        O_CONFIG, SNAPSHOT / 'config.json', SNAPSHOT / 'metrics.json',
        PHASE.parent / 'phase4_1/terminal_results.json', PHASE.parent / 'phase4_2/terminal_results.json',
        PHASE.parent / 'phase4_1/evaluation_started.json', PHASE.parent / 'phase4_2/evaluation_started.json',
        PHASE.parent / 'phase4_1/private_pairing_cache.npz', PHASE.parent / 'phase4_2/private_pairing_cache.npz',
        Path(__file__)}
    required |= {SNAPSHOT / name for name in ('config.json', 'metrics.json', 'predictions.csv', 'llm_calls.jsonl', 'run.log', 'README.md')}
    required |= {Path(path) for path in cfg['source_hashes']}
    for step in gate['steps'].values():
        if 'receipt_path' not in step:
            raise TerminalContinuationError('Each ordered local check must name its bound durable receipt')
        required.add(Path(step['receipt_path']))
        if _read(step['receipt_path']).get('status') != 'PASS':
            raise TerminalContinuationError('A bound local receipt itself does not report PASS')
    for seed in range(3):
        required.add(PHASE / f'states/seed{seed}.json')
        final_path = PHASE / f'final/L1_seed{seed}/final_predictions.json'; required.add(final_path)
        final = _sealed(final_path)
        if (final.get('status') != 'complete' or final.get('seed') != seed or final['train_count'] != 2000
            or final['test_count'] != 1000 or final.get('TEST_target_values_read') is not False
            or final.get('TEST_scoring_performed') is not False):
            raise TerminalContinuationError('Only existing complete unscored final model/prediction outputs can be consumed')
        for ref in final['artifacts'].values():
            required.add(Path(ref['path']))
            if _sha(ref['path']) != ref['sha256']:
                raise TerminalContinuationError('An already completed final artifact changed')
    missing = {_read_path(path) for path in required} - set(artifacts)
    if missing:
        raise TerminalContinuationError('The gate omits mandatory source/model/prediction/ledger inputs: ' + ', '.join(str(p) for p in sorted(missing)))
    for path, expected in cfg['source_hashes'].items():
        if artifacts[_read_path(path)] != expected:
            raise TerminalContinuationError('An unchanged frozen scientific source hash differs from the gate')
    _old_results()
    states = {seed: _read(PHASE / f'states/seed{seed}.json') for seed in range(3)}
    for seed, state in states.items():
        if (state.get('round_completed') != 20 or len(state.get('labels', {})) != 2000
            or any(len(state.get(key, [])) != 20 for key in ('history', 'round_model_paths', 'query_history', 'memory_audits'))
            or not set(cfg['data']['perseed'][str(seed)]['S0_ids']) <= set(state['labels'])):
            raise TerminalContinuationError('All3 seeds must have complete20 rounds and2000 labels before TEST scoring')
    arms = _read(PHASE / 'committed_prediction_arms.json'); _prediction_structure(arms, cfg)
    if not post and any((PHASE / name).exists() for name in ('evaluation_started.json', 'terminal_results.json', 'private_pairing_cache.npz', 'paired_comparisons.json')):
        raise TerminalContinuationError('An L1 scoring gate or terminal output already exists; never reread TEST labels')
    return gate, cfg, states, arms


def _dependencies():
    for directory in (PHASE.parent / 'src', PHASE / 'src', REVISION / 'src'):
        if str(directory) not in sys.path: sys.path.insert(0, str(directory))
    import l1_data
    import l1_stats
    import l1_revision_report
    return l1_data, l1_stats, l1_revision_report


def _metrics(**values):
    record = _read(PUBLIC / 'metrics.json'); O, L0 = _old_results()
    record.update(stage='4.3', condition='L1', updated_utc=_utc(),
        prior_stage4_1_results=deepcopy(O), prior_stage4_2_results=deepcopy(L0), next_stage_started=False, **values)
    _json_write(PUBLIC / 'metrics.json', record)
    return record


def _readme(status, cfg, *, target_read, paired_pass=False):
    """Keep every original revision-description line; update status/pass only."""
    path = _read_path(PUBLIC / 'README.md')
    chunks = path.read_bytes().splitlines(keepends=True)
    status_rows = [i for i, row in enumerate(chunks) if row.startswith(b'Status:')]
    pass_rows = [i for i, row in enumerate(chunks) if row.startswith(b'Actual TEST scoring passes ')]
    if len(status_rows) != 1 or len(pass_rows) != 1:
        raise TerminalContinuationError('Preserve the original README: exactly one status and one TEST-pass line are required')
    for index, kind in ((status_rows[0], 'status'), (pass_rows[0], 'passes')):
        raw = chunks[index]; ending = b'\r\n' if raw.endswith(b'\r\n') else b'\n'
        line = raw.rstrip(b'\r\n').decode('utf-8')
        if kind == 'status':
            suffix = line[line.index(';'):] if ';' in line else ''
            line = 'Status: ' + status + suffix
        else:
            line, n1 = re.subn(r'L1=\d+', 'L1=' + str(int(target_read)), line, count=1)
            line, n2 = re.subn(r'totalH4=\d+', 'totalH4=' + str(2 + int(target_read)), line, count=1)
            if n1 != 1 or n2 != 1:
                raise TerminalContinuationError('The original TEST-pass disclosure format changed')
        chunks[index] = line.encode('utf-8') + ending
    approval = b'Public updates explicitly approved by user; see memory_content_amendment_L1.md and prior_stages_preservation_current.json.'
    if not any(row.rstrip(b'\r\n') == approval for row in chunks):
        if chunks and not chunks[-1].endswith(b'\n'):
            chunks[-1] += b'\n'
        chunks.append(approval + b'\n')
    if len(chunks) > 20:
        raise TerminalContinuationError('Preserved README plus the approved public-update link must remain at most20 lines')
    _write(PUBLIC / 'README.md', b''.join(chunks))


def _final_public_hashes():
    result = {}
    for name in ('config.json', 'metrics.json', 'predictions.csv', 'llm_calls.jsonl', 'run.log', 'README.md'):
        old, new = _read_path(SNAPSHOT / name), _read_path(PUBLIC / name)
        old_sha, new_sha = _sha(old), _sha(new)
        size = old.stat().st_size; prefix = hashlib.sha256()
        with new.open('rb') as stream:
            remaining = size
            while remaining:
                block = stream.read(min(4 * 1024 * 1024, remaining))
                if not block: break
                prefix.update(block); remaining -= len(block)
        result[name] = {'old_path': str(old), 'new_path': str(new),
            'old_sha256': old_sha, 'new_sha256': new_sha, 'old_bytes': size, 'new_bytes': new.stat().st_size,
            'identical_bytes': old_sha == new_sha and size == new.stat().st_size,
            'old_prefix_preserved': remaining == 0 and prefix.hexdigest() == old_sha}
    return result


def score(*, gate_path=MANUAL / 'authorization_gate.json'):
    gate, cfg, states, arms = _validate_gate(gate_path)
    started = time.monotonic(); marker = PHASE / 'evaluation_started.json'; target_read = False
    _json_write(MANUAL / 'continuation_started.json', {'utc': _utc(), 'pid': os.getpid(), 'gate_path': str(_read_path(gate_path)),
        'gate_sha256': _sha(gate_path), 'config_sha256': _sha(PHASE / 'config.json'), 'new_LLM_calls': 0,
        'new_fits_or_predictions': 0}, exclusive=True)
    data = None
    try:
        data, stats, report = _dependencies()
        summary = _read(PHASE / 'run_stop.json')['transport']
        if (summary.get('cli_attempt_count') != 183 or summary.get('completed_cli_attempt_count') != 183
            or summary.get('revision_accounting', {}).get('revised_successes') != 60):
            raise TerminalContinuationError('The existing183 physical/60 valid revised-call accounting changed')
        finals = [_sealed(PHASE / f'final/L1_seed{seed}/final_predictions.json') for seed in range(3)]
        bundle = data.load_bundle()  # Label-free; no reveal or target accessor.
        stats._normalise(arms, bundle['test_ids'])
        _json_write(marker, {**cfg['statistics']['marker_contract'], 'utc': _utc(),
            'config_sha256': _sha(PHASE / 'config.json'),
            'committed_prediction_arms_sha256': _sha(PHASE / 'committed_prediction_arms.json'),
            'public_predictions_sha256': _sha(PUBLIC / 'predictions.csv'),
            'manual_terminal_continuation_gate_sha256': _sha(gate_path)}, exclusive=True)
        terminal = stats.evaluate(arms, bundle, evaluation_marker=marker)
        target_read = True
        _json_write(PHASE / 'terminal_results.json', terminal, exclusive=True)
        _npz_write(PHASE / 'private_pairing_cache.npz', stats.private_pairing_cache())
        _metrics(status='running', phase='manual_terminal_scored_waiting_cached_pairs', results=terminal,
            error=None, current_L1_scoring_pass=1, prior_H4_TEST_scoring_passes=2,
            total_H4_TEST_scoring_passes=3, TEST_targets_read_this_condition=1000,
            manual_terminal_resume={'gate_sha256': _sha(gate_path), 'new_LLM_calls': 0, 'new_fits_or_predictions': 0})
        paired = stats.paired_comparisons({'O': PHASE.parent / 'phase4_1/private_pairing_cache.npz',
            'L0': PHASE.parent / 'phase4_2/private_pairing_cache.npz', 'L1': PHASE / 'private_pairing_cache.npz'},
            l1_evaluation_marker=marker)
        if paired.get('registered_L0_minus_O_MAE_check', {}).get('status') != 'PASS':
            raise TerminalContinuationError('Registered L0-O six-decimal reference check failed; do not reread TEST or change rules')
        _json_write(PHASE / 'paired_comparisons.json', paired, exclusive=True)
        original_elapsed = _read(PHASE / 'run_stop.json')['elapsed_seconds']
        elapsed = time.monotonic() - started
        paths = report.make_report(states, terminal, finals, paired, summary, original_elapsed + elapsed, cfg)
        report_path = _read_path(paths['report']); before = _sha(report_path)
        _write(report_path, report_path.read_text(encoding='utf-8') + '\n' + gate['report_appendix'].rstrip() + '\n')
        appendix_stage_sha256 = _sha(report_path)
        _readme('complete', cfg, target_read=True, paired_pass=True)
        elapsed = time.monotonic() - started
        _metrics(status='complete', phase='manual_terminal_complete_waiting_for_continue', results=terminal,
            paired_comparisons=paired, final_models=finals, reports=paths, rounds_complete=60, rounds_total=60,
            error=None, current_L1_scoring_pass=1, total_H4_TEST_scoring_passes=3,
            TEST_targets_read_this_condition=1000, LLM_calls_global=183, LLM_calls_stage4_3=63,
            LLM_calls_revised_L1=60, voided_original_calls=3, transport=summary,
            original_R2_run_elapsed_seconds=original_elapsed, manual_terminal_elapsed_seconds=elapsed,
            elapsed_seconds=original_elapsed + elapsed)
        public_hashes = _final_public_hashes()
        hash_lines = ['', '## Public file hashes after terminal completion', '',
            'The old files are the readonly completed-stage4.2 six-file snapshot. The new hashes below are taken after the final L1 metrics and README updates; this continuation makes no further public-file changes.',
            'Prefix facts refer to literal old-file bytes and do not imply whole JSON/config equality. No LLM calls or prediction rows were added by the terminal-only continuation.',
            '', '|File|Old SHA256|New SHA256|Old bytes|New bytes|Identical bytes|Old byte prefix preserved|', '|---|---|---|---|---|---|---|']
        for name, item in public_hashes.items():
            hash_lines.append('|' + name + '|' + '|'.join(str(item[key]) for key in
                ('old_sha256', 'new_sha256', 'old_bytes', 'new_bytes', 'identical_bytes', 'old_prefix_preserved')) + '|')
        _write(report_path, report_path.read_text(encoding='utf-8') + '\n'.join(hash_lines) + '\n')
        _json_write(MANUAL / 'report_appendix_audit.json', {'utc': _utc(),
            'original_generated_report_sha256': before, 'gate_appendix_stage_report_sha256': appendix_stage_sha256,
            'final_report_sha256': _sha(report_path), 'appended_report_sha256': _sha(report_path),
            'gate_sha256': _sha(gate_path), 'public_file_old_new_hashes': public_hashes,
            'appendix_utf8_sha256': hashlib.sha256(gate['report_appendix'].encode('utf-8')).hexdigest()}, exclusive=True)
        output_paths = [PHASE / name for name in META_OUTPUTS] + [PUBLIC / 'metrics.json', PUBLIC / 'README.md']
        output_paths += [Path(path) for path in paths.values()] + [MANUAL / 'report_appendix_audit.json']
        result = {'status': 'complete', 'utc': _utc(), 'manual_terminal_elapsed_seconds': elapsed,
            'L1_TEST_target_accessor_calls': 1, 'new_LLM_calls': 0, 'new_fits_or_predictions': 0,
            'next_stage_started': False, 'gate_sha256': _sha(gate_path),
            'output_sha256': {str(_read_path(path)): _sha(path) for path in output_paths}, 'reports': paths}
        _json_write(MANUAL / 'continuation_stop.json', result, exclusive=True)
        return result
    except BaseException as error:
        target_read = target_read or bool(getattr(data, '_TRUTH_READ', False))
        failure = {'status': 'failed', 'utc': _utc(), 'error': type(error).__name__ + ': ' + str(error),
            'manual_terminal_elapsed_seconds': time.monotonic() - started,
            'L1_evaluation_marker_exists': marker.exists(), 'L1_TEST_truth_accessor_consumed': target_read,
            'new_LLM_calls': 0, 'new_fits_or_predictions': 0, 'next_stage_started': False,
            'automatic_retry_permitted': False, 'gate_sha256': _sha(gate_path)}
        _json_write(MANUAL / 'continuation_stop.json', failure, exclusive=True)
        _write(MANUAL / 'traceback.txt', traceback.format_exc())
        _metrics(status='failed', phase='manual_terminal_stopped', error=failure['error'],
            current_L1_scoring_pass=int(target_read), total_H4_TEST_scoring_passes=2 + int(target_read),
            TEST_targets_read_this_condition=1000 if target_read else 0,
            manual_terminal_resume=failure)
        _readme('failed', cfg, target_read=target_read)
        return failure


def verify(*, gate_path=MANUAL / 'authorization_gate.json'):
    stop_path = MANUAL / 'continuation_stop.json'
    post = stop_path.is_file()
    gate, cfg, states, arms = _validate_gate(gate_path, post=post)
    if not post:
        return {'status': 'PASS', 'terminal_state': 'not_started', 'new_TEST_reads_from_verify': 0}
    stop = _read(stop_path)
    if stop.get('gate_sha256') != _sha(gate_path):
        raise TerminalContinuationError('Continuation stop no longer binds the explicit user gate')
    if stop['status'] != 'complete':
        return {'status': 'PASS', 'terminal_state': stop['status'],
            'L1_evaluation_marker_exists': (PHASE / 'evaluation_started.json').exists(), 'new_TEST_reads_from_verify': 0}
    for path, expected in stop['output_sha256'].items():
        if _sha(path) != expected:
            raise TerminalContinuationError('A committed terminal/report/public output changed')
    marker = _read(PHASE / 'evaluation_started.json'); m = _read(PUBLIC / 'metrics.json')
    terminal = _read(PHASE / 'terminal_results.json'); paired = _read(PHASE / 'paired_comparisons.json')
    if (m.get('status') != 'complete' or m.get('results') != terminal or m.get('paired_comparisons') != paired
        or m.get('current_L1_scoring_pass') != 1 or m.get('total_H4_TEST_scoring_passes') != 3
        or m.get('TEST_targets_read_this_condition') != 1000 or terminal.get('TEST_target_accessor_calls') != 1
        or paired.get('registered_L0_minus_O_MAE_check', {}).get('status') != 'PASS'
        or terminal.get('evaluation_marker', {}).get('sha256') != _sha(PHASE / 'evaluation_started.json')
        or marker.get('manual_terminal_continuation_gate_sha256') != _sha(gate_path)
        or marker.get('committed_prediction_arms_sha256') != _sha(PHASE / 'committed_prediction_arms.json')
        or marker.get('public_predictions_sha256') != _sha(PUBLIC / 'predictions.csv')
        or m.get('next_stage_started') is not False):
        raise TerminalContinuationError('Terminal/public/pass/paired-reference commitment differs')
    O, L0 = _old_results()
    if m.get('prior_stage4_1_results') != O or m.get('prior_stage4_2_results') != L0:
        raise TerminalContinuationError('Published preserved O/L0 numerical results changed')
    names = {'config.json', 'predictions.csv', 'llm_calls.jsonl', 'metrics.json', 'run.log', 'README.md'}
    if {p.name for p in PUBLIC.iterdir()} != names or len((PUBLIC / 'README.md').read_text(encoding='utf-8').splitlines()) > 20:
        raise TerminalContinuationError('Public six-file/README20 boundary changed')
    return {'status': 'PASS', 'terminal_state': 'complete', 'L1_TEST_target_accessor_calls': 1,
        'O_L0_results_preserved': True, 'new_TEST_reads_from_verify': 0, 'no4_4': True,
        'output_artifacts_checked': len(stop['output_sha256'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('score', 'verify'))
    parser.add_argument('--gate', default=str(MANUAL / 'authorization_gate.json'))
    args = parser.parse_args()
    result = score(gate_path=args.gate) if args.action == 'score' else verify(gate_path=args.gate)
    if sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
