"""Reporting from already-completed L1/L0 metadata and statistical outputs.

No dataset/target accessor, model, generated feature, LOFO or bootstrap is run.
Derivative of the preserved original reporting module; only explicitly called
make_report writes below phase4_3/report. No old report/statistical source is changed.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[6]
PHASE = ROOT / 'work/h-data/H4/phase4_3'
OLD_PHASE = ROOT / 'work/h-data/H4/phase4_2'

from collections import Counter
import csv
import hashlib
import io
import json
import math
import os
import re
import uuid

METRICS = ('mae', 'rmse', 'r2')
MODES = ('row_weighted', 'group_weighted')
PAIRS = ('L1_minus_L0', 'L1_minus_O', 'L0_minus_O')
LEDGER = ROOT / 'H4_feature_loop/llm_calls.jsonl'
MEMORY_LIMIT = 20000
INPUT_LIMIT = 189000
ORIGINAL_REPORT_SHA256 = '832c395a01704f3e559cc81c318eb35672a3af416956147950cc585c487f62d9'
TOKEN_KEYS = ('input_tokens', 'output_tokens', 'cached_input_tokens')


class ReportIntegrityError(ValueError):
    pass


def _read_path(path):
    path = Path(path).resolve(strict=True)
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(ROOT.resolve()):
        raise ReportIntegrityError('Report inputs must resolve within the non-OneDrive research root')
    return path


def _output(path):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to((PHASE / 'report').resolve()):
        raise ReportIntegrityError('L1 report writes must stay below the new phase4_3/report directory')
    return path


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _read(path, *, sealed=False):
    value = json.loads(_read_path(path).read_text(encoding='utf-8-sig'))
    if sealed:
        raw = dict(value); expected = raw.pop('record_sha256', None)
        if expected is None or _hash(raw) != expected:
            raise ReportIntegrityError('Saved model/LOFO metadata self-hash changed')
    return value


def _write(path, content):
    target = _output(path); _output(target.parent).mkdir(parents=True, exist_ok=True)
    temp = _output(target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp'))
    with temp.open('wb') as stream:
        stream.write(content.encode('utf-8') if isinstance(content, str) else content)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, _output(target))


def _csv(path, rows, columns):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows); _write(path, stream.getvalue())


def _state(states, seed):
    state = states[seed] if seed in states else states[str(seed)]
    if state['seed'] != seed or state.get('round_completed', 20) != 20:
        raise ReportIntegrityError('Report requires all three completed twenty-round states')
    return state


def _rounds(state, condition, seed):
    records = [_read(path, sealed=True) for path in state['round_model_paths']]
    if len(records) != 20 or {r['round_index'] for r in records} != set(range(1, 21)):
        raise ReportIntegrityError('Report requires exactly rounds1..20 for every condition/seed')
    if any(r['seed'] != seed or r.get('condition') != condition or r.get('status') != 'complete' for r in records):
        raise ReportIntegrityError('Round metadata condition/seed/status mismatch')
    return sorted(records, key=lambda r: r['round_index'])


def _number(value, *, signed=False):
    if value is None: return 'undefined'
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ReportIntegrityError('Report values must be existing finite metrics or explicit None')
    return format(value, '+.6f' if signed else '.6f')


def _metric(value, *, signed=False):
    if not isinstance(value, dict) or len(value['ci95']) != 2:
        raise ReportIntegrityError('Malformed saved metric interval')
    return (_number(value['point'], signed=signed) + ' [' +
        ', '.join(_number(x, signed=signed) for x in value['ci95']) + ']')


def _score_rows(lines, result, *, paired=False):
    for mode in MODES:
        for seed in ('0', '1', '2', 'mean'):
            scores = result['arithmetic_seed_mean'] if seed == 'mean' else result['per_seed'][seed]
            if not paired and seed != 'mean': scores = scores['scores']
            cells = [_metric(scores[mode][metric], signed=paired) for metric in METRICS]
            lines.append('|' + mode + '|' + seed + '|' + '|'.join(cells) + '|')


def _failure(section, names, kept, n):
    rates = section['failure_rates']; counts = dict(section['status_counts'])
    if (len(rates) != len(names) or not set(kept) <= set(names)
        or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in rates)
        or sum(counts.values()) != n * len(names)):
        raise ReportIntegrityError('Existing feature-failure metadata is inconsistent')
    failed = sum(value for key, value in counts.items() if key != 'ok')
    total = sum(counts.values())
    kept_rates = [rates[i] for i, name in enumerate(names) if name in set(kept)]
    return {'failed_cells': failed, 'total_cells': total,
        'raw_rate': failed / total if total else None,
        'kept_mean_rate': sum(kept_rates) / len(kept_rates) if kept_rates else None,
        'statuses': counts}


def _memory_rows(states, input_limit):
    """Record exact saved bytes separately from the conservative whole-input bound."""
    if input_limit != INPUT_LIMIT:
        raise ReportIntegrityError('The unchanged whole-input limit must be189000 bytes')
    rows = []
    for seed in (0, 1, 2):
        state = _state(states, seed)
        audits = state['memory_audits']; history = state['history']
        if (len(audits) != 20 or len(history) != 20
            or {a['round_index'] for a in audits} != set(range(1, 21))):
            raise ReportIntegrityError('Complete revised per-request memory/input audits are required')
        for audit in sorted(audits, key=lambda a: a['round_index']):
            rnd = audit['round_index']; ordinary = audit['input_budget']; repaired = audit['format_repair_input_budget']
            note = history[rnd - 1].get('notes')
            if type(note) is not str or len(note) > 300:
                raise ReportIntegrityError('Actual complete round notes must contain at most300 Unicode characters')
            if (audit['request_id'] != f"L1R2_seed{seed}_round{rnd:02d}"
                or type(audit['memory_token_count']) is not int or not 0 <= audit['memory_token_count'] <= MEMORY_LIMIT
                or type(audit['memory_UTF8_bytes']) is not int or audit['memory_UTF8_bytes'] < 0
                or type(audit['memory_trimmed']) is not bool):
                raise ReportIntegrityError('A saved revised memory audit violates its frozen20000-token bound')
            for budget in (ordinary, repaired):
                keys = ('canonical_prompt_bytes', 'wire_prompt_bytes', 'known_fixed_context_bytes', 'input_token_upper_bound')
                if (any(type(budget.get(k)) is not int or budget[k] < 0 for k in keys)
                    or budget['input_limit'] != input_limit or budget.get('within_budget') is not True
                    or budget['input_token_upper_bound'] != budget['wire_prompt_bytes'] + budget['known_fixed_context_bytes'] + 1500
                    or budget['input_token_upper_bound'] > input_limit):
                    raise ReportIntegrityError('Saved byte accounting is inconsistent or exceeds the unchanged guard')
            if ordinary['canonical_prompt_bytes'] != repaired['canonical_prompt_bytes']:
                raise ReportIntegrityError('Repair budget must bind the same original canonical prompt')
            if repaired['wire_prompt_bytes'] < ordinary['wire_prompt_bytes']:
                raise ReportIntegrityError('Repair accounting cannot be smaller than the ordinary wire payload')
            policy = audit.get('memory_policy_audit', {})
            expected_recent = list(range(max(1, rnd - 3), rnd))
            if (policy.get('name_description_source_or_source_summary_retained') is not False
                or policy.get('older_round_notes_queries_retained_in_any_counted_view') is not False
                or policy.get('recent_rounds_preserved') != expected_recent):
                raise ReportIntegrityError('Saved memory-content audit must confirm metadata-only and recent3-only policy')
            if ordinary['known_fixed_context_bytes'] != repaired['known_fixed_context_bytes']:
                raise ReportIntegrityError('Ordinary and repair accounting must share frozen fixed context')
            prior_notes = [{'round': n, 'characters': len(history[n - 1]['notes'])}
                for n in range(max(1, rnd - 3), rnd)]
            # Input round r sees completed notes r-3..r-1; the current note is output.
            if 'memory_notes_chars_by_round' in audit and audit['memory_notes_chars_by_round'] != prior_notes:
                raise ReportIntegrityError('Saved retained note counts differ from the actual recent-three history')
            if 'memory_note_chars' in audit and audit['memory_note_chars'] != sum(x['characters'] for x in prior_notes):
                raise ReportIntegrityError('Saved total memory note characters differ from the history')
            rows.append({'seed': seed, 'round': rnd, 'request_id': audit['request_id'],
                'memory_tokens': audit['memory_token_count'], 'memory_UTF8_bytes': audit['memory_UTF8_bytes'],
                'memory_trimmed': audit['memory_trimmed'],
                'canonical_prompt_UTF8_bytes': ordinary['canonical_prompt_bytes'],
                'ordinary_wire_prompt_UTF8_bytes': ordinary['wire_prompt_bytes'],
                'repair_wire_prompt_UTF8_bytes': repaired['wire_prompt_bytes'],
                'known_fixed_context_UTF8_bytes': ordinary['known_fixed_context_bytes'],
                'framing_allowance_bytes': 1500,
                'ordinary_complete_UTF8_bound_bytes': ordinary['input_token_upper_bound'],
                'repair_complete_UTF8_bound_bytes': repaired['input_token_upper_bound'],
                'input_limit_bytes': input_limit, 'memory_limit_tokens': MEMORY_LIMIT,
                'retained_prior_notes_characters': sum(x['characters'] for x in prior_notes),
                'current_response_note_characters': len(note)})
    return rows


def _revision_provenance(cfg):
    provenance = cfg.get('revision_provenance')
    required = {'original_config_sha256', 'original_frozen_utc', 'revision_utc',
        'reason', 'voided_calls', 'new_request_namespace', 'archive_manifest_path'}
    if not isinstance(provenance, dict) or not required <= set(provenance):
        raise ReportIntegrityError('The original freeze and explicit later revision provenance are required')
    if (provenance['new_request_namespace'] != 'L1R2'
        or cfg['memory']['limit_tokens'] != MEMORY_LIMIT):
        raise ReportIntegrityError('Only the authorized L1R2 metadata-memory20000 revision can use this report')
    for key in ('original_config_sha256',):
        if not re.fullmatch('[0-9a-f]{64}', provenance[key]):
            raise ReportIntegrityError('Malformed preserved original configuration hash')
    if not all(isinstance(provenance[key], str) and provenance[key] for key in
        ('original_frozen_utc', 'revision_utc', 'reason', 'archive_manifest_path')):
        raise ReportIntegrityError('Revision chronology/reason cannot be omitted or inferred')
    voided = provenance['voided_calls']
    if (len(voided) != 3 or {item['request_id'] for item in voided} !=
        {f'L1_seed{seed}_round01' for seed in (0, 1, 2)}
        or {item['call_index'] for item in voided} != {121, 122, 123}):
        raise ReportIntegrityError('Exactly the three original first-round L1 calls must remain voided')
    for item in voided:
        for key in ('prompt_sha256', 'canonical_prompt_sha256', 'validated_response_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', item.get(key, '')):
                raise ReportIntegrityError('Complete voided-call prompt/response SHA256 provenance is required')
    return provenance


def _call_accounting(summary, provenance):
    raw = _read_path(LEDGER).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ReportIntegrityError('Only newline-committed complete terminal call ledgers can be reported')
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    reservations = {}; results = {}
    for item in records:
        kind = item.get('record_type')
        if kind not in ('reservation', 'result'):
            raise ReportIntegrityError('Unexpected call-ledger record type')
        destination = reservations if kind == 'reservation' else results
        index = item.get('call_index')
        if type(index) is not int or index in destination:
            raise ReportIntegrityError('Duplicate or malformed call-ledger index')
        destination[index] = item
    total = len(reservations)
    if (set(reservations) != set(range(1, total + 1)) or set(results) != set(reservations)
        or not 183 <= total <= 300 or summary.get('cli_attempt_count', summary.get('n_calls')) != total):
        raise ReportIntegrityError('Complete contiguous global physical-call accounting must include the voided3')
    for index, reservation in reservations.items():
        result = results[index]
        if any(result.get(key) != reservation.get(key) for key in ('request_id', 'prompt_sha256')):
            raise ReportIntegrityError('Result identity differs from its physical reservation')
    void_ids = {item['request_id'] for item in provenance['voided_calls']}
    formal_ids = {f'L1R2_seed{seed}_round{rnd:02d}' for seed in (0, 1, 2) for rnd in range(1, 21)}
    prior = [item for item in results.values() if item['call_index'] <= 120]
    void = [item for item in results.values() if item['request_id'] in void_ids]
    revised = [item for item in results.values() if item['request_id'] in formal_ids]
    if (len(prior) != 120 or len({item['request_id'] for item in prior}) != 120
        or len(void) != 3 or len(prior) + len(void) + len(revised) != total
        or {item['request_id'] for item in revised} != formal_ids):
        raise ReportIntegrityError('The60 valid revised logical requests and3 voided originals must be separately accounted')
    provider_valid = set(); query_fallback = set()
    for request in formal_ids:
        attempts = [item for item in revised if item['request_id'] == request]
        if any(item.get('status') == 'ok' and item.get('validated_response_sha256') for item in attempts):
            provider_valid.add(request)
            continue
        # The unchanged protocol permits only a query-format fallback after the
        # single repair; completed models must preserve the valid features/notes.
        last = max(attempts, key=lambda item: item['call_index'])
        logical = _read(PHASE / 'logical_results' / (request + '.json'))
        query = _read(PHASE / 'query_records' / (request + '.json'))
        audit = query.get('response_audit', {})
        if (last.get('status') != 'format_invalid' or last.get('format_round') != 1
            or logical.get('request_id') != request or logical.get('last_call_index') != last['call_index']
            or logical.get('last_status') != 'format_invalid' or logical.get('admissible_format_failure') is not True
            or query.get('request_id') != request or query.get('query_payload') != {'queries': [], 'rationale': ''}
            or any(audit.get(key) is not True for key in
                ('query_fallback', 'features_preserved', 'notes_preserved', 'admissible_format_failure'))):
            raise ReportIntegrityError('A revised request lacks either a validated response or an explicitly committed admissible query-only fallback')
        query_fallback.add(request)
    for declared in provenance['voided_calls']:
        item = results.get(declared['call_index'], {})
        if any(item.get(key) != declared[key] for key in
            ('request_id', 'prompt_sha256', 'canonical_prompt_sha256', 'validated_response_sha256')):
            raise ReportIntegrityError('Voided original call provenance no longer matches the immutable ledger')
    def tokens(items):
        return {**{key: sum(item.get(key) or 0 for item in items) for key in TOKEN_KEYS},
            'unknown_token_results': sum(any(item.get(key) is None for key in TOKEN_KEYS) for item in items)}
    return {'global_physical_cli_attempts': total, 'global_completed_cli_attempts': len(results),
        'global_physical_logical_requests': len({item['request_id'] for item in results.values()}),
        'formal_valid_logical_requests': 180, 'voided_logical_requests': 3, 'voided_cli_attempts': len(void),
        'formal_L1_logical_requests': 60, 'formal_L1_completed_valid_rounds': 60,
        'formal_L1_successful_requests': len(provider_valid),
        'formal_L1_accepted_query_fallback_requests': len(query_fallback),
        'formal_L1_cli_attempts': len(revised), 'all_L1_cli_attempts': len(void) + len(revised),
        'formal_L1_extra_physical_attempts': len(revised) - 60,
        'prior_O_L0_cli_attempts': len(prior),
        'global_valid_physical_attempts_excluding_voided': total - len(void),
        'ledger_path': str(_read_path(LEDGER)), 'ledger_sha256': hashlib.sha256(raw).hexdigest(),
        'tokens_by_scope': {name: tokens(items) for name, items in
            (('formal_L1', revised), ('voided_L1', void), ('all_L1', void + revised), ('global', list(results.values())))}}


def _top_features(winner, final):
    path = Path(winner['metadata_path']).parent / 'lofo/lofo.json'
    lofo = _read(path, sealed=True)
    if (lofo['status'] != 'complete' or lofo['seed'] != winner['seed']
        or lofo['round_index'] != winner['round_index']
        or lofo['identity']['source_round_record_sha256'] != winner['record_sha256']
        or lofo['raw_features'] != winner['features']
        or lofo['baseline_validation_mae'] != winner['validation_mae']):
        raise ReportIntegrityError('Winner LOFO contributions are not bound to the saved winner')
    kept = set(final['full_train_features']['kept_names']); candidates = []; unavailable = []
    contributions = {item['feature_index']: item['delta_without_minus_full'] for item in lofo['contributions']}
    if set(contributions) != set(range(len(winner['features']))):
        raise ReportIntegrityError('LOFO contribution coverage changed')
    for index, feature in enumerate(winner['features']):
        if feature['name'] not in kept: continue
        delta = contributions[index]
        if delta is None:
            unavailable.append(feature['name']); continue
        if type(delta) not in (int, float) or not math.isfinite(delta):
            raise ReportIntegrityError('LOFO contribution must be a saved finite delta or None')
        candidates.append({'seed': winner['seed'], 'winner_round': winner['round_index'],
            'feature_index': index, 'name': feature['name'], 'description': feature['description'],
            'source': feature['source'], 'source_sha256': hashlib.sha256(feature['source'].encode()).hexdigest(),
            'delta_without_minus_full': delta, 'LOFO_metadata_path': str(path.resolve()),
            'LOFO_record_sha256': lofo['record_sha256']})
    candidates.sort(key=lambda item: (-item['delta_without_minus_full'], item['feature_index']))
    return {'features': candidates[:5], 'retained_without_winner_LOFO': unavailable,
        'ranking_scope': 'saved winner-round internal validation; positive MAE_without-minus-full is beneficial; never TEST selection'}


def _plot(curves, target):
    cache = _output(PHASE / 'report/.matplotlib'); cache.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for seed, ax in enumerate(axes):
        for condition, style, color in (('L0', '--', '#727880'), ('L1', '-', '#1768ac')):
            own = [r for r in curves if r['condition'] == condition and r['seed'] == seed]
            ax.plot([r['round'] for r in own], [r['validation_mae'] for r in own],
                linestyle=style, marker='.', color=color, label=condition)
        ax.set(title='Seed ' + str(seed), xlabel='Round', xticks=[1, 5, 10, 15, 20], xlim=(1, 20))
        ax.grid(alpha=.25); ax.legend()
    axes[0].set_ylabel('Internal validation row MAE (eV/atom)')
    figure.suptitle('L1 and L0: fixed template roles; revealed validation membership grows', fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94))
    try: figure.savefig(_output(target), dpi=160)
    finally: plt.close(figure)


def make_report(states, terminal, finals, paired, summary, elapsed, cfg):
    """Render already-existing metrics and metadata; never recompute science.

    Revised memory_audits retains original request byte-budget fields: request_id/round_index,
    memory_token_count/memory_UTF8_bytes/memory_trimmed, input_budget and
    format_repair_input_budget; histories retain actual notes. The saved public
    physical ledger separates voided3 from60 valid L1R2 requests and retries.
    cfg.revision_provenance and cfg.input_limit_basis are reported as recorded.
    """
    if terminal.get('stage') != '4.3' or terminal.get('condition') != 'L1':
        raise ReportIntegrityError('Only committed L1 terminal output can be reported')
    if paired.get('registered_L0_minus_O_MAE_check', {}).get('status') != 'PASS' or set(paired['comparisons']) != set(PAIRS):
        raise ReportIntegrityError('All three authorized pairs and the registered L0-O reference must pass')
    if len(finals) != 3 or {item['seed'] for item in finals} != {0, 1, 2}:
        raise ReportIntegrityError('Report requires one final model for each seed')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ReportIntegrityError('Elapsed run time must be an existing finite nonnegative duration')
    provenance = _revision_provenance(cfg)
    accounting = _call_accounting(summary, provenance)
    input_limit = cfg['prompts']['input_limit']
    memory_rows = _memory_rows(states, input_limit)
    curves = []; models = {}; top = {}; budgets = []; failure_rows = []
    for seed in (0, 1, 2):
        state = _state(states, seed); current = _rounds(state, 'L1', seed)
        prior = _rounds(_read(OLD_PHASE / 'states' / f'seed{seed}.json'), 'L0', seed)
        for condition, records in (('L0', prior), ('L1', current)):
            for record in records:
                curves.append({'condition': condition, 'seed': seed, 'round': record['round_index'],
                    **{key: record[key] for key in ('labelled_count', 'fit_count', 'validation_count', 'validation_mae', 'validation_group_mae')}})
        final = next(item for item in finals if item['seed'] == seed)
        winner = current[final['selected_round_index'] - 1]
        if (final['train_count'] != 2000 or final['test_count'] != 1000
            or final['selected_round_record_sha256'] != winner['record_sha256']):
            raise ReportIntegrityError('Final model identity differs from the saved chosen round')
        if len(state['history']) != 20:
            raise ReportIntegrityError('The complete twenty-round memory history is required')
        fill = state['final_fill']['selected_n']
        if type(fill) is not int or not 0 <= fill <= 1500:
            raise ReportIntegrityError('Malformed final uniform-fill count')
        budgets.append({'seed': seed, 'query_labels': 1500 - fill, 'uniform_fill': fill,
            'final_labels': final['train_count'], 'winner_round': final['selected_round_index'],
            'raw_features': len(winner['features']), 'kept_features': len(final['full_train_features']['kept_names'])})
        names = [f['name'] for f in winner['features']]; kept = final['full_train_features']['kept_names']
        train = _failure(final['full_train_features'], names, kept, final['train_count'])
        test = _failure(final['test_features'], names, kept, final['test_count'])
        failure_rows.append({'seed': seed, 'TRAIN_raw': train, 'TEST_raw': test})
        models[seed] = {'final': final, 'winner': winner}; top[str(seed)] = _top_features(winner, final)
    report_dir = _output(PHASE / 'report'); report_dir.mkdir(parents=True, exist_ok=True)
    paths = {'validation_curve': str(_output(report_dir / 'L1_L0_validation_MAE.png')),
        'curve_csv': str(_output(report_dir / 'L1_L0_validation_curve.csv')),
        'request_input_ledger': str(_output(report_dir / 'L1_request_input_ledger.csv')),
        'top_feature_sources': str(_output(report_dir / 'L1_winner_top5_features.json')),
        'report': str(_output(report_dir / 'H4_stage4_3_report.md')),
        'revision_audit': str(_output(report_dir / 'L1_revision_accounting.json'))}
    _csv(paths['curve_csv'], curves, list(curves[0]))
    _csv(paths['request_input_ledger'], memory_rows, list(memory_rows[0]))
    _write(paths['top_feature_sources'], json.dumps(top, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    _plot(curves, paths['validation_curve'])
    _write(paths['revision_audit'], json.dumps({'revision_provenance': provenance, 'call_accounting': accounting,
        'preserved_original_report_source_sha256': ORIGINAL_REPORT_SHA256}, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# H4 stage4.3 L1R2 revised results', 'Status: complete; stage4.4 waits for explicit continue.',
        'All values below come from saved terminal metrics, paired draws, round metadata and request ledgers. No report-time fit, feature execution, TEST access or resampling occurs.',
        f"Fixed TEST{terminal['n_test']}; {terminal['n_groups']} original template groups; {terminal['n_boot']} shared group bootstrap draws.",
        'MAE/RMSE units: eV/atom. R2 is dimensionless. Seed means are arithmetic means of metrics, never prediction ensembles.',
        '', '## Original freeze and later authorized revision', '',
        'Original L1 freeze UTC: ' + provenance['original_frozen_utc'] + '; original config SHA256: ' + provenance['original_config_sha256'] + '.',
        'Authorized revision UTC: ' + provenance['revision_utc'] + '. Reason: ' + provenance['reason'],
        'The original first three L1 responses are preserved but voided for this revised experiment. All60 formal L1 requests restart from S0 in the new L1R2 namespace; the original successful responses are not reused as revised rounds.',
        'Only metadata memory content, its20000-token cap and the necessary memory-description text changed. The whole-input189000-byte guard, tokenizer/encoding, schema, query fields, HGB, seeds and template split rules remain unchanged.',
        'Archive manifest: ' + provenance['archive_manifest_path'],
        '', '|Voided original request|Physical call index|Prompt SHA256|Canonical prompt SHA256|Validated response SHA256|', '|---|---|---|---|---|']
    for item in sorted(provenance['voided_calls'], key=lambda row: row['call_index']):
        lines.append('|' + '|'.join(str(item[key]) for key in ('request_id', 'call_index', 'prompt_sha256', 'canonical_prompt_sha256', 'validated_response_sha256')) + '|')
    lines += [
        '', '## L1 terminal metrics', '', '|Weight|Seed|MAE [95% CI]|RMSE [95% CI]|R2 [95% CI]|', '|---|---|---|---|---|']
    _score_rows(lines, terminal)
    lines += ['', '## Prespecified paired differences', '',
        'Definition: arm A minus arm B on the same TEST identities and shared stored group draws. Negative MAE/RMSE and positive R2 favor A. No new bootstrap draws or TEST reads were made.',
        'L0-O six-decimal reference check: PASS.']
    for name in PAIRS:
        lines += ['', '### ' + name.replace('_minus_', ' minus '), '',
            '|Weight|Seed|Delta MAE [95% CI]|Delta RMSE [95% CI]|Delta R2 [95% CI]|', '|---|---|---|---|---|']
        _score_rows(lines, paired['comparisons'][name], paired=True)
    lines += ['', '## Labels and final features', '',
        '|Seed|Query labels|Uniform fill|Final labels|Winner round|Raw features|Kept features|', '|---|---|---|---|---|---|---|']
    for row in budgets: lines.append('|' + '|'.join(str(row[k]) for k in row) + '|')
    lines += ['', 'Failure rates below include all raw generated functions, including functions rejected by the AST policy. Kept rates are the mean of the saved per-function rates for final retained names; TEST failures are diagnostic only.', '',
        '|Seed|Domain|Raw failed/total cells|Raw failure rate|Kept failure rate|Status counts|', '|---|---|---|---|---|---|']
    for row in failure_rows:
        for domain in ('TRAIN', 'TEST'):
            item = row[domain + '_raw']
            statuses = ', '.join(str(k) + '=' + str(v) for k, v in sorted(item['statuses'].items()))
            lines.append(f"|{row['seed']}|{domain}|{item['failed_cells']}/{item['total_cells']}|{_number(item['raw_rate'])}|{_number(item['kept_mean_rate'])}|{statuses}|")
    lines += ['', '## Memory and complete request sizes', '',
        'Memory tokens use the unchanged pinned local counter on the exact retained metadata-only canonical JSON. Historical source/name/description/source summaries are absent from memory; current active feature source remains in its separate unchanged prompt section. Recent3 query/statistics/note records are retained; top30 source-hash ranking and historical-instance membership preserve the prior algorithm. Exact canonical/wire prompt and memory UTF8 sizes are separate from the whole-input bound, which adds fixed context and a1500-byte framing allowance. Byte bounds are not measured model-token counts.',
        '|Seed|Requests|Max memory tokens|Max memory bytes|Prescribed trims|Max ordinary bound bytes|Max repair bound bytes|', '|---|---|---|---|---|---|---|']
    for seed in (0, 1, 2):
        own = [row for row in memory_rows if row['seed'] == seed]
        values = [seed, len(own), max(r['memory_tokens'] for r in own), max(r['memory_UTF8_bytes'] for r in own),
            sum(r['memory_trimmed'] for r in own), max(r['ordinary_complete_UTF8_bound_bytes'] for r in own), max(r['repair_complete_UTF8_bound_bytes'] for r in own)]
        lines.append('|' + '|'.join(str(v) for v in values) + '|')
    lines += ['', 'The full60-request size ledger is [L1_request_input_ledger.csv](L1_request_input_ledger.csv), including each exact canonical/wire payload byte count, memory token/byte count, trim flag, retained prior-note characters and current response-note Unicode character count. The current response note is not part of its own request memory.',
        'Frozen L1 complete-input bound: ' + str(input_limit) + ' bytes.',
        'Frozen accounting method: ' + cfg['prompts']['input_budget_method'],
        'Active revised memory cap:20000 tokens. The same pinned tiktoken0.12.0/o200k_base.encode_ordinary counter and complete canonical-JSON counting method are retained.',
        'The unchanged counter verification record still contains its original6000-token limit field as legacy provenance; that historical field is not the active revised memory cap.',
        'Unchanged token-counter verification record SHA256: ' + cfg['token_counter']['token_counter_sha256'] + '.',
        'L0 raised its input limit during round7 manual recovery. Original L1 was frozen and made3 calls before this explicit revision. This later revision is frozen before the new L1R2 run; it was not part of the original freeze. O/L0 results and original L1 records remain preserved.']
    if 'input_limit_basis' in cfg:
        lines += ['', 'Recorded pre-revision-freeze measurement basis (displayed unchanged):', '```json',
            json.dumps(cfg['input_limit_basis'], ensure_ascii=False, indent=2, allow_nan=False), '```']
    else:
        lines.append('The input-limit measurement basis remains in the frozen config/projection; no report-time projection is performed.')
    total = accounting['global_physical_cli_attempts']; valid_attempts = accounting['formal_L1_cli_attempts']
    lines += ['', '## Calls, tokens and elapsed time', '',
        f'Elapsed revised run time: {elapsed:.3f} seconds. Prior O/L0 physical attempts:120; voided original L1 attempts:3; valid L1R2 logical requests:60; revised L1 physical attempts:{valid_attempts}; all L1 physical attempts:{accounting["all_L1_cli_attempts"]}; global physical attempts:{total}/300.',
        f'Formal valid logical requests across O/L0/L1R2:180. Global physical attempts excluding the voided3:{total - 3}; revised repair/retry extra attempts:{valid_attempts - 60}. With no extra attempts these counts are183 physical,180 formal valid and63 physical L1 versus60 valid revised L1.',
        'Validated revised response requests:' + str(accounting['formal_L1_successful_requests'])
            + '; completed rounds with the unchanged admissible query-only empty-query fallback:'
            + str(accounting['formal_L1_accepted_query_fallback_requests'])
            + '. Accepted fallbacks preserve valid features/notes and are not relabelled as fully validated provider responses.',
        'Reported provider token sums include every completed physical attempt in each scope, including the voided3 in all-L1/global scopes. Unknown usage is separately counted and is not claimed to be zero.',
        '|Scope|Input tokens, known sum|Output tokens, known sum|Cached input tokens, known sum|Results with unknown usage|', '|---|---|---|---|---|']
    for name, key in (('Valid revised L1R2', 'formal_L1'), ('Voided original L1', 'voided_L1'), ('All physical L1', 'all_L1'), ('Global H4 physical ledger', 'global')):
        values = accounting['tokens_by_scope'][key]
        lines.append('|' + name + '|' + '|'.join(str(values.get(token, 0)) for token in TOKEN_KEYS)
            + '|' + str(values['unknown_token_results']) + '|')
    lines += ['', '## Interpretation and audit limits', '',
        '- O, L0 and L1 each have one independent terminal TEST1000 pass; the total is three. This report reads no TEST labels and does not rescore O or L0.',
        '- Validation labels enter feedback and memory by design; they are not an untouched-label holdout. Validation membership grows, so aligned round curves compare different revealed sets.',
        '- L1-L0 reflects the complete memory, feedback and query trajectories; it does not isolate a causal effect of notes alone.',
        '- LOFO contributions below are saved winner-round internal-validation deltas. They do not use TEST performance, and no new LOFO fits were done for this report.',
        '- The winner and its saved validation score were fixed before uniform fill. TEST feature failure rates did not select or rank features.',
        '- Sample membership is seeded; local CLI output reproducibility is not claimed.',
        '', '## Winner-round validation contributions and full source', '',
        'Each seed lists up to five final-retained functions with finite saved LOFO deltas, descending delta; ties use original feature index. Positive delta means omission increased validation MAE. Negative deltas are retained in the ranking rather than hidden.']
    for seed in (0, 1, 2):
        item = top[str(seed)]
        lines += ['', '### Seed ' + str(seed), '',
            f"Winner round:{models[seed]['winner']['round_index']}; saved validation row MAE:{models[seed]['winner']['validation_mae']:.9f}."]
        if item['retained_without_winner_LOFO']:
            lines.append('Final-retained names without an available winner-round LOFO delta: ' + ', '.join(item['retained_without_winner_LOFO']) + '. They were not assigned invented contributions.')
        if not item['features']: lines.append('No final-retained function has a finite saved winner-round LOFO contribution.')
        for rank, feature in enumerate(item['features'], 1):
            source = feature['source']; fence = '`' * max(3, 1 + max([len(x) for x in re.findall(r'`+', source)] or [0]))
            lines += ['', f"#### {rank}. {feature['name']}", '',
                'MAE_without minus saved full-model MAE: ' + _number(feature['delta_without_minus_full'], signed=True) + ' eV/atom.',
                'Source SHA256: ' + feature['source_sha256'], fence + 'python', source, fence]
    _write(paths['report'], '\n'.join(lines) + '\n')
    return paths
