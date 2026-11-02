"""Reporting from already-completed L1/L0 metadata and statistical outputs.

No dataset/target accessor, model, generated feature, LOFO or bootstrap is run.
Only the explicitly called make_report writes, strictly below phase4_3/report.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
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
    rows = []
    for seed in (0, 1, 2):
        audits = _state(states, seed)['memory_audits']
        if len(audits) != 20 or {a['round_index'] for a in audits} != set(range(1, 21)):
            raise ReportIntegrityError('Complete per-request memory/input audits are required')
        for audit in sorted(audits, key=lambda a: a['round_index']):
            ordinary = audit['input_budget']; repaired = audit['format_repair_input_budget']
            if (audit['request_id'] != f"L1_seed{seed}_round{audit['round_index']:02d}"
                or type(audit['memory_token_count']) is not int or not 0 <= audit['memory_token_count'] <= 6000
                or type(audit['memory_UTF8_bytes']) is not int or audit['memory_UTF8_bytes'] < 0
                or type(audit['memory_trimmed']) is not bool
                or any(b['input_limit'] != input_limit or b.get('within_budget') is not True
                    or b['input_token_upper_bound'] > input_limit for b in (ordinary, repaired))):
                raise ReportIntegrityError('A saved memory/input audit violates the frozen completed-run bound')
            rows.append({'seed': seed, 'round': audit['round_index'], 'request_id': audit['request_id'],
                'memory_tokens': audit['memory_token_count'], 'memory_UTF8_bytes': audit['memory_UTF8_bytes'],
                'memory_trimmed': audit['memory_trimmed'],
                'ordinary_complete_UTF8_bound_bytes': ordinary['input_token_upper_bound'],
                'repair_complete_UTF8_bound_bytes': repaired['input_token_upper_bound'],
                'input_limit_bytes': input_limit, 'memory_limit_tokens': 6000})
    return rows


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

    memory_audits retains original request spec fields: request_id/round_index,
    memory_token_count/memory_UTF8_bytes/memory_trimmed, input_budget and
    format_repair_input_budget. cfg.input_limit_basis is reported as recorded.
    """
    if terminal.get('stage') != '4.3' or terminal.get('condition') != 'L1':
        raise ReportIntegrityError('Only committed L1 terminal output can be reported')
    if paired.get('registered_L0_minus_O_MAE_check', {}).get('status') != 'PASS' or set(paired['comparisons']) != set(PAIRS):
        raise ReportIntegrityError('All three authorized pairs and the registered L0-O reference must pass')
    if len(finals) != 3 or {item['seed'] for item in finals} != {0, 1, 2}:
        raise ReportIntegrityError('Report requires one final model for each seed')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ReportIntegrityError('Elapsed run time must be an existing finite nonnegative duration')
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
        'report': str(_output(report_dir / 'H4_stage4_3_report.md'))}
    _csv(paths['curve_csv'], curves, list(curves[0]))
    _csv(paths['request_input_ledger'], memory_rows, list(memory_rows[0]))
    _write(paths['top_feature_sources'], json.dumps(top, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    _plot(curves, paths['validation_curve'])
    lines = ['# H4 stage4.3 L1 results', 'Status: complete; stage4.4 waits for explicit continue.',
        'All values below come from saved terminal metrics, paired draws, round metadata and request ledgers. No report-time fit, feature execution, TEST access or resampling occurs.',
        f"Fixed TEST{terminal['n_test']}; {terminal['n_groups']} original template groups; {terminal['n_boot']} shared group bootstrap draws.",
        'MAE/RMSE units: eV/atom. R2 is dimensionless. Seed means are arithmetic means of metrics, never prediction ensembles.',
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
        'Memory tokens use the pre-frozen local counter on the exact retained canonical JSON. Memory UTF8 bytes and complete-request conservative UTF8 accounting are separate quantities; byte bounds are not measured model-token counts.',
        '|Seed|Requests|Max memory tokens|Max memory bytes|Prescribed trims|Max ordinary bound bytes|Max repair bound bytes|', '|---|---|---|---|---|---|---|']
    for seed in (0, 1, 2):
        own = [row for row in memory_rows if row['seed'] == seed]
        values = [seed, len(own), max(r['memory_tokens'] for r in own), max(r['memory_UTF8_bytes'] for r in own),
            sum(r['memory_trimmed'] for r in own), max(r['ordinary_complete_UTF8_bound_bytes'] for r in own), max(r['repair_complete_UTF8_bound_bytes'] for r in own)]
        lines.append('|' + '|'.join(str(v) for v in values) + '|')
    lines += ['', 'The full 60-request size ledger is [L1_request_input_ledger.csv](L1_request_input_ledger.csv).',
        'Frozen L1 complete-input bound: ' + str(input_limit) + ' bytes.',
        'Frozen accounting method: ' + cfg['prompts']['input_budget_method'],
        'L0 raised its limit during its round7 manual recovery. The L1 limit and memory counter were set before the first L1 call; old L0 prompts, results and rules are preserved.']
    if 'input_limit_basis' in cfg:
        lines += ['', 'Recorded pre-freeze measurement basis (displayed unchanged):', '```json',
            json.dumps(cfg['input_limit_basis'], ensure_ascii=False, indent=2, allow_nan=False), '```']
    else:
        lines.append('The input-limit measurement basis remains in the frozen config/projection; no report-time projection is performed.')
    global_calls = summary.get('cli_attempt_count', summary.get('n_calls'))
    if type(global_calls) is not int or not 120 <= global_calls <= 300:
        raise ReportIntegrityError('Global CLI accounting must include exactly120 prior O/L0 calls')
    kinds = summary['per_kind']; stage_tokens = kinds.get('features_query_memory')
    if stage_tokens is None: raise ReportIntegrityError('Missing L1 request-kind token ledger')
    global_tokens = {key: sum(row.get(key, 0) or 0 for row in kinds.values())
        for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')}
    lines += ['', '## Calls, tokens and elapsed time', '',
        f'Elapsed run time: {elapsed:.3f} seconds. Prior O/L0 CLI attempts:120; L1 attempts:{global_calls - 120}; global attempts:{global_calls}/300, including repairs and retries.',
        '|Scope|Input tokens|Output tokens|Cached input tokens|', '|---|---|---|---|']
    for name, values in (('L1 ledger', stage_tokens), ('Global H4 ledger', global_tokens)):
        lines.append('|' + name + '|' + '|'.join(str(values.get(key, 0) or 0) for key in ('input_tokens', 'output_tokens', 'cached_input_tokens')) + '|')
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
