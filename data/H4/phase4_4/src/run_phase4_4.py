"""Finite H4.4: freeze, train/predict, two exclusive gates, then stop before H5."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import io
import json
import os
import stat
import time
import traceback
import p44_common as c


class ForecastStop(RuntimeError):
    pass


def freeze():
    import p44_data as data
    import p44_models as models
    import p44_stats as stats
    if (c.PHASE / 'config.json').exists() or (c.PHASE / 'run_started.json').exists():
        raise RuntimeError('Existing freeze/run is immutable; no automatic reset')
    c.public_guard(full=True)
    definitions = data.prepare_definitions()
    model_cfg = models.configuration()
    statistics = stats.configuration()
    # Both operations below inspect stored metadata only; no target accessor.
    references = stats.readonly_existing_feature_pairs()
    audit = c.read(c.PHASE / 'audit/cgcnn_reuse_audit.json')
    decisions = {name: value['decision'] for name, value in audit['arms'].items()}
    if len(decisions) != 6 or sum(value == 'reuse' for value in decisions.values()) != 3:
        raise RuntimeError('Expected three approved n5000 reuses and three fresh n2000 arms')
    c.write(c.PHASE / 'audit/step1_complete.json', {'status': 'PASS', 'utc': c.utc(), 'reuse': decisions,
        'HOLDOUT_actual_historical_access_count_known': False, 'PermissionError_filename': 'cannot confirm',
        'no_training_or_evaluation_before_freeze': True}, exclusive=True)
    prior = {}
    for name, path in [('O', c.PHASE.parent / 'phase4_2/stage4_1_public_snapshot/metrics.json'),
                       ('L0', c.PHASE.parent / 'phase4_3/stage4_2_public_snapshot/metrics.json'),
                       ('L1', c.PUBLIC / 'metrics.json')]:
        value = c.read(path)
        prior[name] = {key: value[key] for key in ('elapsed_seconds', 'manual_elapsed_seconds',
            'historical_recorded_run_elapsed_seconds', 'original_R2_run_elapsed_seconds',
            'manual_terminal_elapsed_seconds') if key in value}
    source_hashes = {str(path.resolve()): c.sha(path) for path in (c.PHASE / 'src').glob('*.py') if not path.name.startswith('test_')}
    inputs = dict(definitions['source_hashes'])
    inputs.update(model_cfg['scientific_sources'])
    inputs.update(statistics['numerical_sources'])
    for path in (c.PHASE / 'audit').glob('*.json'):
        inputs[str(path.resolve())] = c.sha(path)
    old_plan_path = c.ROOT / 'work/h-data/H3b/cgcnn/plan.json'
    old_plan = c.read(old_plan_path)
    for key in ('graphs_path', 'graph_checkpoint_index_path'):
        path = Path(old_plan[key])
        inputs[str(path.resolve())] = c.sha(path)
    for condition, phase in [('O', 'phase4_1'), ('L0', 'phase4_2'), ('L1', 'phase4_3')]:
        for filename in ('evaluation_started.json', 'terminal_results.json', 'private_pairing_cache.npz'):
            path = c.PHASE.parent / phase / filename
            inputs[str(path.resolve())] = c.sha(path)
    inputs[references['source_path']] = references['source_sha256']
    cfg = {'protocol': 'H4-stage4.4-controls-transfer-first-freeze-v1', 'stage': '4.4', 'frozen_utc': c.utc(),
        'data': definitions, 'feature_sets': definitions['frozen_feature_sets'], 'models': model_cfg,
        'statistics': statistics, 'source_hashes': source_hashes, 'input_hashes': inputs,
        'prior_elapsed': prior, 'CGCNN_reuse_decisions': decisions,
        'forecast': {'threshold_hours': 10, 'historical_fresh_CGCNN_seconds': 2463.047,
            'historical_fresh_CGCNN_seconds_by_seed': [510.422, 384.563, 1568.062],
            'historical_CGCNN_hours_estimate': 2463.047 / 3600,
            'not_a_guarantee': True, 'first_real_arm_extrapolates_by_frozen_feature_count': True,
            'first_CGCNN_arm_updates_remaining_historical_ratio': True},
        'budget': {'LLM_calls_before': 183, 'LLM_global_cap': 300, 'new_LLM_calls_allowed': 0,
            'CGCNN_training_attempts': 3, 'CGCNN_training_limit_seconds_per_attempt': 7200, 'GPU_workers': 1},
        'seed_policy': '0/1/2 preserve frozen sample memberships/feature sets; HGB random_state20261018 remains fixed, no ensemble',
        'scope': {'public_directory_readonly': True, 'all_new_outputs': str(c.PHASE), 'next_stage_started': False,
            'complete_H4_then_wait': True},
        'HOLDOUT_history': {'actual_field_access_count': None, 'status': 'cannot confirm',
            'identified_prior_formal_stages': ['H2', 'H3', 'H3b'], 'current_H4_transfer_accessor_calls_planned': 1},
        'old_PermissionError': {'filename': None, 'status': 'cannot confirm',
            'reason': 'Original traceback recorded no failed filename; current reads do not establish original cause.'}}
    c.write(c.PHASE / 'config.json', cfg, exclusive=True)
    c.write(c.PHASE / 'frozen_config.json', {'utc': c.utc(), 'sha256': c.sha(c.PHASE / 'config.json'),
        'scientific_LLM_calls': 0, 'new_fit_predict_calls': 0, 'new_TEST_HOLDOUT_target_reads': 0}, exclusive=True)
    c.write(c.PHASE / 'README.md', '# H4 stage4.4\nStatus: frozen; training and scoring have not started.\nAll new outputs remain in phase4_4; H4_feature_loop and completed stages are readonly.\nNo new LLM calls are authorized; global count183/300.\nThree n5000 CGCNN arms reuse frozen predictions; three n2000 arms require new training.\nOriginal DFT geometry is used for real transfer.\nSynthetic TEST and real HOLDOUT each have one separate, exclusive final gate.\nHistorical HOLDOUT field-read count and original PermissionError filename cannot be confirmed.\nAfter completion H4 ends; H5 and other stages will not start.\n')
    return {'status': 'frozen', 'config_sha256': c.sha(c.PHASE / 'config.json'),
        'CGCNN_estimated_training_hours': cfg['forecast']['historical_CGCNN_hours_estimate'],
        'new_LLM_calls': 0, 'new_target_reads': 0}


def _npz(path, arrays):
    import numpy as np
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    c.write(path, output.getvalue(), exclusive=True)


def _prediction_csv(synthetic, real):
    text = io.StringIO(newline='')
    writer = csv.writer(text)
    writer.writerow(['domain', 'group', 'arm_id', 'seed', 'split', 'id', 'prediction_eV_per_atom'])
    for domain, arms in [('synthetic', synthetic), ('real', real)]:
        for arm in arms:
            for row in arm['prediction_rows']:
                writer.writerow([domain, arm['group'], arm['arm_id'], arm['seed'],
                    'TEST' if domain == 'synthetic' else 'HOLDOUT', row['id'], row['prediction_eV_per_atom']])
    c.write(c.PHASE / 'predictions.csv', text.getvalue(), exclusive=True)


def _prediction_artifacts(arms, bundle):
    """Read-only binding of derived caches and every existing fit/predict artifact."""
    for path, expected in bundle['source_hashes'].items():
        if c.sha(path) != expected:
            raise RuntimeError('A prediction input/cache changed before scoring: ' + path)
    for arm in arms:
        for reference in arm.get('artifacts', {}).values():
            if c.sha(reference['path']) != reference['sha256']:
                raise RuntimeError('A fitted HGB/prediction artifact changed: ' + reference['path'])
        for reference in arm.get('reuse_provenance', {}).get('artifacts', {}).values():
            if c.sha(reference['path']) != reference['sha256']:
                raise RuntimeError('A reused CGCNN artifact changed: ' + reference['path'])
        if arm['group'] == 'CGCNN_n2000':
            metadata = arm['metadata']
            directory = Path(metadata['attempt_folder'])
            if (c.sha(directory / 'latest_full_refit.pt') != metadata['evaluated_checkpoint_sha256']
                or c.sha(directory / 'prediction_rows.json') != metadata['prediction_rows_sha256']
                or metadata.get('test_targets_read') is not False or metadata.get('test_scored') is not False):
                raise RuntimeError('Fresh CGCNN checkpoint/predictions or no-scoring boundary changed')


def _identity():
    result = {'pid': os.getpid(), 'utc': c.utc(), 'command': 'run_phase4_4.py run', 'finite': True,
        'restart': False, 'task_scheduler_used': False}
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class FILETIME(ctypes.Structure):
            _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(FILETIME)] * 4
        kernel.GetProcessTimes.restype = wintypes.BOOL
        times = [FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *[ctypes.byref(value) for value in times]):
            raise OSError(ctypes.get_last_error(), 'GetProcessTimes failed')
        result['creation_time_100ns'] = times[0].low | (times[0].high << 32)
    return result


def run():
    import p44_data as data
    import p44_models as models
    import p44_stats as stats
    import p44_report as report
    cfg = c.verify_frozen(full=True)
    for name in ('run_started.json', 'evaluation_started.json', 'holdout_evaluation_started.json'):
        if (c.PHASE / name).exists():
            raise RuntimeError('An existing finite run or scoring gate forbids automatic restart: ' + name)
    started = time.monotonic()
    c.write(c.PHASE / 'process_identity.json', _identity(), exclusive=True)
    c.write(c.PHASE / 'run_started.json', {'utc': c.utc(), 'pid': os.getpid(),
        'config_sha256': c.sha(c.PHASE / 'config.json'), 'new_LLM_calls': 0}, exclusive=True)
    c.write(c.PHASE / 'metrics.json', {'status': 'running', 'stage': '4.4', 'step': 3,
        'new_LLM_calls': 0, 'global_CLI_calls': 183, 'TEST_accesses': 0, 'HOLDOUT_accesses': 0})

    def progress(value):
        c.public_guard()
        c.event('progress', **value)
        c.write(c.PHASE / 'current_operation.json', {'utc': c.utc(), **value})
        if value.get('phase') == 'HGB_arm_complete' and value.get('arm_id') == 'O_seed0':
            count = sum(len(record['features']) for group in cfg['feature_sets'].values() for record in group.values())
            first = len(cfg['feature_sets']['O']['0']['features'])
            expected = time.monotonic() - started + 1.2 * value['elapsed_seconds'] * max(0, count / first - 1) + 1.2 * 2463.047 + 60
            forecast = {'utc': c.utc(), 'method': 'Measured original-DFT first transfer arm, extrapolated by frozen feature-cell count, plus historical3 freshCGtime and20percent margin',
                'first_arm_seconds': value['elapsed_seconds'], 'estimated_total_hours': expected / 3600,
                'threshold_hours': 10, 'forecast_not_measured_total': True}
            c.write(c.PHASE / 'pilot_forecast.json', forecast, exclusive=True)
            if expected > 36000:
                raise ForecastStop('Measured first-transfer-arm forecast exceeds10hours; no scoring or continuation')
        if value.get('phase') == 'CGCNN_arm_complete' and value.get('arm_id') == 'CGCNN_n2000_seed0':
            expected = time.monotonic() - started + 1.2 * value['elapsed_seconds'] / 510.422 * (384.563 + 1568.062) + 60
            c.write(c.PHASE / 'CGCNN_first_arm_forecast.json', {'utc': c.utc(), 'estimated_total_hours': expected / 3600,
                'first_new_CGCNN_training_seconds': value['elapsed_seconds'], 'forecast_not_measured_total': True}, exclusive=True)
            if expected > 36000:
                raise ForecastStop('First new CGCNN arm forecasts more than10hours; no next arm or scoring')

    try:
        with c.prevent_sleep():
            c.event('step3_training_prediction_started', historical_CGCNN_forecast_hours=2463.047 / 3600)
            bundle = data.load_bundle(cfg, prepare_real_geometry=True, progress=progress)
            result = models.train_predict_all(cfg, bundle, progress=progress)
            synthetic, real = result['synthetic_arms'], result['real_arms']
            stats._normalise(synthetic, 'synthetic', bundle['synthetic_test_ids'])
            stats._normalise(real, 'real', bundle['real_holdout_ids'])
            _prediction_artifacts(synthetic + real, bundle)
            if any(data._TRUTH_STARTED.values()):
                raise RuntimeError('An evaluation target accessor started before predictions were committed')
            c.write(c.PHASE / 'committed_predictions.json', {'synthetic_arms': synthetic, 'real_arms': real,
                'evaluation_targets_read_before_commit': False}, exclusive=True)
            _prediction_csv(synthetic, real)
            c.write(c.PHASE / 'training_prediction_metadata.json', {
                'synthetic_arms': [{key: value for key, value in arm.items() if key != 'prediction_rows'} for arm in synthetic],
                'real_arms': [{key: value for key, value in arm.items() if key != 'prediction_rows'} for arm in real]}, exclusive=True)
            c.verify_frozen(full=True)
            prediction_hash = c.sha(c.PHASE / 'committed_predictions.json')
            statistics = cfg['statistics']
            marker = c.PHASE / 'evaluation_started.json'
            c.write(marker, {**statistics['domains']['synthetic']['marker_contract'], 'utc': c.utc(),
                'owner_pid': os.getpid(), 'config_sha256': c.sha(c.PHASE / 'config.json'),
                'committed_predictions_sha256': prediction_hash}, exclusive=True)
            c.event('step4_synthetic_scoring_started')
            syn = stats.evaluate_synthetic(synthetic, bundle, evaluation_marker=marker)
            c.write(c.PHASE / 'synthetic_terminal_results.json', syn, exclusive=True)
            _npz(c.PHASE / 'synthetic_pairing_cache.npz', stats.private_pairing_cache('synthetic'))
            c.write(c.PHASE / 'metrics.json', {'status': 'running', 'stage': '4.4', 'step': 5,
                'global_CLI_calls': 183, 'new_LLM_calls': 0, 'H4_TEST_passes': 4, 'TEST_accesses': 1, 'HOLDOUT_accesses': 0})
            c.public_guard(full=True)
            holdout_marker = c.PHASE / 'holdout_evaluation_started.json'
            c.write(holdout_marker, {**statistics['domains']['real']['marker_contract'], 'utc': c.utc(),
                'owner_pid': os.getpid(), 'config_sha256': c.sha(c.PHASE / 'config.json'),
                'committed_predictions_sha256': prediction_hash,
                'synthetic_evaluation_marker_sha256': c.sha(marker),
                'synthetic_terminal_sha256': c.sha(c.PHASE / 'synthetic_terminal_results.json'),
                'synthetic_pairing_cache_sha256': c.sha(c.PHASE / 'synthetic_pairing_cache.npz')}, exclusive=True)
            c.event('step5_real_HOLDOUT_scoring_started')
            real_result = stats.evaluate_real(real, bundle, evaluation_marker=holdout_marker)
            c.write(c.PHASE / 'real_terminal_results.json', real_result, exclusive=True)
            _npz(c.PHASE / 'real_pairing_cache.npz', stats.private_pairing_cache('real'))
            c.event('step6_cached_pairs_started')
            controls = stats.paired_controls_with_prior({**{condition: c.PHASE.parent / phase / 'private_pairing_cache.npz'
                for condition, phase in [('O', 'phase4_1'), ('L0', 'phase4_2'), ('L1', 'phase4_3')]},
                'controls': c.PHASE / 'synthetic_pairing_cache.npz'}, synthetic_marker=marker)
            transfer = stats.paired_real_transfer(c.PHASE / 'real_pairing_cache.npz', real_marker=holdout_marker)
            c.write(c.PHASE / 'paired_controls.json', controls, exclusive=True)
            c.write(c.PHASE / 'paired_transfer.json', transfer, exclusive=True)
            wall = time.monotonic() - started
            run_metadata = {'elapsed_wall_seconds': wall, 'new_LLM_calls': 0, 'new_CGCNN_fits': 3,
                'reused_CGCNN_predictions': 3, 'new_HGB_fits': 18, 'synthetic_TEST_accessor_calls': 1,
                'real_HOLDOUT_accessor_calls': 1, 'H4_TEST_passes': 4, 'next_stage_started': False}
            reports = report.make_report(cfg, syn, real_result, controls, transfer, run_metadata)
            c.verify_frozen(full=True)
            c.write(c.PHASE / 'metrics.json', {'status': 'complete', 'stage': '4.4', **run_metadata,
                'global_CLI_calls': 183, 'synthetic_results': syn, 'real_results': real_result,
                'reports': reports, 'historical_HOLDOUT_field_access_count': None,
                'existing_feature_pair_reference_check': 'PASS', 'H4_ended': True})
            c.write(c.PHASE / 'README.md', '# H4 stage4.4\nStatus: complete; H4 ended.\nAll outputs remain in phase4_4; the public H4 directory and completed phases are unchanged.\nLLM calls remain183/300; this phase adds0.\nCGCNN n5000:3 reused; n2000:3 newly trained because of prior repeated probe scoring.\nHGB controls and all O/L0/L1/Magpie transfer fits use the frozen parameters and TRAIN median.\nTransfer uses original DFT structures; HOLDOUT does not select features or models.\nH4 synthetic TEST passes3->4; this gate reads TEST once for12arms.\nThis H4 real transfer reads HOLDOUT once for12arms; the historical field-access total cannot be confirmed.\nOriginal PermissionError failed filename remains unrecorded and cannot be confirmed.\nAll2000 group-bootstrap intervals and cached paired differences are in report/H4_stage4_4_report.md.\nExisting validation curves and winner top5 source are referenced without new selection.\nNo H5 or other stage starts; wait for the user to decide the next step.\n')
            paths = [c.PHASE / name for name in ('config.json', 'frozen_config.json', 'predictions.csv', 'committed_predictions.json',
                'training_prediction_metadata.json', 'evaluation_started.json', 'holdout_evaluation_started.json',
                'synthetic_terminal_results.json', 'synthetic_pairing_cache.npz', 'real_terminal_results.json', 'real_pairing_cache.npz',
                'paired_controls.json', 'paired_transfer.json', 'metrics.json', 'README.md')]
            paths += [path for path in (c.PHASE / 'report').iterdir() if path.is_file()]
            outcome = {'status': 'complete', 'utc': c.utc(), **run_metadata,
                'output_sha256': {str(path): c.sha(path) for path in paths}, 'reports': reports}
            c.write(c.PHASE / 'process_finished.json', outcome, exclusive=True)
            c.event('stage4_4_complete_H4_ended', new_LLM_calls=0)
            return outcome
    except BaseException as error:
        stop = {'status': 'stopped_forecast' if isinstance(error, ForecastStop) else 'failed', 'utc': c.utc(),
            **c.permission_details(error), 'elapsed_wall_seconds': time.monotonic() - started,
            'synthetic_gate_exists': (c.PHASE / 'evaluation_started.json').exists(),
            'real_gate_exists': (c.PHASE / 'holdout_evaluation_started.json').exists(),
            'TEST_accessor_started': bool(data._TRUTH_STARTED.get('synthetic')),
            'HOLDOUT_accessor_started': bool(data._TRUTH_STARTED.get('real')),
            'automatic_restart_permitted': False, 'next_stage_started': False}
        c.write(c.PHASE / 'run_stop.json', stop, exclusive=True)
        c.write(c.PHASE / 'traceback.txt', traceback.format_exc(), exclusive=True)
        c.write(c.PHASE / 'metrics.json', stop)
        c.write(c.PHASE / 'process_finished.json', stop, exclusive=True)
        c.event('stage4_4_stopped', **stop)
        return stop


def verify():
    cfg = c.verify_frozen(full=True)
    if not (c.PHASE / 'process_finished.json').exists():
        return {'status': 'PASS', 'state': 'running_or_frozen', 'new_target_reads_from_verify': 0}
    finished = c.read(c.PHASE / 'process_finished.json')
    if finished['status'] != 'complete':
        return {'status': 'PASS', 'state': finished['status'], 'new_target_reads_from_verify': 0}
    for path, expected in finished['output_sha256'].items():
        if c.sha(path) != expected:
            raise RuntimeError('A completed output changed: ' + path)
    synthetic = c.read(c.PHASE / 'synthetic_terminal_results.json')
    real = c.read(c.PHASE / 'real_terminal_results.json')
    pairs = c.read(c.PHASE / 'paired_controls.json')
    if (synthetic['target_accessor_calls'] != 1 or real['target_accessor_calls'] != 1
        or pairs['old_feature_pairs']['status'] != 'PASS' or c.read(c.PHASE / 'metrics.json')['H4_TEST_passes'] != 4):
        raise RuntimeError('Final one-pass/reference accounting changed')
    return {'status': 'PASS', 'state': 'complete', 'new_target_reads_from_verify': 0,
        'H4_ended': True, 'next_stage_started': False, 'public_files_unchanged': True, 'global_CLI_calls': 183}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'verify'))
    args = parser.parse_args()
    result = freeze() if args.action == 'freeze' else run() if args.action == 'run' else verify()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
