"""Frozen stage4.4 training/prediction only; no evaluation-target accessor.

Imports are inert. Public runners require the stage's sealed config. All writes,
including subprocess temporary files, stay below phase4_4. Legacy CGCNN code
remains immutable; its one global lock is held read-only outside its exact-path
redirected internal lock. No failed arm is retried or repaired automatically.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
PHASE = ROOT / 'work/h-data/H4/phase4_4'
OLD = ROOT / 'work/h-data/H4/src'
for directory in (ROOT, OLD, Path(__file__).resolve().parent):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
from contextlib import contextmanager
from copy import deepcopy
import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import time
import traceback
import uuid
import numpy as np
import p44_common as c

HGB_PARAMETERS = dict(loss='squared_error', learning_rate=.05, max_iter=300,
    max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.,
    early_stopping=False, random_state=20261018)
GLOBAL_GPU_LOCK = ROOT / 'work/h-data/cgcnn_single_gpu.lock'
CG_SOURCE_PLAN = ROOT / 'work/h-data/H3b/cgcnn/plan_handshake_v2.json'


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _old_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _seal(value):
    return {**value, 'record_sha256': _hash(value)}


def _sealed(path):
    value = c.read(path); raw = dict(value); expected = raw.pop('record_sha256', None)
    if _hash(raw) != expected: raise RuntimeError('Completed model record self-hash changed: ' + str(path))
    return value


def _artifacts(value):
    for ref in value.get('artifacts', {}).values():
        if c.sha(ref['path']) != ref['sha256']:
            raise RuntimeError('Completed artifact changed; refuse repeated fit: ' + ref['path'])


def _reference(path):
    return {'path': str(c.safe(path)), 'sha256': c.sha(path)}


def _authorized(cfg):
    frozen = c.verify_frozen(full=False)
    if _hash(frozen) != _hash(cfg): raise RuntimeError('Caller configuration differs from frozen stage4.4 config')


def configuration():
    return {'HGB_parameters': HGB_PARAMETERS, 'HGB_fit_weights': 'unweighted',
        'imputer': 'TRAIN-only median; keep_empty_features=True; empty columns fill0',
        'real_feature_selection': 'Original final retained functions only; real TRAIN failure>20% drops; HOLDOUT diagnostic only',
        'sandbox_workers': 2, 'timeout_ms': 50, 'checkpoint_every': 100,
        'CGCNN_reuse': 'n5000 three complete arms only; n2000 three fresh attempts after repeated H4A299 scoring',
        'CGCNN_training_limit_seconds': 7200, 'GPU_workers': 1,
        'global_GPU_lock': 'Existing legacy byte lock opened rb; exact legacy write path redirected to new phase4_4 lock',
        'CGCNN_historical_three_n2000_training_seconds': 2463.047,
        'scientific_sources': {str(path): c.sha(path) for path in
            (Path(__file__), OLD/'h4_models.py', OLD/'h4_sandbox.py', ROOT/'h3_models.py',
             ROOT/'h3_sandbox.py', ROOT/'h3b_cgcnn.py', ROOT/'h3_cgcnn.py')}}


@contextmanager
def _environment(directory):
    names = ('TEMP', 'TMP', 'TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCH_HOME',
             'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'NUMBA_CACHE_DIR')
    before = {name: os.environ.get(name) for name in names}
    for name in names:
        target = c.safe(Path(directory)/name.lower(), write=True); target.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(target)
    try: yield
    finally:
        for name, value in before.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value


class _Sandbox:
    """Only the explicit checkpoint boundary is adapted; numerical sandbox unchanged."""
    def __init__(self, *args, **kwargs):
        import h4_sandbox
        if kwargs.get('n_workers') != 2 or kwargs.get('timeout_ms') != 50:
            raise RuntimeError('Frozen sandbox2/50ms boundary changed')
        self.runner = h4_sandbox.FeatureBatchRunner(*args, **kwargs)
    def __enter__(self): self.runner.__enter__(); return self
    def __exit__(self, *args): return self.runner.__exit__(*args)
    def evaluate_many(self, records, *, checkpoint_dir, checkpoint_every, progress):
        return self.runner.evaluate_many(records, checkpoint_dir=c.safe(checkpoint_dir, write=True),
            checkpoint_every=checkpoint_every, progress=progress)


def _fit(train, y, predict):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    import joblib
    started = time.monotonic()
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x = imputer.fit_transform(train); z = imputer.transform(predict)
    if not np.isfinite(x).all() or not np.isfinite(z).all(): raise ValueError('Nonfinite imputed model input')
    model = HistGradientBoostingRegressor(**HGB_PARAMETERS); model.fit(x, y)
    predictions = np.asarray(model.predict(z), dtype=np.float64)
    stream = io.BytesIO(); joblib.dump({'imputer': imputer, 'estimator': model}, stream, compress=3)
    return predictions, {'parameters': HGB_PARAMETERS, 'fit_weights': 'unweighted',
        'imputer_fit_n': len(y), 'fit_predict_seconds': time.monotonic()-started}, stream.getvalue()


def _hgb_arm(cfg, bundle, *, domain, group, seed, train_ids, predict_ids, labels,
             features, progress=None, model_fitter=None, sandbox_factory=None):
    """No truth value for predict_ids is accepted, loaded, stored or scored."""
    import h4_models as frozen
    if frozen.HGB_PARAMETERS != HGB_PARAMETERS: raise RuntimeError('Frozen HGB parameters changed')
    if set(train_ids) & set(predict_ids) or set(labels) != set(train_ids):
        raise ValueError('Explicit TRAIN targets and prediction identities must be disjoint and exact')
    if len(set(train_ids)) != len(train_ids) or len(set(predict_ids)) != len(predict_ids):
        raise ValueError('Duplicate train or prediction identities')
    features = deepcopy(features)
    if features: features = frozen._features(features)
    directory = c.safe(PHASE/'models'/domain/f'{group}_seed{seed}', write=True)
    y = np.asarray([labels[rid] for rid in train_ids], dtype=np.float64)
    if not np.isfinite(y).all(): raise ValueError('TRAIN targets must be finite')
    magpie = frozen._design(bundle, train_ids); test_magpie = frozen._design(bundle, predict_ids)
    identity = {'stage': '4.4', 'domain': domain, 'group': group, 'seed': seed,
        'train_ids': train_ids, 'prediction_ids': predict_ids, 'features': features,
        'labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
        'source_hashes': dict(bundle['source_hashes']), 'configuration_sha256': _hash(cfg),
        'parameters': HGB_PARAMETERS, 'source_sha256': c.sha(__file__),
        'TRAIN_feature_drop_threshold': .2, 'prediction_targets_read': False}
    complete = directory/'outcome.json'
    if complete.exists():
        result = _sealed(complete)
        if result.get('identity_sha256') != _hash(identity) or result.get('status') != 'complete':
            raise RuntimeError('Completed HGB binding mismatch; no refit permitted')
        _artifacts(result); return result
    started = time.monotonic(); c.write(directory/'started.json', {'identity': identity, 'utc': c.utc()})
    with _environment(PHASE/'environment/HGB'), c.prevent_sleep():
        train = frozen._execute_features(bundle, train_ids, features, directory/'TRAIN_checkpoint',
            progress, sandbox_factory or _Sandbox)
        holdout = frozen._execute_features(bundle, predict_ids, features, directory/'prediction_checkpoint',
            progress, sandbox_factory or _Sandbox)
        rates = np.mean(~np.isfinite(train['values']), axis=0)
        keep = rates <= .2
        x = np.column_stack((magpie, train['values'][:, keep]))
        z = np.column_stack((test_magpie, holdout['values'][:, keep]))
        predictions, fitted, model_bytes = (model_fitter or _fit)(x, y, z)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(predict_ids),) or not np.isfinite(predictions).all():
        raise RuntimeError('HGB must predict every frozen evaluation identity finitely')
    refs = {}
    for label, result in (('TRAIN_features', train), ('prediction_features', holdout)):
        stream = io.BytesIO(); np.savez_compressed(stream, ids=result['ids'], feature_names=result['feature_names'],
            values=result['values'], status=result['status'])
        path = directory/(label+'.npz'); c.write(path, stream.getvalue()); refs[label] = _reference(path)
    stream = io.BytesIO(); np.savez_compressed(stream, ids=np.asarray(predict_ids), predictions=predictions)
    path = directory/'predictions.npz'; c.write(path, stream.getvalue()); refs['predictions'] = _reference(path)
    path = directory/'model.joblib'; c.write(path, model_bytes); refs['model'] = _reference(path)
    metadata = {'arm_id': f'{group}_seed{seed}', 'group': group, 'seed': seed, 'dataset': domain,
        'condition': group, 'model': 'HGB', 'method': 'Magpie132+frozen_LLM' if features else 'Magpie132',
        'target': 'CHGNet' if domain == 'synthetic' else 'DFT', 'status': 'complete',
        'n_train': len(train_ids), 'n_prediction': len(predict_ids), 'identity': identity,
        'metadata_path': str(complete),
        'identity_sha256': _hash(identity), 'prediction_rows': [
            {'id': rid, 'prediction_eV_per_atom': float(value)} for rid, value in zip(predict_ids, predictions)],
        'feature_selection': {'original_features': features, 'TRAIN_failure_rates': rates.tolist(),
            'kept_names': [f['name'] for i, f in enumerate(features) if keep[i]],
            'dropped_names': [f['name'] for i, f in enumerate(features) if not keep[i]],
            'prediction_failure_rates': np.mean(~np.isfinite(holdout['values']), axis=0).tolist(),
            'prediction_failures_used_for_selection': False},
        'TRAIN_diagnostics': frozen._diagnostics(train), 'prediction_diagnostics': frozen._diagnostics(holdout),
        'fit': fitted, 'elapsed_seconds': time.monotonic()-started, 'artifacts': refs,
        'evaluation_target_values_read': False, 'evaluation_scoring_performed': False}
    result = _seal(metadata); c.write(complete, result)
    if progress: progress({'phase': 'HGB_arm_complete', 'arm_id': metadata['arm_id'], 'elapsed_seconds': metadata['elapsed_seconds']})
    return result


def run_hgb_controls(cfg, bundle, *, progress=None):
    _authorized(cfg)
    import p44_data
    labels = p44_data.synthetic_training_targets(cfg, bundle); arms = []
    for size in (2000, 5000):
        for seed in (0, 1, 2):
            ids = list(bundle['perseed_O_ids'][str(seed)] if size == 2000 else bundle['synthetic_pool_ids'])
            if len(ids) != size: raise ValueError('Frozen synthetic TRAIN size changed')
            arms.append(_hgb_arm(cfg, bundle, domain='synthetic', group=f'HGB_Magpie132_n{size}', seed=seed,
                train_ids=ids, predict_ids=list(bundle['synthetic_test_ids']), labels={rid: labels[rid] for rid in ids},
                features=[], progress=progress))
    return arms


def run_real_transfer(cfg, bundle, *, progress=None):
    _authorized(cfg)
    import p44_data
    labels = p44_data.real_training_targets(cfg, bundle); arms = []
    train, test = list(bundle['real_train_ids']), list(bundle['real_holdout_ids'])
    if len(train) != 2000 or len(test) != 2000: raise ValueError('Original DFT TRAIN/HOLDOUT2000 required')
    for group in ('O', 'L0', 'L1', 'Magpie_HGB'):
        for seed in (0, 1, 2):
            record = None if group == 'Magpie_HGB' else cfg['feature_sets'][group][str(seed)]
            features = [] if record is None else record['features']
            if record is not None:
                selected = _sealed(record['selected_metadata_path']); final = _sealed(record['final_metadata_path'])
                if selected['record_sha256'] != record['selected_record_sha256'] or final['record_sha256'] != record['final_record_sha256']:
                    raise RuntimeError('Frozen original final feature-set identity changed')
                actual = [f for f in selected['features'] if f['name'] in set(final['full_train_features']['kept_names'])]
                if actual != features: raise RuntimeError('Transfer must use exactly original final-kept complete feature definitions')
            arms.append(_hgb_arm(cfg, bundle, domain='real', group=group, seed=seed, train_ids=train,
                predict_ids=test, labels=labels, features=features, progress=progress))
    return arms


def _new_cg_plan(cfg, bundle, labels):
    old = c.read(CG_SOURCE_PLAN); plan = deepcopy(old)
    arms = {}; union = set()
    for seed in (0, 1, 2):
        spec = deepcopy(old['arms'][f'CGCNN_synthetic_trainCHGNet_n2000_seed{seed}'])
        if spec['train_ids'] != list(bundle['perseed_O_ids'][str(seed)]) or spec['test_ids'] != list(bundle['synthetic_test_ids']):
            raise RuntimeError('New CGCNN membership must match frozen H4 random2000/TEST1000')
        name = f'CGCNN_n2000_seed{seed}'; spec['arm_id'] = name; arms[name] = spec; union.update(spec['train_ids'])
    targets = {rid: labels[rid] for rid in sorted(union)}
    if set(targets) & set(bundle['synthetic_test_ids']): raise RuntimeError('CGCNN targets may contain TRAIN only')
    target_path = PHASE/'cgcnn/private_training_targets.json'; plan_path = PHASE/'cgcnn/plan.json'
    if plan_path.exists():
        current = c.read(plan_path)
        if current['parent_provenance']['frozen_stage4_4_sha256'] != _hash(cfg) or current['arms'] != arms:
            raise RuntimeError('Existing new CGCNN plan identity changed')
        if c.read(target_path) != targets: raise RuntimeError('New CGCNN TRAIN target binding changed')
        return plan_path
    c.write(target_path, targets, exclusive=True)
    plan.update(arms=arms, targets_path=str(target_path), targets_sha256=c.sha(target_path),
        source_hashes={name: c.sha(ROOT/name) for name in ('h3b_cgcnn.py', 'h3_cgcnn.py')},
        parent_provenance={'stage': '4.4', 'frozen_stage4_4_sha256': _hash(cfg),
            'prior_plan': _reference(CG_SOURCE_PLAN), 'old_checkpoint_or_predictions_loaded': False})
    plan.pop('technical_revision', None); plan.pop('plan_content_sha256', None)
    plan['plan_content_sha256'] = _old_hash(plan); c.write(plan_path, plan, exclusive=True)
    return plan_path


@contextmanager
def _read_only_global_gpu_lock(path=GLOBAL_GPU_LOCK):
    """Windows LK_NBLCK works on an existing read-only descriptor; never append."""
    import msvcrt
    path = c.safe(path); before = (c.sha(path), path.stat().st_mtime_ns, path.stat().st_size)
    try:
        with path.open('rb') as stream:
            if not stream.read(1): raise RuntimeError('Existing global GPU lock must have a lockable byte')
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try: yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    finally:
        if (c.sha(path), path.stat().st_mtime_ns, path.stat().st_size) != before:
            raise RuntimeError('Legacy GPU lock bytes/mtime changed; stage must stop')


def _cg_child(args):
    import h3b_cgcnn as legacy
    ready, gate = c.safe(args.ready, write=True), c.safe(args.gate, write=True)
    if ready.exists(): raise RuntimeError('CG adapter refuses an existing readiness record')
    c.write(ready, {'worker_pid': os.getpid(), 'parent_pid': os.getppid(),
        'creation_time': legacy._creation_time(), 'nonce': args.nonce})
    deadline = time.monotonic()+30
    while not gate.exists():
        if time.monotonic() > deadline: raise TimeoutError('CG adapter startup gate timed out')
        time.sleep(.025)
    admission, _gate_retry = legacy._read_handshake_json(gate)
    if admission['nonce'] != args.nonce or admission['pid'] != os.getpid() or not legacy._in_named_job(admission['job_name']):
        raise RuntimeError('CG adapter actual PID is not owned by the supervisor Job')
    cfg = c.verify_frozen(full=False)
    if c.read(args.plan)['parent_provenance']['frozen_stage4_4_sha256'] != _hash(cfg):
        raise RuntimeError('CG adapter plan is not bound to the sealed stage4.4 configuration')
    old_safe = legacy._safe
    def isolated(path):
        resolved = Path(path).resolve()
        if resolved == GLOBAL_GPU_LOCK.resolve(): return c.safe(PHASE/'cgcnn/gpu_internal.lock', write=True)
        # Legacy writes are made only to the new plan/attempt/environment paths.
        return old_safe(path)
    legacy._safe = isolated
    try:
        with _read_only_global_gpu_lock(), _environment(PHASE/'environment/CGCNN'):
            result = legacy.run_arm_supervised(args.plan, args.arm, allow_retry=False)
        c.write(Path(args.ready).parent/'adapter_complete.json', {'status': result['status'], 'legacy_global_lock_unchanged': True})
        if result['status'] != 'complete': raise RuntimeError('A new CGCNN arm did not complete; no scoring or automatic repair allowed')
    except BaseException as error:
        c.write(Path(args.ready).parent/'adapter_error.json', {**c.permission_details(error),
            'path_context': str(Path(args.plan).resolve()), 'arm': args.arm, 'traceback': traceback.format_exc(),
            'filename_unavailable': getattr(error, 'filename', None) is None,
            'worker_error_files': [str(p.resolve()) for p in (Path(args.plan).parent/'arms'/args.arm).glob('attempt_*/*error.json')],
            'automatic_retry': False, 'legacy_global_lock_write_attempted': False})
        raise
    finally: legacy._safe = old_safe


def _launch_cg(plan, arm):
    import h3b_cgcnn as legacy
    directory = c.safe(PHASE/'cgcnn/adapter'/arm, write=True); directory.mkdir(parents=True, exist_ok=True)
    ready, gate = directory/'ready.json', directory/'gate.json'; nonce = uuid.uuid4().hex
    if ready.exists() or gate.exists(): raise RuntimeError('Existing unfinished CG adapter requires explicit manual decision')
    name = 'Local\\Codex.H4p44.'+nonce; job = legacy._WindowsJob(name); process = None
    started = time.monotonic(); last_guard = started
    global_before = (c.sha(GLOBAL_GPU_LOCK), c.safe(GLOBAL_GPU_LOCK).stat().st_mtime_ns,
                     c.safe(GLOBAL_GPU_LOCK).stat().st_size)
    try:
        with _environment(PHASE/'environment/CGCNN'), c.safe(directory/'stdout.log', write=True).open('wb') as stdout, c.safe(directory/'stderr.log', write=True).open('wb') as stderr:
            process = subprocess.Popen([str(ROOT/'work/p1-env/Scripts/pythonw.exe'), '-B', str(Path(__file__).resolve()),
                'cg-child', '--plan', str(plan), '--arm', arm, '--ready', str(ready), '--gate', str(gate), '--nonce', nonce],
                cwd=str(ROOT), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8'),
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign_pid(process.pid)
            while not ready.exists():
                if process.poll() is not None: raise RuntimeError('CG adapter exited before owned-PID admission')
                if time.monotonic()-started > 30: raise TimeoutError('CG adapter readiness timed out')
                if time.monotonic()-last_guard >= 2:
                    c.public_guard(); last_guard = time.monotonic()
                time.sleep(.025)
            identity, _ready_retry = legacy._read_handshake_json(ready)
            if identity['nonce'] != nonce or identity['parent_pid'] != process.pid: raise RuntimeError('Foreign CG adapter readiness')
            job.assign_pid(identity['worker_pid'], identity['creation_time'])
            if gate.exists(): raise RuntimeError('CG supervisor refuses an existing admission gate')
            c.write(gate, {'nonce': nonce, 'pid': identity['worker_pid'], 'job_name': name})
            while process.poll() is None:
                if time.monotonic()-started > 7500: raise TimeoutError('CG adapter exceeded finite train7200+prediction180+startup allowance')
                if time.monotonic()-last_guard >= 2:
                    c.public_guard(); last_guard = time.monotonic()
                time.sleep(.25)
            if process.returncode: raise RuntimeError('CG adapter failed; inspect explicit per-arm error/worker logs')
    except BaseException as error:
        c.write(directory/'supervisor_error.json', {**c.permission_details(error), 'path_context': str(plan),
            'arm': arm, 'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
    finally:
        job.close()
        if process is not None and process.poll() is None: process.kill(); process.wait(timeout=15)
        global_after = (c.sha(GLOBAL_GPU_LOCK), c.safe(GLOBAL_GPU_LOCK).stat().st_mtime_ns,
                        c.safe(GLOBAL_GPU_LOCK).stat().st_size)
        c.write(directory/'global_lock_integrity.json', {'status': 'PASS' if global_before == global_after else 'FAIL',
            'path': str(c.safe(GLOBAL_GPU_LOCK)), 'before_sha256_mtime_ns_bytes': list(global_before),
            'after_sha256_mtime_ns_bytes': list(global_after), 'opened_for_writing': False})
        if global_before != global_after: raise RuntimeError('Existing global GPU lock changed during the new arm')


def run_cgcnn(cfg, bundle, *, progress=None):
    _authorized(cfg)
    import p44_data
    audit = c.read(PHASE/'audit/cgcnn_reuse_audit.json'); arms = []
    for seed in (0, 1, 2):
        spec = audit['arms'][f'CGCNN_synthetic_trainCHGNet_n5000_seed{seed}']
        if spec['decision'] != 'reuse': raise RuntimeError('CGCNN5000 reuse decision changed')
        for ref in spec['artifacts'].values():
            if c.sha(ref['path']) != ref['sha256']: raise RuntimeError('Approved reused CGCNN artifact changed')
        rows = c.read(spec['artifacts']['predictions']['path'])
        if [r['id'] for r in rows] != list(bundle['synthetic_test_ids']): raise RuntimeError('Reused CGCNN must cover the exact TEST1000')
        arms.append({'arm_id': f'CGCNN_n5000_seed{seed}', 'group': 'CGCNN_n5000', 'seed': seed,
            'dataset': 'synthetic', 'method': 'CGCNN', 'status': 'reused_complete', 'n_train': 5000,
            'prediction_rows': [{'id': r['id'], 'prediction_eV_per_atom': r['prediction_eV_per_atom']} for r in rows],
            'reuse_provenance': spec, 'evaluation_targets_read': False})
    plan = _new_cg_plan(cfg, bundle, p44_data.synthetic_training_targets(cfg, bundle))
    for seed in (0, 1, 2):
        name = f'CGCNN_n2000_seed{seed}'; result_path = plan.parent/'arms'/name/'outcome.json'
        if not result_path.exists(): _launch_cg(plan, name)
        result = c.read(result_path)
        if result['status'] != 'complete': raise RuntimeError('Incomplete CGCNN arm forbids TEST scoring; no automatic retry')
        spec = c.read(plan)['arms'][name]
        if (result['metadata'].get('plan_sha256') != c.sha(plan)
            or result['metadata'].get('normalizer_fit_n') != 2000
            or result['metadata'].get('test_targets_read') is not False
            or result['metadata'].get('test_scored') is not False):
            raise RuntimeError('New CGCNN training/normalizer/target gate metadata changed')
        for fname, key in (('latest_full_refit.pt', 'evaluated_checkpoint_sha256'),
                           ('prediction_rows.json', 'prediction_rows_sha256'),
                           ('training_prediction_rows.json', 'training_prediction_rows_sha256')):
            if c.sha(Path(result['metadata']['attempt_folder'])/fname) != result['metadata'][key]: raise RuntimeError('New CGCNN artifact changed')
        if (c.read(Path(result['metadata']['attempt_folder'])/'prediction_rows.json') != result['prediction_rows']
            or [r['id'] for r in result['training_prediction_rows']] != spec['train_ids']):
            raise RuntimeError('Saved new CGCNN prediction membership or immutable row payload changed')
        if [r['id'] for r in result['prediction_rows']] != list(bundle['synthetic_test_ids']): raise RuntimeError('New CGCNN evaluation identities changed')
        folder = c.safe(result['metadata']['attempt_folder'], write=True)
        marker = c.read(folder/'evaluation_started.json')
        if (marker.get('arm') != name or marker.get('n_TEST') != 1000 or marker.get('n_TRAIN') != 2000
            or marker.get('checkpoint_sha256') != result['metadata']['evaluated_checkpoint_sha256']):
            raise RuntimeError('New CGCNN prediction gate/checkpoint identity changed')
        refs = {'plan': _reference(plan), 'outcome': _reference(result_path)}
        refs.update({fname: _reference(folder/fname) for fname in ('latest_full_refit.pt',
            'prediction_rows.json', 'training_prediction_rows.json', 'training_result.json', 'evaluation_started.json')})
        arms.append({'arm_id': name, 'group': 'CGCNN_n2000', 'seed': seed, 'dataset': 'synthetic',
            'method': 'CGCNN', 'status': 'complete', 'n_train': 2000, 'metadata': result['metadata'],
            'artifacts': refs,
            'prediction_rows': [{'id': r['id'], 'prediction_eV_per_atom': r['prediction_eV_per_atom']} for r in result['prediction_rows']]})
        if progress: progress({'phase': 'CGCNN_arm_complete', 'arm_id': name, 'elapsed_seconds': result['metadata']['elapsed_seconds']})
    c.write(PHASE/'cgcnn/committed_arms.json', arms)
    return arms


def train_predict_all(cfg, bundle, *, progress=None):
    synthetic = run_hgb_controls(cfg, bundle, progress=progress)
    real = run_real_transfer(cfg, bundle, progress=progress)
    synthetic.extend(run_cgcnn(cfg, bundle, progress=progress))
    return {'synthetic_arms': synthetic, 'real_arms': real,
        'artifacts': {arm['arm_id']: arm.get('artifacts', arm.get('reuse_provenance', {}).get('artifacts', {}))
            for arm in synthetic+real}, 'new_LLM_calls': 0, 'evaluation_targets_read': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('cg-child',))
    for name in ('plan', 'arm', 'ready', 'gate', 'nonce'): parser.add_argument('--'+name, required=True)
    arguments = parser.parse_args()
    try: _cg_child(arguments)
    except BaseException as error:
        c.write(Path(arguments.ready).parent/'adapter_entry_error.json', {**c.permission_details(error),
            'path_context': str(Path(arguments.plan).resolve()), 'arm': arguments.arm,
            'filename_unavailable': getattr(error, 'filename', None) is None,
            'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
