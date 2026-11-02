"""H4.3 dynamic-label validation and winner-only prediction, without TEST truth.

The frozen H4.1 modules remain untouched. Their numeric HGB/sandbox helpers are
reused under a separately bound H4.3 protocol and H4.3-only output boundary.
No generated code, model fitting or target access occurs on import.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import os
import sys
import time

sys.dont_write_bytecode = True
for _thread_name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_name] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / 'work/h-data/H4/src') not in sys.path:sys.path.insert(0,str(ROOT / 'work/h-data/H4/src'))
import h4_models as base

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "work/h-data/H4/phase4_3"
HGB_PARAMETERS = dict(base.HGB_PARAMETERS)
FAILURE_RATE_THRESHOLD = .20
MODEL_PROTOCOL = "H4.3-dynamic-label-fixed-template-map-20rounds-v1"


def safe(path):
    target = Path(path).resolve()
    root = DATA.resolve()
    if "onedrive" in str(target).casefold() or not target.is_relative_to(root):
        raise ValueError("H4.3 model output must stay under non-OneDrive phase4_3")
    return target


def _write_json(path, value):
    base._write_json(safe(path), value)


def _write_npz(path, **arrays):
    base._write_npz(safe(path), **arrays)


def _code_binding():
    return {"models_sha256": base._file_hash(__file__),
            "frozen_H4_1_model_helper_sha256": base._file_hash(base.__file__),
            "frozen_L0_model_source_sha256": base._file_hash(ROOT / "work/h-data/H4/src/h4_l0_models.py"),
            "L1_data_source_sha256": base._file_hash(Path(__file__).with_name("l1_data.py")),
            "sandbox_sha256": base._file_hash(ROOT / "work/h-data/H4/src/h4_sandbox.py"),
            "original_ast_policy_sha256": base._file_hash(ROOT / "h3_sandbox.py"),
            "frozen_h3_hgb_source_sha256": base._file_hash(ROOT / "h3_models.py")}


def _split(bundle, seed, labels, *, initial=False, round_index=None, final=False):
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Seed must be0,1,2")
    if not isinstance(labels, dict) or not all(type(rid) is str for rid in labels):
        raise ValueError("Labelled data must be an ID-to-finite-target mapping")
    ids = sorted(labels)
    pool = set(bundle["pool_ids"])
    if not set(ids) <= pool or set(ids) & set(bundle["test_ids"]):
        raise ValueError("Only currently revealed POOL targets may enter H4.3 validation")
    seed_data = bundle["perseed"][seed]
    initial_ids = set(seed_data["S0_ids"])
    if len(initial_ids) != 500 or not initial_ids <= set(ids):
        raise ValueError("Current labelled membership must include the exact frozen S0500")
    if not 500 <= len(ids) <= 2000 or initial and set(ids) != initial_ids:
        raise ValueError("Invalid initial/current labelled membership")
    if final and len(ids) != 2000:
        raise ValueError("Final TRAIN must contain exactly2000 labelled POOL members")
    if round_index is not None:
        if type(round_index) is not int or not 1 <= round_index <= 20:
            raise ValueError("Round index must be1..20")
        if len(ids) > 500 + 75 * round_index:
            raise ValueError("Round labelled count exceeds its cumulative75-label quota")
    if any(type(value) not in (float, int) or not math.isfinite(value) for value in labels.values()):
        raise ValueError("Only finite labelled POOL targets may be supplied")
    mapping = seed_data["fit_group_mapping"]
    all_pool_groups = {bundle["groups"][rid] for rid in bundle["pool_ids"]}
    if set(mapping) != all_pool_groups or any(value not in ("fit", "val") for value in mapping.values()):
        raise ValueError("The fixed fit/val assignment must cover every POOL template group")
    fit_ids = [rid for rid in ids if mapping[bundle["groups"][rid]] == "fit"]
    val_ids = [rid for rid in ids if mapping[bundle["groups"][rid]] == "val"]
    if not fit_ids or not val_ids:
        raise ValueError("Fixed template assignment leaves fit or validation empty")
    if set(fit_ids) & set(val_ids) or set(fit_ids) | set(val_ids) != set(ids):
        raise ValueError("Fixed template assignment failed to partition labelled members")
    index = {rid: i for i, rid in enumerate(ids)}
    y = np.asarray([labels[rid] for rid in ids], dtype=np.float64)
    return {"ids": ids, "fit_ids": fit_ids, "val_ids": val_ids, "y": y,
            "fit": np.array([index[rid] for rid in fit_ids], dtype=int),
            "val": np.array([index[rid] for rid in val_ids], dtype=int),
            "fixed_template_assignment_sha256": base._json_hash(mapping),
            "fixed_template_assignment_cutoff": seed_data.get("cutoff")}


def _identity(bundle, seed, split, features, phase, round_index=None):
    matrix = base._design(bundle, split["ids"])
    identity = {"protocol": MODEL_PROTOCOL, "stage": "4.3", "condition": "L1",
                "seed": seed, "phase": phase, "round_index": round_index,
                "features": features, "train_ids": split["ids"],
                "fit_ids": split["fit_ids"], "val_ids": split["val_ids"],
                "train_labels_sha256": base._matrix_hash(split["y"]),
                "train_magpie132_sha256": base._matrix_hash(matrix),
                "fixed_template_assignment_sha256": split["fixed_template_assignment_sha256"],
                "fixed_template_assignment_cutoff": split["fixed_template_assignment_cutoff"],
                "frozen_data_source_hashes": dict(bundle.get("source_hashes", {})),
                "parameters": HGB_PARAMETERS, "code": _code_binding(),
                "failure_rate_threshold": FAILURE_RATE_THRESHOLD,
                "sandbox_workers": 2, "deadline_ms": 50,
                "selection_metric": "unweighted current-round validation row MAE",
                "TEST_target_values_read": False}
    return identity, matrix


def _cached(directory, filename, binding):
    path = safe(directory / filename)
    if not path.exists():
        return None
    metadata = base._read_sealed(path)
    if metadata.get("identity_sha256") != binding or metadata.get("status") != "complete":
        raise RuntimeError("Completed H4.3 result binding changed; refusing rerun")
    base._check_artifacts(metadata)
    return metadata


def _worst20(split, predictions):
    truth = split["y"][split["val"]]
    errors = np.abs(predictions - truth)
    order = sorted(range(len(split["val_ids"])),
                   key=lambda i: (-float(errors[i]), split["val_ids"][i]))[:20]
    return [{"id": split["val_ids"][i],
             "prediction_eV_per_atom": float(predictions[i]),
             "label_eV_per_atom": float(truth[i]),
             "absolute_error_eV_per_atom": float(errors[i])} for i in order]


def _validate(bundle, seed, split, features, output_dir, phase, round_index,
              progress, sandbox_factory, model_fitter):
    directory = safe(output_dir)
    identity, magpie = _identity(bundle, seed, split, features, phase, round_index)
    binding = base._json_hash(identity)
    filename = "initial_state.json" if phase == "initial" else "round.json"
    cached = _cached(directory, filename, binding)
    if cached is not None:
        return cached
    started = time.perf_counter()
    _write_json(directory / "started.json", base._seal({"identity": identity,
                "identity_sha256": binding, "whole_unfinished_round_resume_only": True}))
    base._emit(progress, "initial_validation" if phase == "initial" else "round_features",
               seed=seed, round_index=round_index, labelled_count=len(split["ids"]))
    result = base._execute_features(bundle, split["ids"], features,
                  safe(directory / "sandbox_checkpoint"), progress, sandbox_factory)
    values = np.asarray(result["values"], dtype=np.float64)
    inner = base._feature_metadata(result, split["fit"])
    validation = base._feature_metadata(result, split["val"])
    keep = np.asarray(inner["failure_rates"]) <= FAILURE_RATE_THRESHOLD
    matrix = np.column_stack((magpie, values[:, keep]))
    predictions, fit = base._predict(model_fitter, matrix[split["fit"]],
                                   split["y"][split["fit"]], matrix[split["val"]])
    val_truth = split["y"][split["val"]]
    row_mae = float(np.mean(np.abs(predictions - val_truth)))
    group_mae = base._group_mae(val_truth, predictions,
                     [bundle["groups"][rid] for rid in split["val_ids"]])
    feature_path = safe(directory / "train_features.npz")
    _write_npz(feature_path, ids=np.asarray(split["ids"]), feature_names=result["feature_names"],
               values=values, status=result["status"])
    prediction_path = safe(directory / "validation_predictions.npz")
    _write_npz(prediction_path, ids=np.asarray(split["val_ids"]),
               predictions=predictions, truth=val_truth)
    names = [feature["name"] for feature in features]
    worst = _worst20(split, predictions)
    metadata = base._seal({"schema_version": "H4.3-model-validation-v1", "status": "complete",
        "identity": identity, "identity_sha256": binding, "stage": "4.3", "condition": "L1",
        "phase": phase, "seed": seed, "round_index": round_index, "features": features,
        "active_features": [feature for feature, flag in zip(features, keep) if flag],
        "winner_eligible": phase == "round", "labelled_count": len(split["ids"]),
        "fit_count": len(split["fit_ids"]), "validation_count": len(split["val_ids"]),
        "validation_mae": row_mae, "validation_group_mae": group_mae,
        "worst20_ids": [row["id"] for row in worst], "worst20": worst,
        "inner_fit_features": inner, "validation_features": validation,
        "full_round_train_features": base._feature_metadata(result, np.arange(len(split["ids"]))),
        "feature_selection": {"domain": "current-round inner-fit TRAIN only",
            "threshold": .20, "drop_comparison": ">",
            "kept_names": [name for name, flag in zip(names, keep) if flag],
            "dropped_names": [name for name, flag in zip(names, keep) if not flag],
            "fallback_to_magpie": not bool(keep.any()),
            "validation_failure_rates_used_for_selection": False,
            "TEST_failure_rates_used_for_selection": False},
        "fit": fit, "diagnostics": base._diagnostics(result),
        "elapsed_seconds": time.perf_counter() - started,
        "metadata_path": str(safe(directory / filename)),
        "artifacts": {"train_features": {"path": str(feature_path), "sha256": base._file_hash(feature_path)},
            "validation_predictions": {"path": str(prediction_path), "sha256": base._file_hash(prediction_path)}},
        "TEST_structures_executed": 0, "TEST_target_values_read": False})
    _write_json(directory / filename, metadata)
    base._emit(progress, "initial_validation_complete" if phase == "initial" else "round_complete",
               seed=seed, round_index=round_index, labelled_count=len(split["ids"]), validation_mae=row_mae)
    return metadata


def _fit_new_initial_state(bundle, seed, labels, output_dir, progress=None, *, model_fitter=None):
    """Magpie-only S0 feedback, with no generated code and no LLM request.

    This baseline is strictly excluded from the twenty-round winner set.
    """
    split = _split(bundle, seed, labels, initial=True)
    return _validate(bundle, seed, split, [], output_dir, "initial", None,
                     progress, None, model_fitter)


def _evaluate_round_baseline(bundle, seed, round_index, features, labels, output_dir, progress=None,
                   *, sandbox_factory=None, model_fitter=None):
    """Validate returned features after this round's authorized label reveal."""
    features = base._features(features)
    split = _split(bundle, seed, labels, round_index=round_index)
    return _validate(bundle, seed, split, features, output_dir, "round", round_index,
                     progress, sandbox_factory, model_fitter)


def select_best(rounds):
    """Exactly twenty saved round MAEs; no post-fill revalidation or reranking."""
    rounds = list(rounds)
    if len(rounds) != 20 or {item.get("round_index") for item in rounds} != set(range(1, 21)):
        raise ValueError("H4.3 selection requires exactly20 complete rounds1..20")
    if len({item.get("seed") for item in rounds}) != 1:
        raise ValueError("Every round must belong to the same seed")
    assignments = {item["identity"]["fixed_template_assignment_sha256"] for item in rounds}
    if len(assignments) != 1:
        raise ValueError("Fixed template assignment differs across rounds")
    for item in rounds:
        if (item.get("status") != "complete" or not item.get("winner_eligible")
                or item.get("phase") != "round" or not math.isfinite(item["validation_mae"])):
            raise ValueError("Only complete finite round-validation candidates may compete")
    ordered = sorted(rounds, key=lambda item: item["round_index"])
    for previous, current in zip(ordered, ordered[1:]):
        if not set(previous["identity"]["train_ids"]) <= set(current["identity"]["train_ids"]):
            raise ValueError("Labelled memberships must grow monotonically across rounds")
    return min(rounds, key=lambda item: (item["validation_mae"], item["round_index"]))


def finalize(bundle, seed, selected, final_labels, test_ids, output_dir, progress=None,
             *, sandbox_factory=None, model_fitter=None):
    """Extend winner TRAIN cache to final2000, refit and predict TEST1000 once.

    The chosen round and its saved validation MAE remain unchanged after the
    round20 uniform fill. Only winner training rows absent from its round cache
    execute generated functions; completed old cells are immutable evidence.
    """
    cached_round = base._read_sealed(safe(selected["metadata_path"]))
    if cached_round["record_sha256"] != selected["record_sha256"]:
        raise RuntimeError("Selected round changed after selection")
    if cached_round.get("phase") != "round" or not cached_round.get("winner_eligible"):
        raise ValueError("Initial Magpie-only state cannot be a final winner")
    if cached_round["seed"] != seed:
        raise ValueError("Selected round belongs to another seed")
    base._check_artifacts(cached_round)
    split = _split(bundle, seed, final_labels, final=True)
    previous = cached_round["identity"]
    if (previous["code"] != _code_binding()
            or previous["frozen_data_source_hashes"] != dict(bundle.get("source_hashes", {}))
            or previous["fixed_template_assignment_sha256"] != split["fixed_template_assignment_sha256"]):
        raise RuntimeError("Frozen source, numeric policy or split changed after selected round")
    old_ids = previous["train_ids"]
    if not set(old_ids) <= set(split["ids"]):
        raise ValueError("Final labelled set lost members from the selected round")
    if base._matrix_hash([final_labels[rid] for rid in old_ids]) != previous["train_labels_sha256"]:
        raise RuntimeError("Previously revealed selected-round labels changed")
    if base._matrix_hash(base._design(bundle, old_ids)) != previous["train_magpie132_sha256"]:
        raise RuntimeError("Selected-round Magpie rows changed")
    test_ids = base._ids(test_ids, "TEST")
    if len(test_ids) != 1000 or test_ids != [str(rid) for rid in bundle["test_ids"]]:
        raise ValueError("Final prediction requires the exact frozen ordered TEST1000")
    extra_ids = sorted(set(split["ids"]) - set(old_ids))
    train_magpie, test_magpie = base._design(bundle, split["ids"]), base._design(bundle, test_ids)
    identity = {"protocol": MODEL_PROTOCOL, "stage": "4.3", "condition": "L1", "seed": seed,
        "selected_round_index": cached_round["round_index"],
        "selected_round_record_sha256": cached_round["record_sha256"],
        "selected_round_validation_mae": cached_round["validation_mae"],
        "selected_round_train_cache": cached_round["artifacts"]["train_features"],
        "train_ids": split["ids"], "additional_train_execution_ids": extra_ids,
        "train_labels_sha256": base._matrix_hash(split["y"]),
        "train_magpie132_sha256": base._matrix_hash(train_magpie),
        "test_ids": test_ids, "test_magpie132_sha256": base._matrix_hash(test_magpie),
        "fixed_template_assignment_sha256": split["fixed_template_assignment_sha256"],
        "frozen_data_source_hashes": dict(bundle.get("source_hashes", {})),
        "parameters": HGB_PARAMETERS, "code": _code_binding(),
        "fill_changes_saved_validation_or_winner": False, "TEST_target_values_read": False}
    binding = base._json_hash(identity)
    directory = safe(output_dir)
    cached = _cached(directory, "final_predictions.json", binding)
    if cached is not None:
        return cached
    started = time.perf_counter()
    _write_json(directory / "started.json", base._seal({"identity": identity, "identity_sha256": binding}))
    with np.load(cached_round["artifacts"]["train_features"]["path"], allow_pickle=False) as archive:
        if archive["ids"].tolist() != old_ids:
            raise RuntimeError("Selected-round feature-cache identities changed")
        old_values, old_status = archive["values"].copy(), archive["status"].copy()
        names = archive["feature_names"].tolist()
    features = cached_round["features"]
    if names != [item["name"] for item in features]:
        raise RuntimeError("Selected-round feature names/order changed")
    extra = None
    if extra_ids:
        base._emit(progress, "winner_additional_train_features", seed=seed, total=len(extra_ids))
        extra = base._execute_features(bundle, extra_ids, features,
                   safe(directory / "additional_train_checkpoint"), progress, sandbox_factory)
    positions = {rid: i for i, rid in enumerate(split["ids"])}
    train_values = np.full((2000, len(names)), np.nan)
    train_status = np.full((2000, len(names)), "pending", dtype="<U32")
    for rid, values, status in zip(old_ids, old_values, old_status):
        train_values[positions[rid]] = values
        train_status[positions[rid]] = status
    if extra is not None:
        for rid, values, status in zip(extra_ids, extra["values"], extra["status"]):
            train_values[positions[rid]] = values
            train_status[positions[rid]] = status
    if np.any((train_status == "ok") != np.isfinite(train_values)) or np.any(train_status == "pending"):
        raise RuntimeError("Extended TRAIN feature cache is incomplete or inconsistent")
    rates = np.mean(~np.isfinite(train_values), axis=0)
    keep = rates <= FAILURE_RATE_THRESHOLD
    base._emit(progress, "winner_TEST_features", seed=seed, total=1000)
    test = base._execute_features(bundle, test_ids, features,
                 safe(directory / "TEST_checkpoint"), progress, sandbox_factory)
    train_matrix = np.column_stack((train_magpie, train_values[:, keep]))
    test_matrix = np.column_stack((test_magpie, np.asarray(test["values"])[:, keep]))
    base._emit(progress, "winner_fullTRAIN_refit_predict", seed=seed)
    predictions, fit = base._predict(model_fitter, train_matrix, split["y"], test_matrix)
    train_path, test_path = safe(directory / "full_train_features.npz"), safe(directory / "test_features.npz")
    _write_npz(train_path, ids=np.asarray(split["ids"]), feature_names=np.asarray(names),
               values=train_values, status=train_status)
    _write_npz(test_path, ids=np.asarray(test_ids), feature_names=test["feature_names"],
               values=test["values"], status=test["status"])
    prediction_path = safe(directory / "predictions.npz")
    _write_npz(prediction_path, ids=np.asarray(test_ids), predictions=predictions)
    metadata = base._seal({"schema_version": "H4.3-winner-predictions-complete-v1", "status": "complete",
        "identity": identity, "identity_sha256": binding, "stage": "4.3", "condition": "L1", "seed": seed,
        "selected_round_index": cached_round["round_index"],
        "selected_validation_mae": cached_round["validation_mae"],
        "selected_round_record_sha256": cached_round["record_sha256"],
        "selected_round_train_cache": cached_round["artifacts"]["train_features"],
        "additional_train_structures_executed": len(extra_ids), "cached_train_structures_reexecuted": 0,
        "train_count": 2000, "test_count": 1000,
        "full_train_features": {"failure_rates": [float(rate) for rate in rates],
            "status_counts": dict(Counter(train_status.flat)), "threshold": .20, "drop_comparison": ">",
            "kept_names": [name for name, flag in zip(names, keep) if flag],
            "dropped_names": [name for name, flag in zip(names, keep) if not flag],
            "selection_domain": "final fullTRAIN2000 only; selected round fixed before fill",
            "fallback_to_magpie": not bool(keep.any())},
        "test_features": base._feature_metadata(test, np.arange(1000)),
        "TEST_failure_rates_used_for_selection": False, "fill_changes_saved_validation_or_winner": False,
        "additional_train_diagnostics": base._diagnostics(extra) if extra is not None else None,
        "TEST_diagnostics": base._diagnostics(test), "fit": fit,
        "elapsed_seconds": time.perf_counter() - started,
        "metadata_path": str(safe(directory / "final_predictions.json")),
        "artifacts": {"full_train_features": {"path": str(train_path), "sha256": base._file_hash(train_path)},
            "test_features": {"path": str(test_path), "sha256": base._file_hash(test_path)},
            "predictions": {"path": str(prediction_path), "sha256": base._file_hash(prediction_path)}},
        "TEST_target_values_read": False, "TEST_scoring_performed": False})
    _write_json(directory / "final_predictions.json", metadata)
    base._emit(progress, "winner_prediction_complete", seed=seed, completed=1000)
    return metadata


def _original_initial_path(seed):
    return ROOT / 'work/h-data/H4/phase4_2/initial' / ('L0_seed' + str(seed)) / 'initial_state.json'


def initial_state(bundle, seed, labels, output_dir, progress=None, *, model_fitter=None):
    """Reuse the identical original L0 Magpie-only S0500 feedback, never refit."""
    split = _split(bundle, seed, labels, initial=True)
    source = _original_initial_path(seed)
    old = base._read_sealed(source)
    base._check_artifacts(old)
    if (old.get('phase') != 'initial' or old.get('winner_eligible') is not False
            or old['features'] != [] or old['active_features'] != [] or old['labelled_count'] != 500
            or old['identity']['train_ids'] != split['ids']
            or old['identity']['fit_ids'] != split['fit_ids'] or old['identity']['val_ids'] != split['val_ids']
            or old['identity']['train_labels_sha256'] != base._matrix_hash(split['y'])
            or old['identity']['train_magpie132_sha256'] != base._matrix_hash(base._design(bundle, split['ids']))
            or old['identity']['fixed_template_assignment_sha256'] != split['fixed_template_assignment_sha256']
            or old['fit']['parameters'] != HGB_PARAMETERS or old['fit']['training_weights'] != 'unweighted'):
        raise RuntimeError('Original L0 initial feedback is not exactly the same S0/numeric policy')
    identity, _ = _identity(bundle, seed, split, [], 'initial', None)
    identity.update(reused_original_initial_path=str(source.resolve()),
                    reused_original_initial_sha256=base._file_hash(source),
                    reused_original_initial_record_sha256=old['record_sha256'])
    binding = base._json_hash(identity)
    directory = safe(output_dir)
    cached = _cached(directory, 'initial_state.json', binding)
    if cached is not None:
        return cached
    payload = {key: value for key, value in old.items() if key not in ('record_sha256', 'identity', 'identity_sha256')}
    payload.update(schema_version='H4.3-reused-initial-feedback-v1', stage='4.3', condition='L1',
        identity=identity, identity_sha256=binding, metadata_path=str(safe(directory / 'initial_state.json')),
        reused_initial_baseline=True, reused_original_identity=old['identity'],
        reused_original_initial_path=str(source.resolve()), reused_original_initial_record_sha256=old['record_sha256'],
        new_model_fit_calls=0, new_generated_feature_executions=0, elapsed_seconds=0.)
    result = base._seal(payload)
    _write_json(directory / 'initial_state.json', result)
    base._emit(progress, 'initial_validation_reused', seed=seed, labelled_count=500)
    return result


def _failure_records(metadata, status, values, fit):
    failures = []
    fit_set = set(int(i) for i in fit)
    for feature_index in range(values.shape[1]):
        failed = int((~np.isfinite(values[fit, feature_index])).sum())
        if not failed:
            continue
        detailed = Counter()
        for error in metadata.get('diagnostics', {}).get('errors', []):
            if error.get('feature') == feature_index and error.get('row') in fit_set:
                key = str(error.get('status', 'failure')) + '|' + str(error.get('error', ''))
                detailed[key] += 1
        reasons = detailed if sum(detailed.values()) == failed else Counter(
            str(item) for item in status[fit, feature_index] if item != 'ok')
        failures.append({'feature_index': feature_index, 'phase': 'inner_fit',
            'failed_count': failed, 'total_count': len(fit), 'reasons': dict(sorted(reasons.items()))})
    return failures


def analyze_lofo(bundle, metadata, labels, output_dir, progress=None, *, model_fitter=None):
    """TRAIN/validation-only leave-one-retained-function-out HGB refits.

    Features are read from the completed round's immutable cache. No generated
    function is executed, including for failures. Cached baseline predictions
    remain the baseline; no baseline refit and no TEST data are used.
    """
    if metadata.get('phase') != 'round' or metadata.get('status') != 'complete':
        raise ValueError('LOFO requires a completed round model')
    base._check_artifacts(metadata)
    ids = metadata['identity']['train_ids']
    if not set(ids) <= set(bundle['pool_ids']) or set(ids) & set(bundle['test_ids']):
        raise ValueError('LOFO accepts previously revealed POOL train/validation members only')
    if not set(ids) <= set(labels):
        raise ValueError('LOFO targets do not cover the saved round membership')
    y = np.asarray([labels[rid] for rid in ids], dtype=np.float64)
    if not np.isfinite(y).all() or base._matrix_hash(y) != metadata['identity']['train_labels_sha256']:
        raise ValueError('LOFO saved-round target identity changed')
    magpie = base._design(bundle, ids)
    if base._matrix_hash(magpie) != metadata['identity']['train_magpie132_sha256']:
        raise ValueError('LOFO Magpie rows changed')
    if metadata['fit']['parameters'] != HGB_PARAMETERS or metadata['fit']['training_weights'] != 'unweighted':
        raise ValueError('LOFO numeric training policy differs from the baseline')
    index = {rid: i for i, rid in enumerate(ids)}
    fit = np.asarray([index[rid] for rid in metadata['identity']['fit_ids']], dtype=int)
    val = np.asarray([index[rid] for rid in metadata['identity']['val_ids']], dtype=int)
    with np.load(metadata['artifacts']['train_features']['path'], allow_pickle=False) as cache:
        if cache['ids'].tolist() != ids or cache['feature_names'].tolist() != [f['name'] for f in metadata['features']]:
            raise ValueError('LOFO training feature cache identity changed')
        values, status = cache['values'].copy(), cache['status'].copy()
    with np.load(metadata['artifacts']['validation_predictions']['path'], allow_pickle=False) as cache:
        if cache['ids'].tolist() != metadata['identity']['val_ids'] or not np.array_equal(cache['truth'], y[val]):
            raise ValueError('LOFO validation targets/identities changed')
        baseline_prediction = cache['predictions'].copy()
    baseline_mae = float(np.mean(np.abs(baseline_prediction - y[val])))
    if baseline_mae != metadata['validation_mae']:
        raise ValueError('Cached baseline row MAE is not exactly the saved baseline')
    keep = np.mean(~np.isfinite(values[fit]), axis=0) <= FAILURE_RATE_THRESHOLD
    kept_indices = np.flatnonzero(keep).tolist()
    kept_names = [metadata['features'][i]['name'] for i in kept_indices]
    if kept_names != metadata['feature_selection']['kept_names']:
        raise ValueError('LOFO feature mask differs from the original inner-fit mask')
    matrix = np.column_stack((magpie, values[:, keep]))
    identity = {'protocol': 'H4.3-LOFO-cached-TRAIN-validation-only-v1',
        'source_round_record_sha256': metadata['record_sha256'],
        'source_round_artifacts': metadata['artifacts'], 'baseline_mae': baseline_mae,
        'train_labels_sha256': base._matrix_hash(y), 'code': _code_binding(),
        'parameters': HGB_PARAMETERS, 'kept_feature_indices': kept_indices,
        'delta_definition': 'row validation MAE_without_minus_saved_full; positive beneficial',
        'generated_feature_executions': 0, 'TEST_targets_or_feedback_used': False}
    binding = base._json_hash(identity)
    directory = safe(output_dir)
    cached = _cached(directory, 'lofo.json', binding)
    if cached is not None:
        return cached
    started = time.perf_counter()
    contributions = [{'feature_index': i, 'delta_without_minus_full': None} for i in range(len(metadata['features']))]
    artifacts, feature_records = {}, []
    for position, feature_index in enumerate(kept_indices):
        one_identity = {**identity, 'removed_raw_feature_index': feature_index}
        one_binding = base._json_hash(one_identity)
        filename = 'feature' + str(feature_index).zfill(2) + '.json'
        record = _cached(directory, filename, one_binding)
        if record is None:
            reduced = np.delete(matrix, 132 + position, axis=1)
            prediction, fitted = base._predict(model_fitter, reduced[fit], y[fit], reduced[val])
            without_mae = float(np.mean(np.abs(prediction - y[val])))
            prediction_path = safe(directory / ('feature' + str(feature_index).zfill(2) + '_validation.npz'))
            _write_npz(prediction_path, ids=np.asarray(metadata['identity']['val_ids']), predictions=prediction)
            record = base._seal({'status': 'complete', 'identity': one_identity,
                'identity_sha256': one_binding, 'feature_index': feature_index,
                'validation_mae_without': without_mae, 'baseline_validation_mae': baseline_mae,
                'delta_without_minus_full': without_mae - baseline_mae, 'fit': fitted,
                'artifacts': {'validation_predictions': {'path': str(prediction_path), 'sha256': base._file_hash(prediction_path)}},
                'generated_feature_executions': 0, 'TEST_targets_or_feedback_used': False})
            _write_json(directory / filename, record)
        contributions[feature_index]['delta_without_minus_full'] = record['delta_without_minus_full']
        feature_path = safe(directory / filename)
        artifacts['feature' + str(feature_index)] = {'path': str(feature_path), 'sha256': base._file_hash(feature_path)}
        for artifact_name, ref in record['artifacts'].items():
            artifacts['feature' + str(feature_index) + '_' + artifact_name] = ref
        feature_records.append({'feature_index': feature_index, 'metadata_path': str(feature_path),
                                'record_sha256': record['record_sha256']})
        base._emit(progress, 'lofo_feature_complete', seed=metadata['seed'], round_index=metadata['round_index'],
                   feature_index=feature_index, completed=position + 1, total=len(kept_indices))
    payload = base._seal({'schema_version': 'H4.3-LOFO-complete-v1', 'status': 'complete',
        'identity': identity, 'identity_sha256': binding, 'seed': metadata['seed'],
        'round_index': metadata['round_index'], 'baseline_validation_mae': baseline_mae,
        'raw_features': metadata['features'], 'contributions': contributions,
        'failures': _failure_records(metadata, status, values, fit),
        'retained_feature_count': len(kept_indices), 'LOFO_fit_count': len(kept_indices),
        'feature_records': feature_records, 'artifacts': artifacts,
        'metadata_path': str(safe(directory / 'lofo.json')), 'elapsed_seconds': time.perf_counter() - started,
        'generated_feature_executions': 0, 'TEST_targets_or_feedback_used': False})
    _write_json(directory / 'lofo.json', payload)
    return payload


def history_record(metadata, lofo, query_record, notes=''):
    if type(notes) is not str or len(notes) > 300:
        raise ValueError('L1 notes must be at most300 Unicode characters')
    if metadata['round_index'] != lofo['round_index'] or metadata['seed'] != lofo['seed']:
        raise ValueError('Memory history round/seed differs from its LOFO record')
    selection = query_record['query_selection']
    queries = query_record['query_payload']['queries']
    records = selection['queries']
    if len(queries) != len(records):
        raise ValueError('Query payload/result cardinality differs')
    query_rows, revealed = [], 0
    for query, result in zip(queries, records):
        revealed += result['selected_n']
        query_rows.append({'query': query, 'requested_count': result['requested_n'],
            'matched_count': result['matched_unlabeled_n'], 'revealed_count': result['selected_n'],
            'carry_after': selection['capacity'] - revealed})
    return {'round_index': metadata['round_index'], 'features': metadata['features'],
        'contributions': lofo['contributions'], 'failures': lofo['failures'], 'queries': query_rows,
        'query_summary': {'capacity': selection['capacity'], 'carry_in': selection['carry_in'],
            'carry_out': selection['carry_out'], 'requested_count': sum(row['requested_count'] for row in query_rows),
            'matched_count': sum(row['matched_count'] for row in query_rows), 'revealed_count': selection['selected_n']},
        'notes': notes}


def evaluate_round(bundle, seed, round_index, features, labels, output_dir, progress=None,
                   *, sandbox_factory=None, model_fitter=None, query_record=None, notes=''):
    metadata = _evaluate_round_baseline(bundle, seed, round_index, features, labels, output_dir,
                    progress, sandbox_factory=sandbox_factory, model_fitter=model_fitter)
    lofo = analyze_lofo(bundle, metadata, labels, safe(Path(output_dir) / 'lofo'), progress,
                         model_fitter=model_fitter)
    result = {**metadata, 'lofo': lofo}
    if query_record is not None:
        result['history_record'] = history_record(metadata, lofo, query_record, notes)
    return result


_PROJECTION_ARRAYS = None
_PROJECTION_CFG = None
_PROJECTION_LABELS = {}


def project_l0_round(metadata_path, output_dir):
    """Authorized step2 CPU projection using old L0 caches/POOL labels only."""
    global _PROJECTION_ARRAYS, _PROJECTION_CFG
    import json
    old_phase = ROOT / 'work/h-data/H4/phase4_2'
    if _PROJECTION_CFG is None:
        _PROJECTION_CFG = json.loads((old_phase / 'config.json').read_text(encoding='utf-8'))
    cfg = _PROJECTION_CFG
    if _PROJECTION_ARRAYS is None:
        representation = json.loads((ROOT / 'work/h-data/H3/representation_metadata.json').read_text(encoding='utf-8'))
        path = Path(representation['base_features_path'])
        if base._file_hash(path) != representation['file_sha256']['base_features_path']:
            raise ValueError('Projection Magpie cache hash changed')
        with np.load(path, allow_pickle=False) as cache:
            ids, values = cache['ids'].tolist(), cache['magpie132'].copy()
        _PROJECTION_ARRAYS = {'index': {rid: i for i, rid in enumerate(ids)}, 'magpie132': values,
            'pool_ids': cfg['data']['pool_ids'], 'test_ids': cfg['data']['test_ids']}
    metadata = base._read_sealed(metadata_path)
    seed, rnd = metadata['seed'], metadata['round_index']
    if metadata.get('condition') != 'L0' or metadata.get('stage') != '4.2':
        raise ValueError('Projection accepts only original L0 round metadata')
    if seed not in _PROJECTION_LABELS:
        state = json.loads((old_phase / 'states' / ('seed' + str(seed) + '.json')).read_text(encoding='utf-8'))
        if state['round_completed'] != 20 or len(state['labels']) != 2000 or not set(state['labels']) <= set(cfg['data']['pool_ids']):
            raise ValueError('Projection needs an unchanged completed L0 POOL-only label state')
        _PROJECTION_LABELS[seed] = state['labels']
    query_path = old_phase / 'query_records' / ('L0_seed' + str(seed) + '_round' + str(rnd).zfill(2) + '.json')
    query = json.loads(query_path.read_text(encoding='utf-8'))
    lofo = analyze_lofo(_PROJECTION_ARRAYS, metadata, _PROJECTION_LABELS[seed], safe(output_dir))
    history = history_record(metadata, lofo, query, '')
    result = {'status': 'complete', 'source_condition': 'L0', 'projection_only': True,
        'seed': seed, 'round_index': rnd, 'source_round_path': str(Path(metadata_path).resolve()),
        'source_round_sha256': base._file_hash(metadata_path), 'source_query_path': str(query_path.resolve()),
        'source_query_sha256': base._file_hash(query_path), 'lofo_metadata_path': lofo['metadata_path'],
        'lofo_record_sha256': lofo['record_sha256'], 'history': history,
        'generated_feature_executions': 0, 'TEST_targets_or_feedback_used': False, 'new_LLM_calls': 0}
    _write_json(safe(Path(output_dir) / 'history.json'), result)
    return result


def _projection_job(arguments):
    return project_l0_round(*arguments)


def project_all_l0(output_dir=None, workers=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    directory = safe(DATA / 'projection_data' if output_dir is None else output_dir)
    if workers not in (1, 2, 3, 4):
        raise ValueError('Projection CPU jobs must be1..4, each with one math thread')
    paths = sorted((ROOT / 'work/h-data/H4/phase4_2/rounds').glob('L0_seed*/round*/round.json'))
    if len(paths) != 60:
        raise ValueError('Projection requires all60 existing L0 round caches')
    jobs = []
    for path in paths:
        metadata = base._read_sealed(path)
        jobs.append((str(path), str(directory / ('seed' + str(metadata['seed'])) / ('round' + str(metadata['round_index']).zfill(2)))))
    records = []
    if workers == 1:
        for job in jobs:
            records.append(_projection_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(_projection_job, job) for job in jobs]
            for future in as_completed(pending):
                value = future.result()
                records.append(value)
                print('LOFO projection round complete seed=' + str(value['seed']) + ' round=' + str(value['round_index']) + ' total=' + str(len(records)), flush=True)
    records.sort(key=lambda item: (item['seed'], item['round_index']))
    result = {'status': 'complete', 'records': records, 'rounds': 60,
        'new_LLM_calls': 0, 'generated_feature_executions': 0, 'TEST_targets_or_feedback_used': False,
        'code': _code_binding(), 'math_threads_per_worker': 1, 'workers': workers}
    _write_json(directory / 'projection_index.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['project_l0'])
    parser.add_argument('--workers', type=int, default=4)
    arguments = parser.parse_args()
    project_all_l0(workers=arguments.workers)
