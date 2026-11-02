"""Frozen H4.1 feature execution and HGB prediction-only implementation.

Importing this module never runs generated code, fits a model, or reads targets.
Only labelled POOL targets supplied by the caller enter candidate selection.
TEST labels are deliberately absent from every interface in this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "work/h-data/H4"
HGB_PARAMETERS = dict(loss="squared_error", learning_rate=.05, max_iter=300,
                      max_leaf_nodes=15, min_samples_leaf=20,
                      l2_regularization=1., early_stopping=False,
                      random_state=20261018)
FAILURE_RATE_THRESHOLD = .20
SANDBOX_WORKERS = 2
SANDBOX_TIMEOUT_MS = 50
CHECKPOINT_STRUCTURES = 100
MODEL_PROTOCOL = "H4.1-inner-fit-failure-mask-row-MAE-fullTRAIN-refit-v1"


def safe(path):
    """Resolve links before every H4 output write, including temporary files."""
    target = Path(path).resolve()
    root = DATA.resolve()
    if "onedrive" in str(target).casefold() or not target.is_relative_to(root):
        raise ValueError("H4 model output must stay under non-OneDrive H4 data")
    return target


def _mkdir(path):
    target = safe(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_hash(matrix):
    array = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    target = safe(path)
    _mkdir(target.parent)
    temporary = safe(target.with_name(target.name + ".tmp"))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, safe(target))


def _write_npz(path, **arrays):
    target = safe(path)
    _mkdir(target.parent)
    temporary = safe(target.with_name(target.name + ".tmp"))
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, safe(target))


def _seal(payload):
    return {**payload, "record_sha256": _json_hash(payload)}


def _read_sealed(path):
    value = json.loads(safe(path).read_text(encoding="utf-8"))
    stored = value.pop("record_sha256", None)
    if stored != _json_hash(value):
        raise RuntimeError("H4 cached metadata self-hash mismatch")
    value["record_sha256"] = stored
    return value


def _emit(progress, phase, **details):
    if progress is not None:
        progress({"phase": phase, **details})


def _features(features):
    if not isinstance(features, list) or not 1 <= len(features) <= 20:
        raise ValueError("A candidate must contain one to twenty feature definitions")
    result = []
    seen = set()
    for feature in features:
        if not isinstance(feature, dict):
            raise ValueError("Feature definitions must be objects")
        name, source = feature.get("name"), feature.get("source")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Feature names must be nonempty and unique within a candidate")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Feature source must be a nonempty string")
        seen.add(name)
        result.append({"name": name, "source": source,
                       "description": str(feature.get("description", ""))})
    return result


def _ids(values, label):
    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError(f"{label} must be nonempty unique identities")
    return ids


def _design(bundle, ids):
    index = bundle["index"]
    rows = [index[rid] for rid in ids]
    matrix = np.asarray(bundle["magpie132"], dtype=np.float64)[rows]
    if matrix.shape != (len(ids), 132):
        raise ValueError("Frozen Magpie representation must have exactly132 columns")
    if np.isinf(matrix).any():
        raise ValueError("Magpie input contains infinity")
    return matrix


def _labelled(bundle, labels, split):
    train_ids = _ids(split["O_ids"], "O TRAIN")
    fit_ids = _ids(split["fit_ids"], "inner-fit TRAIN")
    val_ids = _ids(split["val_ids"], "inner validation")
    if len(train_ids) != 2000:
        raise ValueError("H4.1 final TRAIN must contain exactly2000 identities")
    train_set = set(train_ids)
    if set(fit_ids) & set(val_ids) or set(fit_ids) | set(val_ids) != train_set:
        raise ValueError("Frozen inner-fit and validation identities must partition TRAIN")
    if train_set & set(bundle["test_ids"]) or not train_set <= set(bundle["pool_ids"]):
        raise ValueError("Candidate selection is restricted to labelled POOL identities")
    if not isinstance(labels, dict) or set(labels) != train_set:
        raise ValueError("Only the exact labelled TRAIN target mapping may be supplied")
    y = np.asarray([labels[rid] for rid in train_ids], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Labelled TRAIN values must be finite")
    index = {rid: i for i, rid in enumerate(train_ids)}
    fit = np.array([index[rid] for rid in fit_ids], dtype=int)
    val = np.array([index[rid] for rid in val_ids], dtype=int)
    groups = bundle["groups"]
    fit_groups = {str(groups[rid]) for rid in fit_ids}
    val_groups = {str(groups[rid]) for rid in val_ids}
    if fit_groups & val_groups:
        raise ValueError("Inner template groups overlap between fit and validation")
    return train_ids, fit_ids, val_ids, y, fit, val


def _code_binding():
    return {"models_sha256": _file_hash(__file__),
            "sandbox_sha256": _file_hash(Path(__file__).with_name("h4_sandbox.py")),
            "original_ast_policy_sha256": _file_hash(ROOT / "h3_sandbox.py"),
            "frozen_h3_hgb_source_sha256": _file_hash(ROOT / "h3_models.py")}


def _execute_features(bundle, ids, features, directory, progress, sandbox_factory):
    names = [feature["name"] for feature in features]
    if not names:
        return {"ids": np.asarray(ids), "feature_names": np.asarray([], dtype="<U1"),
                "values": np.empty((len(ids), 0)),
                "status": np.empty((len(ids), 0), dtype="<U32"),
                "validation": [], "errors": [],
                "performance": {"n_structures": len(ids), "n_features": 0,
                                "n_workers": 0, "n_timeouts": 0,
                                "status_counts": {}, "wall_seconds": 0.,
                                "generated_code_in_parent": False}}
    if sandbox_factory is None:
        import h4_sandbox
        sandbox_factory = h4_sandbox.FeatureBatchRunner
    records = ({"id": rid, "s": bundle["structures_by_id"][rid]} for rid in ids)
    sources = {feature["name"]: feature["source"] for feature in features}
    with sandbox_factory(sources, n_workers=SANDBOX_WORKERS,
                         timeout_ms=SANDBOX_TIMEOUT_MS, memory_limit_mb=512) as runner:
        result = runner.evaluate_many(records, checkpoint_dir=safe(directory),
                                      checkpoint_every=CHECKPOINT_STRUCTURES,
                                      progress=progress)
    if result["ids"].tolist() != ids or result["feature_names"].tolist() != names:
        raise RuntimeError("Sandbox output identity or feature order mismatch")
    values = np.asarray(result["values"], dtype=np.float64)
    status = np.asarray(result["status"])
    if values.shape != (len(ids), len(names)) or status.shape != values.shape:
        raise RuntimeError("Sandbox output dimensions mismatch")
    if np.any((status == "ok") != np.isfinite(values)):
        raise RuntimeError("Sandbox finite/status contract mismatch")
    return result


def _feature_metadata(result, indices):
    values = np.asarray(result["values"], dtype=np.float64)[indices]
    status = np.asarray(result["status"])[indices]
    rates = np.mean(~np.isfinite(values), axis=0)
    return {"n_structures": len(indices),
            "failure_rates": [float(x) for x in rates],
            "status_counts": dict(Counter(status.flat)),
            "finite_cells": int(np.isfinite(values).sum()),
            "total_cells": int(values.size)}


def _diagnostics(result):
    return {"validation": result.get("validation", []),
            "errors": result.get("errors", []),
            "performance": result.get("performance", {})}


def _fit_hgb(x_train, y_train, x_predict):
    """Identical parameters, unweighted fit, and imputer policy to frozen H3."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    started = time.perf_counter()
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(x_train)
    prediction_input = imputer.transform(x_predict)
    if not np.isfinite(train).all() or not np.isfinite(prediction_input).all():
        raise ValueError("Nonfinite value survived TRAIN-only median imputation")
    estimator = HistGradientBoostingRegressor(**HGB_PARAMETERS)
    estimator.fit(train, y_train)
    predictions = np.asarray(estimator.predict(prediction_input), dtype=np.float64)
    return predictions, {"parameters": dict(HGB_PARAMETERS),
                         "fit_predict_seconds": time.perf_counter() - started,
                         "training_weights": "unweighted",
                         "imputation": "TRAIN-only median; keep_empty_features=True; all-empty columns fill0",
                         "all_missing_train_columns": np.flatnonzero(~np.isfinite(x_train).any(axis=0)).tolist()}


def _predict(model_fitter, train, y, predict):
    predictions, metadata = (model_fitter or _fit_hgb)(train, y, predict)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(predict),) or not np.isfinite(predictions).all():
        raise RuntimeError("HGB prediction shape/finite check failed")
    return predictions, metadata


def _group_mae(truth, predictions, groups):
    errors = np.abs(np.asarray(predictions) - np.asarray(truth))
    buckets = {}
    for error, group in zip(errors, groups):
        buckets.setdefault(str(group), []).append(float(error))
    return float(np.mean([np.mean(values) for values in buckets.values()]))


def _check_artifacts(metadata):
    for record in metadata["artifacts"].values():
        path = safe(record["path"])
        if not path.is_file() or _file_hash(path) != record["sha256"]:
            raise RuntimeError("Completed H4 artifact hash mismatch; refusing replacement")


def evaluate_candidate(bundle, seed, candidate_index, features, labels, split,
                       output_dir, progress=None, *, sandbox_factory=None,
                       model_fitter=None):
    """Run one unfinished candidate on TRAIN2000 only and seal validation MAE.

    Completed candidates are hash-checked and reused without generated-code
    execution or HGB fitting. A partially completed candidate restarts as a
    whole candidate; sandbox checkpoints are evidence, never row-resume input.
    Injectable factories are exclusively for CPU software mock verification.
    """
    features = _features(features)
    if not isinstance(candidate_index, int) or not 0 <= candidate_index < 20:
        raise ValueError("Candidate index must be0..19")
    train_ids, fit_ids, val_ids, y, fit, val = _labelled(bundle, labels, split)
    magpie = _design(bundle, train_ids)
    directory = safe(output_dir)
    identity = {"protocol": MODEL_PROTOCOL, "seed": int(seed),
                "candidate_index": candidate_index, "features": features,
                "train_ids": train_ids, "fit_ids": fit_ids, "val_ids": val_ids,
                "train_labels_sha256": _matrix_hash(y),
                "train_magpie132_sha256": _matrix_hash(magpie),
                "frozen_data_source_hashes": dict(bundle.get("source_hashes", {})),
                "parameters": HGB_PARAMETERS, "code": _code_binding(),
                "failure_rate_threshold": FAILURE_RATE_THRESHOLD,
                "sandbox_workers": SANDBOX_WORKERS, "deadline_ms": 50,
                "selection_metric": "unweighted inner validation row MAE",
                "test_target_values_read": False}
    binding = _json_hash(identity)
    complete_file = directory / "candidate.json"
    if complete_file.exists():
        cached = _read_sealed(complete_file)
        if cached.get("identity_sha256") != binding or cached.get("status") != "complete":
            raise RuntimeError("Completed candidate binding mismatch; refusing rerun")
        _check_artifacts(cached)
        return cached
    started = time.perf_counter()
    _mkdir(directory)
    _write_json(directory / "started.json", _seal({"identity": identity,
                "identity_sha256": binding, "whole_candidate_resume_only": True}))
    _emit(progress, "candidate_features", seed=int(seed), candidate_index=candidate_index)
    result = _execute_features(bundle, train_ids, features, directory / "sandbox_checkpoint",
                               progress, sandbox_factory)
    values = np.asarray(result["values"], dtype=np.float64)
    inner = _feature_metadata(result, fit)
    validation = _feature_metadata(result, val)
    full = _feature_metadata(result, np.arange(len(train_ids)))
    keep = np.asarray(inner["failure_rates"]) <= FAILURE_RATE_THRESHOLD
    matrix = np.column_stack((magpie, values[:, keep]))
    _emit(progress, "candidate_hgb_validation", seed=int(seed), candidate_index=candidate_index)
    predictions, fitted = _predict(model_fitter, matrix[fit], y[fit], matrix[val])
    validation_mae = float(np.mean(np.abs(predictions - y[val])))
    group_mae = _group_mae(y[val], predictions, [bundle["groups"][rid] for rid in val_ids])
    feature_path = directory / "train_features.npz"
    _write_npz(feature_path, ids=np.asarray(train_ids), feature_names=result["feature_names"],
               values=values, status=result["status"])
    validation_path = directory / "validation_predictions.npz"
    _write_npz(validation_path, ids=np.asarray(val_ids), predictions=predictions,
               truth=y[val])
    names = [feature["name"] for feature in features]
    payload = {"schema_version": "H4.1-candidate-complete-v1", "status": "complete",
               "identity": identity, "identity_sha256": binding,
               "seed": int(seed), "candidate_index": candidate_index,
               "features": features, "validation_mae": validation_mae,
               "validation_group_mae": group_mae,
               "inner_fit_features": inner, "validation_features": validation,
               "full_train_features": full,
               "feature_selection": {"domain": "inner-fit TRAIN only",
                 "threshold": FAILURE_RATE_THRESHOLD, "drop_comparison": ">",
                 "kept_names": [name for name, ok in zip(names, keep) if ok],
                 "dropped_names": [name for name, ok in zip(names, keep) if not ok],
                 "fallback_to_magpie": not bool(keep.any()),
                 "validation_failure_rates_used_for_selection": False,
                 "test_failure_rates_used_for_selection": False},
               "fit": fitted, "diagnostics": _diagnostics(result),
               "elapsed_seconds": time.perf_counter() - started,
               "metadata_path": str(safe(complete_file)),
               "artifacts": {"train_features": {"path": str(safe(feature_path)),
                                                 "sha256": _file_hash(feature_path)},
                             "validation_predictions": {"path": str(safe(validation_path)),
                                                 "sha256": _file_hash(validation_path)}},
               "test_structures_executed": 0, "test_target_values_read": False}
    cached = _seal(payload)
    _write_json(complete_file, cached)
    _emit(progress, "candidate_complete", seed=int(seed), candidate_index=candidate_index,
          validation_mae=validation_mae)
    return cached


def select_best(candidates):
    """Predeclared row-MAE selection; exact ties select earliest candidate."""
    candidates = list(candidates)
    if not candidates:
        raise ValueError("No complete candidate can be selected")
    seeds = {int(item["seed"]) for item in candidates}
    indices = [int(item["candidate_index"]) for item in candidates]
    if len(seeds) != 1 or len(indices) != len(set(indices)):
        raise ValueError("Candidate selection requires unique candidates of one seed")
    if any(item.get("status") != "complete" or not math.isfinite(item["validation_mae"])
           for item in candidates):
        raise ValueError("Every candidate must have complete finite validation MAE")
    return min(candidates, key=lambda item: (item["validation_mae"], item["candidate_index"]))


def refit_predict(bundle, seed, selected_candidate, labels, test_ids, output_dir,
                  progress=None, *, sandbox_factory=None, model_fitter=None):
    """FullTRAIN refit plus winner-only TEST1000 prediction, without TEST labels."""
    selected = _read_sealed(selected_candidate["metadata_path"])
    if selected["record_sha256"] != selected_candidate["record_sha256"]:
        raise RuntimeError("Selected candidate changed after selection")
    _check_artifacts(selected)
    if int(selected["seed"]) != int(seed):
        raise ValueError("Selected candidate seed differs from requested seed")
    if selected["identity"]["code"] != _code_binding():
        raise RuntimeError("Model or sandbox source changed after validation selection")
    if selected["identity"]["frozen_data_source_hashes"] != dict(bundle.get("source_hashes", {})):
        raise RuntimeError("Frozen source inputs changed after validation selection")
    split = selected["identity"]
    split = {"O_ids": split["train_ids"], "fit_ids": split["fit_ids"],
             "val_ids": split["val_ids"]}
    train_ids, _, _, y, _, _ = _labelled(bundle, labels, split)
    test_ids = _ids(test_ids, "TEST")
    if len(test_ids) != 1000 or test_ids != [str(rid) for rid in bundle["test_ids"]]:
        raise ValueError("Winner prediction requires the exact frozen ordered1000 TEST identities")
    magpie_train, magpie_test = _design(bundle, train_ids), _design(bundle, test_ids)
    if _matrix_hash(y) != selected["identity"]["train_labels_sha256"]:
        raise RuntimeError("TRAIN labels changed after validation")
    if _matrix_hash(magpie_train) != selected["identity"]["train_magpie132_sha256"]:
        raise RuntimeError("TRAIN Magpie matrix changed after validation")
    identity = {"protocol": MODEL_PROTOCOL, "seed": int(seed),
                "selected_candidate_record_sha256": selected["record_sha256"],
                "train_ids": train_ids, "test_ids": test_ids,
                "test_magpie132_sha256": _matrix_hash(magpie_test),
                "frozen_data_source_hashes": dict(bundle.get("source_hashes", {})),
                "parameters": HGB_PARAMETERS, "code": _code_binding(),
                "test_target_values_read": False}
    binding = _json_hash(identity)
    directory = safe(output_dir)
    complete_file = directory / "final_predictions.json"
    if complete_file.exists():
        cached = _read_sealed(complete_file)
        if cached.get("identity_sha256") != binding or cached.get("status") != "complete":
            raise RuntimeError("Completed final prediction binding mismatch; refusing rerun")
        _check_artifacts(cached)
        return cached
    started = time.perf_counter()
    _mkdir(directory)
    _write_json(directory / "started.json", _seal({"identity": identity,
                 "identity_sha256": binding, "prediction_only": True}))
    with np.load(selected["artifacts"]["train_features"]["path"], allow_pickle=False) as archive:
        if archive["ids"].tolist() != train_ids:
            raise RuntimeError("Selected TRAIN feature cache identity mismatch")
        train_values, train_status = archive["values"].copy(), archive["status"].copy()
        names = archive["feature_names"].tolist()
    rates = np.mean(~np.isfinite(train_values), axis=0)
    keep = rates <= FAILURE_RATE_THRESHOLD
    _emit(progress, "winner_test_features", seed=int(seed),
          candidate_index=selected["candidate_index"], total=1000)
    result = _execute_features(bundle, test_ids, selected["features"],
                               directory / "sandbox_checkpoint", progress, sandbox_factory)
    train_matrix = np.column_stack((magpie_train, train_values[:, keep]))
    test_matrix = np.column_stack((magpie_test, np.asarray(result["values"])[:, keep]))
    _emit(progress, "winner_fullTRAIN_refit_predict", seed=int(seed))
    predictions, fitted = _predict(model_fitter, train_matrix, y, test_matrix)
    feature_path = directory / "test_features.npz"
    _write_npz(feature_path, ids=np.asarray(test_ids), feature_names=result["feature_names"],
               values=result["values"], status=result["status"])
    prediction_path = directory / "predictions.npz"
    _write_npz(prediction_path, ids=np.asarray(test_ids), predictions=predictions)
    payload = {"schema_version": "H4.1-winner-predictions-complete-v1", "status": "complete",
               "identity": identity, "identity_sha256": binding,
               "seed": int(seed), "candidate_index": selected["candidate_index"],
               "selected_validation_mae": selected["validation_mae"],
               "selected_candidate_record_sha256": selected["record_sha256"],
               "train_count": 2000, "test_count": 1000,
               "full_train_features": {"failure_rates": [float(rate) for rate in rates],
                    "status_counts": dict(Counter(train_status.flat)),
                    "kept_names": [name for name, ok in zip(names, keep) if ok],
                    "dropped_names": [name for name, ok in zip(names, keep) if not ok],
                    "selection_domain": "fullTRAIN2000 only; recomputed after winner selection",
                    "threshold": FAILURE_RATE_THRESHOLD, "drop_comparison": ">",
                    "fallback_to_magpie": not bool(keep.any())},
               "test_features": _feature_metadata(result, np.arange(len(test_ids))),
               "test_failure_rates_used_for_selection": False,
               "diagnostics": _diagnostics(result), "fit": fitted,
               "elapsed_seconds": time.perf_counter() - started,
               "metadata_path": str(safe(complete_file)),
               "artifacts": {"test_features": {"path": str(safe(feature_path)),
                                                   "sha256": _file_hash(feature_path)},
                             "predictions": {"path": str(safe(prediction_path)),
                                                   "sha256": _file_hash(prediction_path)}},
               "test_target_values_read": False,
               "test_scoring_performed": False,
               "training_feature_cache_reexecuted": False}
    completed = _seal(payload)
    _write_json(complete_file, completed)
    _emit(progress, "winner_prediction_complete", seed=int(seed), completed=1000)
    return completed
