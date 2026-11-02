"""H4.2 dynamic-label validation and winner-only prediction, without TEST truth.

The frozen H4.1 modules remain untouched. Their numeric HGB/sandbox helpers are
reused under a separately bound H4.2 protocol and H4.2-only output boundary.
No generated code, model fitting or target access occurs on import.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import h4_models as base

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "work/h-data/H4/phase4_2"
HGB_PARAMETERS = dict(base.HGB_PARAMETERS)
FAILURE_RATE_THRESHOLD = .20
MODEL_PROTOCOL = "H4.2-dynamic-label-fixed-template-map-20rounds-v1"


def safe(path):
    target = Path(path).resolve()
    root = DATA.resolve()
    if "onedrive" in str(target).casefold() or not target.is_relative_to(root):
        raise ValueError("H4.2 model output must stay under non-OneDrive phase4_2")
    return target


def _write_json(path, value):
    base._write_json(safe(path), value)


def _write_npz(path, **arrays):
    base._write_npz(safe(path), **arrays)


def _code_binding():
    return {"models_sha256": base._file_hash(__file__),
            "frozen_H4_1_model_helper_sha256": base._file_hash(base.__file__),
            "sandbox_sha256": base._file_hash(Path(__file__).with_name("h4_sandbox.py")),
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
        raise ValueError("Only currently revealed POOL targets may enter H4.2 validation")
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
    identity = {"protocol": MODEL_PROTOCOL, "stage": "4.2", "condition": "L0",
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
        raise RuntimeError("Completed H4.2 result binding changed; refusing rerun")
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
    metadata = base._seal({"schema_version": "H4.2-model-validation-v1", "status": "complete",
        "identity": identity, "identity_sha256": binding, "stage": "4.2", "condition": "L0",
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


def initial_state(bundle, seed, labels, output_dir, progress=None, *, model_fitter=None):
    """Magpie-only S0 feedback, with no generated code and no LLM request.

    This baseline is strictly excluded from the twenty-round winner set.
    """
    split = _split(bundle, seed, labels, initial=True)
    return _validate(bundle, seed, split, [], output_dir, "initial", None,
                     progress, None, model_fitter)


def evaluate_round(bundle, seed, round_index, features, labels, output_dir, progress=None,
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
        raise ValueError("H4.2 selection requires exactly20 complete rounds1..20")
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
    identity = {"protocol": MODEL_PROTOCOL, "stage": "4.2", "condition": "L0", "seed": seed,
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
    metadata = base._seal({"schema_version": "H4.2-winner-predictions-complete-v1", "status": "complete",
        "identity": identity, "identity_sha256": binding, "stage": "4.2", "condition": "L0", "seed": seed,
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
