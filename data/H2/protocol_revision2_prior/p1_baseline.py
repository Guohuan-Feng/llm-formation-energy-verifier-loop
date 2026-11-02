"""Finite P1 composition-only baseline and paired composition-cluster metrics.

No experiment files, caches, models or logs are written by this module. The
Magpie table is fixed; remaining feature medians and the regressor are fitted
on TRAIN rows only. HOLDOUT labels are never used in either fit.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from importlib.metadata import version
import time

import numpy as np


RF_PARAMETERS = dict(
    n_estimators=300,
    max_features=1.0,
    min_samples_leaf=2,
    n_jobs=8,
    random_state=20261017,
)


def _composition(value):
    from pymatgen.core import Composition, Structure
    if isinstance(value, Composition):
        return value
    if isinstance(value, Structure):
        return value.composition
    if isinstance(value, str):
        return Composition(value)
    if isinstance(value, dict) and "sites" in value and "lattice" in value:
        return Structure.from_dict(value).composition
    if isinstance(value, dict):
        return Composition(value)
    if hasattr(value, "composition"):
        return value.composition
    raise TypeError(f"Unsupported composition/structure type: {type(value)!r}")


def _indices(values, n, name):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer index array")
    if result.size == 0 or np.any(result < 0) or np.any(result >= n):
        raise ValueError(f"{name} must select nonempty valid rows")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicates")
    return result.astype(int)


def run_magpie_rf(structures, targets, train_indices, holdout_indices, *,
                  n_jobs=8, logger=None):
    """Return held-out predictions and JSON-safe training/coverage metadata.

    ``structures`` can contain pymatgen structures, compositions, structure
    dictionaries, elemental-amount dictionaries or formula strings. The result
    has ``predictions`` and ``holdout_indices`` NumPy arrays and ``metadata``.
    It intentionally does not return a fitted model or write a model artifact.
    """
    from matminer.featurizers.composition import ElementProperty
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer

    started = time.perf_counter()
    y = np.asarray(targets, dtype=float)
    if y.ndim != 1 or y.size != len(structures) or not np.isfinite(y).all():
        raise ValueError("Targets must be finite and aligned with structures")
    tr = _indices(train_indices, y.size, "train_indices")
    ho = _indices(holdout_indices, y.size, "holdout_indices")
    if np.intersect1d(tr, ho).size:
        raise ValueError("TRAIN and HOLDOUT row indices overlap")
    log = logger if logger is not None else lambda message: None
    comp = [_composition(value) for value in structures]
    tr_groups = {value.reduced_formula for value in (comp[i] for i in tr)}
    ho_groups = {value.reduced_formula for value in (comp[i] for i in ho)}
    if tr_groups & ho_groups:
        raise ValueError("TRAIN and HOLDOUT reduced-composition groups overlap")

    # No data-dependent table imputation: raw nonfinite entries are retained for
    # coverage measurement, then a separate TRAIN-only median imputer is fit.
    featurizer = ElementProperty.from_preset("magpie", impute_nan=False)
    labels = list(featurizer.feature_labels())
    feature_started = time.perf_counter()
    features = np.empty((y.size, len(labels)), dtype=float)
    cached = {}
    for i, composition in enumerate(comp):
        key = tuple(sorted(composition.fractional_composition.get_el_amt_dict().items()))
        if key not in cached:
            cached[key] = np.asarray(featurizer.featurize(composition), dtype=float)
        features[i] = cached[key]
        if (i + 1) % 1000 == 0 or i + 1 == y.size:
            log(f"Magpie feature rows {i + 1}/{y.size}; unique compositions {len(cached)}")
    raw_finite = np.isfinite(features)
    features[~raw_finite] = np.nan
    feature_seconds = time.perf_counter() - feature_started
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_features = imputer.fit_transform(features[tr])
    holdout_features = imputer.transform(features[ho])
    if train_features.shape[1] != len(labels) or not np.isfinite(train_features).all():
        raise ValueError("TRAIN imputation did not yield all finite Magpie features")
    if not np.isfinite(holdout_features).all():
        raise ValueError("HOLDOUT imputation did not yield finite Magpie features")

    parameters = dict(RF_PARAMETERS, n_jobs=int(n_jobs))
    model = RandomForestRegressor(**parameters)
    fit_started = time.perf_counter()
    model.fit(train_features, y[tr])
    fit_seconds = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    prediction = model.predict(holdout_features)
    predict_seconds = time.perf_counter() - prediction_started
    if prediction.shape != ho.shape or not np.isfinite(prediction).all():
        raise ValueError("RF returned malformed/nonfinite held-out predictions")
    importance_order = np.argsort(-model.feature_importances_, kind="stable")[:15]

    def coverage(index):
        selected = raw_finite[index]
        return {
            "n_rows": int(len(index)),
            "all_finite_rows": int(selected.all(axis=1).sum()),
            "all_finite_row_fraction": float(selected.all(axis=1).mean()),
            "finite_feature_fraction": float(selected.mean()),
            "missing_cells": int((~selected).sum()),
        }

    metadata = {
        "model": "Magpie ElementProperty + RandomForestRegressor",
        "parameters": parameters,
        "training_weights": "unweighted rows; evaluation additionally uses original composition-group weights",
        "featurizer": {"preset": "magpie", "impute_nan": False,
                       "n_features": len(labels), "feature_labels": labels,
                       "reference": "https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.composition.html"},
        "feature_imputation": {"strategy": "TRAIN-only median",
                               "all_missing_feature_fill": 0,
                               "all_missing_train_features": [labels[i] for i in np.flatnonzero(~raw_finite[tr].any(axis=0))]},
        "coverage_before_imputation": {"train": coverage(tr), "holdout": coverage(ho)},
        "n_unique_featurized_compositions": len(cached),
        "n_train_groups": len(tr_groups), "n_holdout_groups": len(ho_groups),
        "composition_group_overlap": 0,
        "timing_seconds": {"feature_generation": feature_seconds, "fit": fit_seconds,
                           "predict": predict_seconds, "total": time.perf_counter() - started},
        "packages": {name: version(name) for name in ("matminer", "pymatgen", "scikit-learn", "numpy")},
        "top_feature_importance": [{"feature": labels[i], "importance": float(model.feature_importances_[i])}
                                   for i in importance_order],
    }
    log(f"Magpie RF complete: {len(tr)} TRAIN / {len(ho)} HOLDOUT; fit {fit_seconds:.1f}s")
    return {"predictions": prediction, "holdout_indices": ho, "metadata": metadata}


def _ratio(a, b, zero=np.nan):
    out = np.full(np.shape(a), zero, dtype=float)
    return np.divide(a, b, out=out, where=b > 0)


def _all_metrics(sums):
    # The final dimension is weight, weighted y, y^2, abs(error), error^2,
    # true-negative, false-positive, false-negative, true-positive.
    w, sy, sy2, ae, se, tn, fp, fn, tp = np.moveaxis(sums, -1, 0)
    mse = _ratio(se, w)
    sst = sy2 - _ratio(sy * sy, w)
    # R2 is undefined on constant true values (no force_finite substitution).
    return {
        "mae": _ratio(ae, w), "rmse": np.sqrt(mse),
        "r2": 1.0 - _ratio(se, sst),
        "balanced_accuracy": (_ratio(tp, tp + fn) + _ratio(tn, tn + fp)) / 2,
        "precision": _ratio(tp, tp + fp, zero=0.0),
        "recall": _ratio(tp, tp + fn),
        "accuracy": _ratio(tp + tn, tn + fp + fn + tp),
    }


def _interval(point, draws):
    draws = np.asarray(draws)
    valid = draws[np.isfinite(draws)]
    return {
        "point": float(point) if np.isfinite(point) else None,
        "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else [None, None],
        "n_valid_boot": int(valid.size), "n_undefined_boot": int(draws.size - valid.size),
    }


def evaluate_p1(y_true, predictions_by_arm: Mapping[str, object], groups,
                full_group_sizes=None, *, threshold=0.0, n_boot=2000,
                seed=20261017, mask=None):
    """Summarize regression and formation-sign classification with paired CIs.

    Positive is strictly ``e_form < threshold``; equality is negative. A
    negative formation energy is stability relative to elemental references,
    not stability against all competing phases (energy above hull is needed).
    Scores are energies in eV/atom, never probabilities. Original group weights
    are preserved for a masked subset, and no row is silently excluded.
    """
    y = np.asarray(y_true, dtype=float)
    group_strings = np.asarray([str(value) for value in groups])
    if y.ndim != 1 or group_strings.shape != y.shape or not np.isfinite(y).all():
        raise ValueError("True energies and groups must be finite/aligned 1D arrays")
    if not np.isfinite(threshold) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2000:
        raise ValueError("Threshold must be finite and n_boot >= 2000")
    if not predictions_by_arm:
        raise ValueError("At least one complete prediction arm is required")
    names, prediction = [], []
    for name, values in predictions_by_arm.items():
        values = np.asarray(values, dtype=float)
        if not isinstance(name, str) or values.shape != y.shape or not np.isfinite(values).all():
            raise ValueError(f"Malformed/nonfinite prediction arm {name!r}")
        names.append(name)
        prediction.append(values)
    selection = np.ones(y.size, dtype=bool) if mask is None else np.asarray(mask)
    if selection.shape != y.shape or selection.dtype != np.bool_ or not selection.any():
        raise ValueError("Mask must select at least one aligned row")
    original_counts = Counter(group_strings)
    if full_group_sizes is None:
        sizes = dict(original_counts)
    else:
        sizes = {}
        for key, value in full_group_sizes.items():
            if str(key) in sizes or isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError("Original group sizes must be unique positive integers")
            sizes[str(key)] = int(value)
    for key, count in original_counts.items():
        if sizes.get(key, 0) < count:
            raise ValueError(f"Missing/too-small original group size: {key}")
    yy, gg = y[selection], group_strings[selection]
    pp = np.asarray(prediction)[:, selection].T
    observed_groups, group_index = np.unique(gg, return_inverse=True)
    n_groups = observed_groups.size
    truth = yy < threshold
    predicted = pp < threshold
    conf_cell = 2 * truth[:, None].astype(int) + predicted.astype(int)
    abs_error, square_error = np.abs(pp - yy[:, None]), (pp - yy[:, None]) ** 2
    weight_modes = {"row_weighted": np.ones(yy.size),
                    "group_weighted": np.asarray([1.0 / sizes[g] for g in gg])}
    all_sums = np.zeros((n_groups, 2, len(names), 9), dtype=float)
    for mode_index, weights in enumerate(weight_modes.values()):
        for arm_index in range(len(names)):
            base = np.column_stack((weights, weights * yy, weights * yy ** 2,
                                    weights * abs_error[:, arm_index], weights * square_error[:, arm_index],
                                    *[weights * (conf_cell[:, arm_index] == k) for k in range(4)]))
            np.add.at(all_sums[:, mode_index, arm_index, :], group_index, base)
    point = _all_metrics(all_sums.sum(axis=0))
    rng = np.random.default_rng(seed)
    boot = {metric: np.empty((n_boot, 2, len(names))) for metric in point}
    probabilities = np.full(n_groups, 1.0 / n_groups)
    for start in range(0, n_boot, 128):
        stop = min(start + 128, n_boot)
        multiplicities = rng.multinomial(n_groups, probabilities, size=stop - start)
        sums = (multiplicities @ all_sums.reshape(n_groups, -1)).reshape(stop - start, 2, len(names), 9)
        for metric, value in _all_metrics(sums).items():
            boot[metric][start:stop] = value
    result = {
        "n": int(yy.size), "n_groups": int(n_groups),
        "n_positive": int(truth.sum()), "n_negative": int((~truth).sum()),
        "threshold_eV_per_atom": float(threshold),
        "positive_definition": "formation energy strictly below threshold; not hull stability",
        "n_boot": int(n_boot), "seed": int(seed),
        "bootstrap_unit": "reduced-composition group, paired shared multinomial draws",
        "interval": "percentile 95%; undefined draws excluded and counted",
        "precision_zero_division": 0,
        "group_weight_rule": "1 / original full composition-group row count; preserved in subsets",
    }
    for mode_index, mode in enumerate(weight_modes):
        metrics = {name: {metric: _interval(point[metric][mode_index, i], boot[metric][:, mode_index, i])
                          for metric in point} for i, name in enumerate(names)}
        paired = {}
        for i, arm_a in enumerate(names):
            for j in range(i + 1, len(names)):
                arm_b = names[j]
                paired[f"{arm_a} - {arm_b}"] = {
                    "arm_a": arm_a, "arm_b": arm_b,
                    **{metric: _interval(point[metric][mode_index, i] - point[metric][mode_index, j],
                                         boot[metric][:, mode_index, i] - boot[metric][:, mode_index, j])
                       for metric in point},
                }
        result[mode] = {"metrics": metrics, "paired_differences": paired}
    return result
