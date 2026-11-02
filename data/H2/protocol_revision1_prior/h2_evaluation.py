"""Frozen H2 Magpie/RF evaluation, split manifest, and concise report.

Public API: evaluate_h2(synthetic_records, real_records, config, log=None),
write_readme(metrics, config, path). Evaluation returns JSON-safe metrics,
prediction_rows (TEST/HOLDOUT only), and a label-free split_manifest. Records
are {'row': result_row, 'final_structure': dict | None,
     'initial_structure': original_dict}. No GPU/LLM/model artifacts or cache
writes occur here. Call once after all frozen H2 label attempts have finished.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import Counter
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from p1_baseline import RF_PARAMETERS, evaluate_p1


ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"TEST": 1000, "POOL": 5000, "TRAIN": 2000, "HOLDOUT": 2000}
RF_SEED = 20261017
BOOTSTRAP_SEED = 20261023
DFT_FIELDS = ("dft_energy_eV_per_atom", "dft_formation_energy_eV_per_atom",
              "dft_target", "target")
CELL_FORCE_FIELDS = ("cell_fmax_eV_A", "crystal_fmax_eV_A",
                     "filtered_cell_fmax_eV_A", "filtered_fmax_eV_A")


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _finite(value):
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return answer if math.isfinite(answer) else None


def _true(value):
    return value is True or str(value).strip().lower() in ("true", "1", "yes")


def _row(record):
    result = record.get("row")
    if not isinstance(result, dict):
        raise TypeError("Every evaluation record requires a result-row dictionary")
    if not str(result.get("id", "")):
        raise ValueError("Every result row requires a stable nonempty id")
    return result


def _id(record):
    return str(_row(record)["id"])


def _split(record):
    return str(_row(record).get("split", "")).upper()


def _structure(record):
    # All predictors use composition only; fallback retains failed real rows in
    # the full DFT baseline without inventing their missing CHGNet labels.
    for candidate in (record.get("final_structure"), record.get("initial_structure"),
                      record.get("structure"), _row(record).get("initial_structure"),
                      _row(record).get("structure")):
        if candidate is not None:
            return candidate
    raise ValueError(f"No structure available for {_id(record)}")


def _group(record, synthetic):
    if synthetic:
        value = _row(record).get("template_id")
        if value is None or not str(value):
            raise ValueError(f"Missing synthetic template_id: {_id(record)}")
        return str(value)
    return _composition_group(record)


def _composition_group(record):
    from p1_data import reduced_composition
    return reduced_composition(_structure(record))


def _force_fields(row):
    cell_key = next((key for key in CELL_FORCE_FIELDS if _finite(row.get(key)) is not None), None)
    return _finite(row.get("atomic_fmax_eV_A")), _finite(row.get(cell_key)) if cell_key else None, cell_key


def _label_status(record):
    row = _row(record)
    atomic, cell, cell_key = _force_fields(row)
    failures = []
    if not _true(row.get("converged")):
        failures.append("optimizer_not_converged")
    if atomic is None or atomic > .1 or atomic < 0:
        failures.append("atomic_force_missing_or_above_0.1")
    if cell is None or cell > .1 or cell < 0:
        failures.append("cell_force_missing_or_above_0.1")
    if _finite(row.get("formation_energy_eV_per_atom")) is None:
        failures.append("formation_energy_missing_or_nonfinite")
    if record.get("final_structure") is None:
        failures.append("final_structure_missing")
    if row.get("error"):
        failures.append("relaxation_error")
    if "accepted" in row and not _true(row["accepted"]):
        failures.append(str(row.get("rejection_reason") or "not_accepted"))
    return {"usable": not failures, "failure_reasons": list(dict.fromkeys(failures)),
            "cell_force_field": cell_key}


def _target(record, origin):
    row = _row(record)
    fields = ("formation_energy_eV_per_atom",) if origin == "CHGNet" else DFT_FIELDS
    for key in fields:
        value = _finite(row.get(key))
        if value is not None:
            return value
    raise ValueError(f"Missing finite {origin} target for {_id(record)}")


def _attempted(record):
    row = _row(record)
    duration = _finite(row.get("relax_seconds"))
    return bool((duration is not None and duration >= 0) or row.get("error")
                or row.get("converged") is not None or row.get("steps") is not None)


def _coverage(records):
    statuses = [_label_status(record) for record in records]
    return {"n_requested": len(records), "n_attempted": sum(_attempted(record) for record in records),
            "n_usable": sum(status["usable"] for status in statuses),
            "usable_fraction": sum(status["usable"] for status in statuses) / len(records) if records else None,
            "failure_reasons_overlapping": dict(Counter(reason for status in statuses
                                                         for reason in status["failure_reasons"])),
            "unusable_ids": [_id(record) for record, status in zip(records, statuses) if not status["usable"]],
            "cell_force_fields": dict(Counter(status["cell_force_field"] or "missing" for status in statuses)),
            "policy": "Both atomic and cell force <=0.1 eV/A; finite relaxed label and accepted final structure. No replacements or CHGNet label imputation."}


def _distribution(records, origin, synthetic):
    values = np.asarray([_target(record, origin) for record in records], dtype=float)
    groups = [_group(record, synthetic) for record in records]
    counts = Counter(groups)
    result = {"label_origin": origin, "energy_unit": "eV/atom", "n": len(records),
              "n_groups": len(counts), "group_definition": "template_id" if synthetic else "reduced_composition",
              "weight_rule": "1 / current distribution subset group size"}
    for mode, weights in (("row_weighted", np.ones(len(records))),
                          ("group_weighted", np.asarray([1.0 / counts[group] for group in groups]))):
        if not len(records):
            result[mode] = {"mean": None, "negative_fraction": None, "percentiles": {str(q): None for q in (0, 5, 25, 50, 75, 95, 100)}}
            continue
        quantiles = (0, 5, 25, 50, 75, 95, 100)
        if mode == "row_weighted":
            percentiles = np.percentile(values, quantiles).tolist()
        else:
            order = np.argsort(values, kind="stable")
            cumulative = np.cumsum(weights[order])
            percentiles = [float(values[order[min(int(np.searchsorted(cumulative, q / 100 * cumulative[-1], side="left")), len(values) - 1)]]) for q in quantiles]
        result[mode] = {"mean": float(np.average(values, weights=weights)),
                        "negative_fraction": float(np.average(values < 0, weights=weights)),
                        "percentiles": dict(zip(map(str, quantiles), percentiles))}
    return result


def _features(records, log):
    from matminer.featurizers.composition import ElementProperty
    from p1_baseline import _composition
    feature_maker = ElementProperty.from_preset("magpie", impute_nan=False)
    labels = list(feature_maker.feature_labels())
    if len(labels) != 132:
        raise ValueError(f"Frozen H2 baseline requires 132 Magpie features, found {len(labels)}")
    cache, output = {}, np.empty((len(records), len(labels)), dtype=float)
    for i, record in enumerate(records):
        comp = _composition(_structure(record))
        key = tuple(sorted(comp.fractional_composition.get_el_amt_dict().items()))
        if key not in cache:
            cache[key] = np.asarray(feature_maker.featurize(comp), dtype=float)
        output[i] = cache[key]
    log(f"H2 fixed Magpie features: {len(records)} rows, {len(cache)} unique compositions, 132 features")
    output[~np.isfinite(output)] = np.nan
    return output, {"preset": "magpie", "n_features": len(labels), "feature_labels": labels,
                    "impute_nan": False, "n_unique_compositions": len(cache)}


def _fit_predict(x_train, y_train, x_test, feature_labels, log):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    parameters = dict(RF_PARAMETERS)
    # Parameters, including eight CPU jobs and the RF seed, match frozen P1.
    if parameters != {"n_estimators": 300, "max_features": 1.0, "min_samples_leaf": 2,
                      "n_jobs": 8, "random_state": RF_SEED}:
        raise ValueError("P1 RF parameters no longer match frozen H2 baseline")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train = imputer.fit_transform(x_train)
    test = imputer.transform(x_test)
    if train.shape[1] != 132 or test.shape[1] != 132 or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("TRAIN-only median imputation failed")
    started = time.perf_counter()
    model = RandomForestRegressor(**parameters)
    model.fit(train, np.asarray(y_train, dtype=float))
    predictions = np.asarray(model.predict(test), dtype=float)
    if predictions.shape != (len(x_test),) or not np.isfinite(predictions).all():
        raise ValueError("Malformed RF predictions")
    log(f"H2 Magpie/RF fit: {len(x_train)} TRAIN -> {len(x_test)} evaluation rows")

    def coverage(features):
        finite = np.isfinite(features)
        return {"n_rows": len(features), "all_finite_rows": int(finite.all(axis=1).sum()),
                "all_finite_row_fraction": float(finite.all(axis=1).mean()),
                "finite_feature_fraction": float(finite.mean()), "missing_cells": int((~finite).sum())}

    return predictions, {"model": "Magpie ElementProperty + RandomForestRegressor", "parameters": parameters,
                         "n_train": len(x_train), "n_evaluation": len(x_test),
                         "training_weights": "unweighted rows", "target_unit": "eV/atom",
                         "feature_imputation": {"strategy": "TRAIN-only median", "all_missing_feature_fill": 0,
                                                "all_missing_train_features": [feature_labels[i] for i in np.flatnonzero(~np.isfinite(x_train).any(axis=0))]},
                         "coverage_before_imputation": {"train": coverage(x_train), "evaluation": coverage(x_test)},
                         "fit_and_predict_seconds": time.perf_counter() - started}


def _metrics(records, targets, predictions, synthetic, seed):
    groups = [_group(record, synthetic) for record in records]
    result = evaluate_p1(targets, predictions, groups, full_group_sizes=dict(Counter(groups)),
                         threshold=0.0, n_boot=2000, seed=seed)
    group_name = "template_id" if synthetic else "reduced_composition"
    result.update(group_definition=group_name,
                  bootstrap_unit=f"{group_name} group; paired shared multinomial draws",
                  group_weight_rule="1 / current evaluation subset group size; no silent row exclusion")
    return result


def _sample(records, count, seed):
    # Sorting makes membership independent of parallel worker completion order.
    ordered = sorted(records, key=_id)
    indices = np.random.default_rng(seed).choice(len(ordered), size=count, replace=False)
    return [ordered[int(i)] for i in indices]


def _manifest(synthetic_test, synthetic_pool, real_train, real_holdout, rf_sample, config):
    split_ids = {"synthetic_TEST": [_id(r) for r in synthetic_test],
                 "synthetic_POOL": [_id(r) for r in synthetic_pool],
                 "real_TRAIN": [_id(r) for r in real_train],
                 "real_HOLDOUT": [_id(r) for r in real_holdout]}
    result = {"schema_version": 1, "split_ids": split_ids,
              "S0": {str(seed): [_id(r) for r in _sample(synthetic_pool, 500, seed)] for seed in (0, 1, 2)},
              "S0_seed_method": "NumPy default_rng(seed).choice on ID-sorted frozen POOL without replacement; seeds0/1/2",
              "h2_rf_synthetic_training_ids": [_id(r) for r in rf_sample],
              "h2_rf_sample_seed": RF_SEED,
              "h2_rf_sample_method": "NumPy default_rng(20261017).choice on ID-sorted frozen POOL without replacement",
              "template_ids": {"TEST": sorted({_group(r, True) for r in synthetic_test}),
                               "POOL": sorted({_group(r, True) for r in synthetic_pool})},
              "label_access_roles": {"synthetic_TEST": "sealed final evaluation only; never prompt, feedback, feature selection, or training",
                                     "synthetic_POOL_default": "hidden; labels remain in separate private cache",
                                     "S0": "500 labels per future seed; only that seed's initial labeled set",
                                     "h2_rf_synthetic_training_ids": "2000 labels for fixed H2 RF only; does not reveal them to future arms",
                                     "real_TRAIN": "frozen2000 DFT labels; strict-converged CHGNet labels where available",
                                     "real_HOLDOUT": "sealed final evaluation only; never prompt, feedback, feature selection, or training"},
              "future_scope": "H3/H4 not run by H2; no per-POOL target values in this manifest",
              "frozen_input_provenance": {key: config[key] for key in
                  ("frozen_inputs_sha256", "known_compositions_sha256", "source_hashes", "input_paths", "frozen_paths") if key in config}}
    result["manifest_content_sha256"] = _json_hash(result)
    return result


def evaluate_h2(synthetic_records, real_records, config, log=None):
    """Evaluate each predeclared H2 arm once after the complete label run.

    Counts are fixed at1000 accepted TEST,5000 accepted POOL,2000 frozen TRAIN,
    and2000 frozen HOLDOUT. Failed real labels are reported and never replaced.
    A partial CHGNet comparison is explicitly marked unavailable at full scope;
    its conditional metrics and the complete DFT comparison remain available.
    The caller owns serialization and the once-only completion guard.
    """
    say = log if log is not None else lambda message: None
    started = time.perf_counter()
    synthetic_records, real_records = list(synthetic_records), list(real_records)
    all_ids = [_id(record) for record in synthetic_records + real_records]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("H2 record IDs must be unique across real and synthetic data")
    if any(_split(record) not in ("TEST", "POOL") for record in synthetic_records):
        raise ValueError("Synthetic records must have frozen TEST/POOL membership")
    if any(_split(record) not in ("TRAIN", "HOLDOUT") for record in real_records):
        raise ValueError("Real records must have frozen TRAIN/HOLDOUT membership")
    # Rejected generated candidates may accompany the formal selected set; they
    # are not members of either retained synthetic split and expose no labels.
    accepted = [record for record in synthetic_records if _true(_row(record).get("accepted"))]
    st = sorted([r for r in accepted if _split(r) == "TEST"], key=_id)
    sp = sorted([r for r in accepted if _split(r) == "POOL"], key=_id)
    rt = sorted([r for r in real_records if _split(r) == "TRAIN"], key=_id)
    rh = sorted([r for r in real_records if _split(r) == "HOLDOUT"], key=_id)
    observed = dict(zip(("TEST", "POOL", "TRAIN", "HOLDOUT"), map(len, (st, sp, rt, rh))))
    if observed != EXPECTED_COUNTS:
        raise ValueError(f"H2 requires complete frozen split sizes {EXPECTED_COUNTS}; observed {observed}")
    if not all(_attempted(record) for record in rt + rh):
        raise ValueError("Every frozen real row requires a completed relaxation attempt before H2 evaluation")
    if any(not _label_status(record)["usable"] for record in st + sp):
        raise ValueError("An accepted synthetic member lacks a strict-converged valid label")
    template_overlap = {_group(r, True) for r in st} & {_group(r, True) for r in sp}
    if template_overlap:
        raise ValueError("Synthetic TEST and POOL template IDs overlap")
    real_overlap = {_composition_group(r) for r in rt} & {_composition_group(r) for r in rh}
    if real_overlap:
        raise ValueError("Real TRAIN and HOLDOUT reduced compositions overlap")
    synth_comp_overlap = {_composition_group(r) for r in st} & {_composition_group(r) for r in sp}
    sample = _sample(sp, 2000, RF_SEED)
    manifest = _manifest(st, sp, rt, rh, sample, config)
    # All membership and RF sample choices are fixed independently of target
    # values; TEST target values are subsequently used only for final scoring.
    features, feature_metadata = _features(sample + st + rt + rh, say)
    sx_train, sx_test, rx_train, rx_test = np.split(features, [2000, 3000, 5000])
    labels = feature_metadata["feature_labels"]
    sy_train = [_target(r, "CHGNet") for r in sample]
    sy_test = [_target(r, "CHGNet") for r in st]
    synthetic_prediction, synthetic_model = _fit_predict(sx_train, sy_train, sx_test, labels, say)
    dft_train, dft_test = [_target(r, "DFT") for r in rt], [_target(r, "DFT") for r in rh]
    dft_prediction, dft_model = _fit_predict(rx_train, dft_train, rx_test, labels, say)
    valid_train = np.asarray([_label_status(r)["usable"] for r in rt], dtype=bool)
    valid_holdout = np.asarray([_label_status(r)["usable"] for r in rh], dtype=bool)
    valid_rt = [r for r, valid in zip(rt, valid_train) if valid]
    valid_rh = [r for r, valid in zip(rh, valid_holdout) if valid]
    bootstrap_seed = int(config.get("bootstrap_seed", BOOTSTRAP_SEED))
    coverage = {"real_TRAIN": _coverage(rt), "real_HOLDOUT": _coverage(rh)}
    full_chgnet = bool(valid_train.all() and valid_holdout.all())
    metrics = {"status": "complete" if full_chgnet else "partial_real_label_coverage",
               "scope": "H2 only; no H3/H4 dispatch; zero scientific LLM calls",
               "scientific_llm_calls": 0, "n_evaluation_runs_per_arm": 1,
               "counts": {"synthetic_TEST": len(st), "synthetic_POOL": len(sp),
                          "synthetic_RF_training": len(sample), "S0_per_seed": 500,
                          "real_TRAIN_frozen": len(rt), "real_HOLDOUT_frozen": len(rh)},
               "group_overlap": {"synthetic_template": 0, "synthetic_reduced_composition": len(synth_comp_overlap),
                                 "real_reduced_composition": 0},
               "real_label_coverage": coverage,
               "requested_full_2000_by_2000_CHGNet_comparison_available": full_chgnet,
               "missing_label_policy": "No silent replacements/exclusions. CHGNet metrics condition on disclosed valid subset; complete frozen2000/2000 DFT baseline uses raw-composition fallback on relaxation failures.",
               "feature_generation": feature_metadata,
               "packages": {name: version(name) for name in ("matminer", "pymatgen", "scikit-learn", "numpy")},
               "split_manifest_sha256": manifest["manifest_content_sha256"],
               "baselines": {"synthetic_CHGNet": {"model": synthetic_model,
                   "evaluation": _metrics(st, sy_test, {"magpie_rf_CHGNet": synthetic_prediction}, True, bootstrap_seed)},
                   "real_DFT_full": {"model": dft_model,
                   "evaluation": _metrics(rh, dft_test, {"magpie_rf_DFT": dft_prediction}, False, bootstrap_seed)}},
               "distributions": {"synthetic_TEST_CHGNet": _distribution(st, "CHGNet", True),
                   "synthetic_POOL_CHGNet": _distribution(sp, "CHGNet", True),
                   "real_TRAIN_DFT": _distribution(rt, "DFT", False), "real_HOLDOUT_DFT": _distribution(rh, "DFT", False),
                   "real_TRAIN_CHGNet_available": _distribution(valid_rt, "CHGNet", False),
                   "real_HOLDOUT_CHGNet_available": _distribution(valid_rh, "CHGNet", False)},
               "interpretation": ["CHGNet-derived formation energies are approximate labels, not DFT or convex-hull stability.",
                                  "Absent composition in Matbench does not prove absence from literature or LLM training.",
                                  "Composition-only RF is evaluated across template-disjoint synthetic splits; composition overlap is separately disclosed.",
                                  "No TEST/HOLDOUT labels used in feature generation, RF fitting, imputation, sampling, or tuning."],
               "stop_reason": "H2 complete; stop and wait for human continuation before H3." if full_chgnet else
                              "Full H2 CHGNet comparison unavailable because frozen real rows failed labeling; stop for review."}
    prediction_rows = [{"id": _id(record), "dataset": "synthetic", "split": "TEST",
                        "template_id": _group(record, True), "reduced_composition": _composition_group(record),
                        "chgnet_target_eV_per_atom": float(target), "magpie_rf_chgnet_prediction_eV_per_atom": float(prediction)}
                       for record, target, prediction in zip(st, sy_test, synthetic_prediction)]
    chgnet_prediction = None
    if valid_rt and valid_rh:
        chgnet_prediction, chgnet_model = _fit_predict(rx_train[valid_train],
            [_target(r, "CHGNet") for r in valid_rt], rx_test[valid_holdout], labels, say)
        metrics["baselines"]["real_CHGNet_available"] = {
            "scope": "full2000/2000" if full_chgnet else "conditional on disclosed successful-label subset",
            "model": chgnet_model,
            "evaluation": _metrics(valid_rh, [_target(r, "CHGNet") for r in valid_rh],
                                   {"magpie_rf_CHGNet": chgnet_prediction}, False, bootstrap_seed)}
    else:
        metrics["baselines"]["real_CHGNet_available"] = {"unavailable": "No usable real TRAIN or HOLDOUT CHGNet labels"}
    if valid_rh:
        reference = [_target(r, "DFT") for r in valid_rh]
        arms = {"CHGNet_relaxed_label": [_target(r, "CHGNet") for r in valid_rh],
                "magpie_rf_DFT_fulltrain": dft_prediction[valid_holdout]}
        if chgnet_prediction is not None:
            arms["magpie_rf_CHGNet_availabletrain"] = chgnet_prediction
        metrics["real_HOLDOUT_CHGNet_vs_DFT"] = {
            "scope": "same successfully labeled HOLDOUT rows; DFT as reference; full split coverage reported separately",
            "evaluation": _metrics(valid_rh, reference, arms, False, bootstrap_seed)}
    else:
        metrics["real_HOLDOUT_CHGNet_vs_DFT"] = {"unavailable": "No usable real HOLDOUT CHGNet labels"}
    chgnet_by_id = {} if chgnet_prediction is None else dict(zip((_id(r) for r in valid_rh), map(float, chgnet_prediction)))
    for record, target, prediction in zip(rh, dft_test, dft_prediction):
        status = _label_status(record)
        prediction_rows.append({"id": _id(record), "dataset": "real", "split": "HOLDOUT",
                                "reduced_composition": _composition_group(record),
                                "dft_target_eV_per_atom": float(target),
                                "magpie_rf_dft_prediction_eV_per_atom": float(prediction),
                                "chgnet_label_usable": status["usable"],
                                "chgnet_label_failure_reasons": status["failure_reasons"],
                                "chgnet_target_eV_per_atom": _target(record, "CHGNet") if status["usable"] else None,
                                "magpie_rf_chgnet_prediction_eV_per_atom": chgnet_by_id.get(_id(record))})
    metrics["elapsed_evaluation_seconds"] = time.perf_counter() - started
    return {"metrics": metrics, "prediction_rows": prediction_rows, "split_manifest": manifest}


def write_readme(metrics, config, path):
    """Write at most20 lines; resolve links/junctions and forbid OneDrive."""
    destination = Path(path).resolve()
    if "onedrive" in str(destination).casefold() or not destination.is_relative_to(ROOT):
        raise ValueError(f"Forbidden H2 report destination: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)

    def display(value):
        return "不可用" if value is None else f"{float(value):.4f}"

    def point(baseline, metric, mode="row_weighted"):
        result = metrics["baselines"].get(baseline, {}).get("evaluation", {})
        arms = result.get(mode, {}).get("metrics", {})
        return next(iter(arms.values()), {}).get(metric, {}).get("point")

    coverage = metrics["real_label_coverage"]
    lines = ["# H2 正式数据集（止于本阶段）",
             "合成 TEST 1000 / POOL 5000，模板互不重叠；S0 各500，种子0/1/2；固定随机2000标签用于 H2 RF。",
             f"真实固定 TRAIN/HOLDOUT 各2000；可用 CHGNet 标签 {coverage['real_TRAIN']['n_usable']}/{coverage['real_HOLDOUT']['n_usable']}；失败不替换。",
             "|固定 Magpie132 + RF300，min_leaf=2|MAE（不加权）|MAE（组加权）|RMSE|R²|符号 BA|",
             "|---|---:|---:|---:|---:|---:|"]
    for key, title in (("synthetic_CHGNet", "合成 CHGNet"), ("real_CHGNet_available", "真实 CHGNet（可用子集）"), ("real_DFT_full", "真实 DFT（完整）")):
        lines.append(f"|{title}|{display(point(key, 'mae'))}|{display(point(key, 'mae', 'group_weighted'))}|{display(point(key, 'rmse'))}|{display(point(key, 'r2'))}|{display(point(key, 'balanced_accuracy'))}|")
    match = metrics.get("real_HOLDOUT_CHGNet_vs_DFT", {}).get("evaluation", {})
    match_mae = match.get("row_weighted", {}).get("metrics", {}).get("CHGNet_relaxed_label", {}).get("mae", {})
    lines.extend([f"真实 HOLDOUT 的 CHGNet 标签对 DFT：MAE={display(match_mae.get('point'))} eV/atom；n={match.get('n', 0)}/2000。",
                  "完整 MAE/RMSE/R²/BA、95% CI 及同一参考目标上的配对差值见 metrics.json。",
                  "合成按模板、真实按约化成分分组；权重=1/当前评测子集组大小；整组 bootstrap 2000次。",
                  "标签来源为 CHGNet checkpoint0.3.0 弛豫；原子最大力与晶胞力均≤0.1 eV/Å；DFT另存。",
                  "POOL 逐条标签存于独立缓存；公开划分清单仅有 ID 和访问角色；predictions.csv 仅含 TEST/HOLDOUT。",
                  "Matbench 未收录不等于文献首见或 LLM 未见；形成能为负不等于凸包稳定或可合成。",
                  f"状态：{metrics['status']}；科学 LLM 调用0。",
                  metrics["stop_reason"]])
    if metrics.get('benchmark'):
        summary='；'.join(f"{s['workers']} workers={s['throughput_per_hour']:.1f}个/h、{s['peak_device_memory_used_MiB']}MiB、最大ΔE={display(s['energy_max_abs_difference_eV_per_atom'])}" for s in metrics['benchmark'])
        lines.insert(-2,'GPU基准（同40个候选，含启动耗时）：'+summary+'。')
        lines.insert(-2,f"最大并行数{metrics['selected_workers']}；大图按显存准入降至2/1个worker。总活跃耗时{metrics['total_active_seconds']/3600:.2f}/24 h。")
    if metrics.get('resource_revision'):
        lines.insert(-2,'原硬件基准NHe3提交25.3GB显存，资源失败留档；替换该基准候选后完整重算40×3，正式数据生成与物理过滤规则未改。')
    if len(lines) > 20:
        raise AssertionError("H2 README exceeds20 lines")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")

