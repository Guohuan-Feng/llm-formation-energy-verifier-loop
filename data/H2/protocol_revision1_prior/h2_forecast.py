"""Read-only, CPU-only conservative remaining-work forecast for finite H2.

The estimate conditions on the frozen calibration design, which includes the
size extremes in every stratum. Its bootstrap interval is a planning range,
not an unbiased confidence interval for an IID random sample of4000 records.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261023
CONTINGENCY_FRACTION = .05
EVALUATION_RESERVE_SECONDS = 1800.
SIZE_BINS = ((1, 20, "1-20"), (21, 40, "21-40"), (41, 80, "41-80"),
             (81, 160, "81-160"), (161, None, "161-inf"))


def _row(value):
    return value.get("row", value)


def _positive(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError):
        return None


def _planning_seconds(row):
    """Real work at selected scheduler capacity, respecting GPU admission."""
    normalized = _positive(row.get("planning_worker_seconds"))
    return normalized if normalized is not None else _positive(row.get("relax_seconds"))


def _size_bin(n):
    for low, high, label in SIZE_BINS:
        if n >= low and (high is None or n <= high):
            return label
    raise ValueError(f"Invalid real site count {n}")


def _wilson(k, n, z=1.959963984540054):
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("Invalid benchmark success count")
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0., center - radius), min(1., center + radius)


def _summaries(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        if "benchmark" in value:
            return value["benchmark"]
        return list(value.values())
    raise TypeError("benchmark_summaries must be a list or mapping")


def _benchmark_rows(value, workers, summary):
    if value is None:
        value = summary.get("results")
    elif isinstance(value, dict):
        value = value.get(str(workers), value.get(workers))
    if value is None:
        return []
    return [_row(r) for r in value]


def _synthetic_admission_work(candidates, prefix_n, selected_workers):
    """Convert only the requested frozen prefix to selected-worker work units.

    Measured benchmark seconds/candidate already includes division by selected
    concurrency. One candidate admitted at a lower concurrency therefore costs
    selected/actual such units; the caller must not divide by workers again.
    """
    from h2_admission import group_candidates

    prefix = candidates[:prefix_n]
    groups = group_candidates(prefix, selected_workers)
    distribution = {str(workers): len(rows) for workers, rows in groups}
    units = math.fsum(len(rows) * selected_workers / workers
                      for workers, rows in groups)
    n = len(prefix)
    return {
        "requested_prefix_n": prefix_n, "available_prefix_n": n,
        "prefix_last_id": prefix[-1]["id"] if prefix else None,
        "admission_workers_distribution": distribution,
        "effective_work_units": units,
        "extra_work_units": units - n,
        "work_multiplier": units / n if n else 1.,
    }


def build_forecast(plan, info, calibration_results, benchmark_summaries,
                   selected_workers, remaining_seconds,
                   calibration_wall_seconds=None, benchmark_results=None):
    """Return a JSON-safe forecast; never read/write files or initialize a GPU.

    Optional benchmark_results is either the selected worker's result list or
    a mapping1/2/4->result lists. Supplying it enables the2000-replicate runtime
    bootstrap; otherwise measured benchmark throughput gets a10% fallback
    slowdown before the common5% contingency. The30-minute evaluation reserve
    is always included. Already completed calibration work is excluded.
    Real calibration timings prefer positive planning_worker_seconds, supplied
    as relax_seconds*selected_max_workers/actual_admitted_workers by the runner;
    otherwise raw relax_seconds is used. Synthetic benchmark timing stays raw,
    while each required formal synthetic prefix is weighted selected/actual
    admission workers before its cost is multiplied by benchmark seconds/item.
    """
    workers = int(selected_workers)
    remaining = float(remaining_seconds)
    if workers not in (1, 2, 4) or not math.isfinite(remaining):
        raise ValueError("Invalid selected worker count or remaining budget")
    summaries = _summaries(benchmark_summaries)
    summary_by_workers = {int(s["workers"]): s for s in summaries}
    selected = summary_by_workers.get(workers)
    serial = summary_by_workers.get(1)
    result = {
        "method": "Frozen size/split-stratified calibration with admission-normalized real planning_worker_seconds; conditional bootstrap planning upper bound; serial strict-physics Wilson yield lower bound",
        "selected_workers": workers, "remaining_seconds": remaining,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_interval": [2.5, 97.5],
        "contingency_fraction": CONTINGENCY_FRACTION,
        "evaluation_reserve_seconds": EVALUATION_RESERVE_SECONDS,
        "throughput_table": [{k: s.get(k) for k in (
            "workers", "n", "throughput_per_hour", "compute_throughput_per_hour",
            "wall_seconds_including_startup_shutdown", "peak_device_memory_used_MiB",
            "energy_max_abs_difference_eV_per_atom", "matches_serial",
            "n_accepted_physics", "n_errors")} for s in summaries],
        "warnings": [], "blocking_reasons": [],
        "remaining_real_seconds_upper": None,
        "remaining_synthetic_seconds_upper": None,
        "remaining_total_seconds_upper": None,
        "remaining_total_seconds_point": None,
        "per_split_candidates_needed_upper": None,
        "calibration_design_note": "Each split x site-count stratum contains its smallest and largest structure plus4 independent-hash selected others. The6-point mean deliberately includes size extremes. Bootstrap conditions on this design and is not an unbiased random-sample CI. Population weights use the frozen4000 real records, never1/10 per stratum.",
        "real_timing_policy": "Use positive planning_worker_seconds when present, otherwise positive raw relax_seconds, in every real timing calculation. Runner normalization is raw_relax_seconds*selected_max_workers/actual_admitted_workers, so dividing planning work by selected_max_workers preserves the wall cost of memory-limited1/2-worker admission. Synthetic benchmark throughput/bootstraps always use raw relax_seconds.",
    }
    block = result["blocking_reasons"]
    warnings = result["warnings"]
    if selected is None or serial is None:
        block.append("Missing selected-worker or serial benchmark summary")
    elif selected.get("matches_serial") is False:
        block.append("Selected worker setting did not pass the frozen serial-equivalence criterion")
    real = plan["real_candidates"]
    if len(real) != 4000 or Counter(r["split"] for r in real) != Counter({"TRAIN": 2000, "HOLDOUT": 2000}):
        block.append("Frozen real population is not TRAIN2000/HOLDOUT2000")
    population_by_id = {r["id"]: r for r in real}
    calibration_ids = set(plan["real_calibration_ids"])
    rows = [_row(r) for r in calibration_results]
    if len({r["id"] for r in rows}) != len(rows):
        block.append("Duplicate calibration result IDs")
    if {r["id"] for r in rows} != calibration_ids:
        block.append("Calibration results do not cover the exact frozen calibration IDs")
    by_id = {r["id"]: r for r in rows}
    for row in rows:
        source = population_by_id.get(row["id"])
        if source is None:
            block.append(f"Unknown calibration ID {row['id']}")
            continue
        if row.get("split", source["split"]) != source["split"] or row.get("n_sites", source["n_sites"]) != source["n_sites"]:
            block.append(f"Calibration split/site metadata differs from frozen plan: {row['id']}")
        if row.get("error"):
            block.append(f"Calibration worker error prevents a reliable completion estimate: {row['id']}: {row['error']}")
    strata = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    real_bootstrap_work = np.zeros(BOOTSTRAP_REPLICATES, dtype=float)
    mean_work = 0.
    interior_sensitivity_work = 0.
    all_times = []
    configured_strata = {(s["split"], s["size_bin"]): s
                         for s in info.get("real_calibration_strata", [])}
    for split in ("TRAIN", "HOLDOUT"):
        for _, _, label in SIZE_BINS:
            population = [r for r in real if r["split"] == split and _size_bin(r["n_sites"]) == label]
            if not population:
                continue
            expected = [r for r in population if r["id"] in calibration_ids]
            valid = [by_id[r["id"]] for r in expected if r["id"] in by_id and
                     not by_id[r["id"]].get("error") and
                     _planning_seconds(by_id[r["id"]]) is not None]
            cfg_stratum = configured_strata.get((split, label))
            if cfg_stratum and cfg_stratum["population"] != len(population):
                block.append(f"Frozen calibration population count mismatch for {split}/{label}")
            if not valid:
                block.append(f"No valid runtime measurements for {split}/{label}")
                continue
            if len(valid) != len(expected):
                block.append(f"Incomplete valid runtime measurements for {split}/{label}")
            times = np.asarray([_planning_seconds(r) for r in valid], dtype=float)
            raw_times = [_positive(r.get("relax_seconds")) for r in valid]
            raw_times = [t for t in raw_times if t is not None]
            all_times.extend(times.tolist())
            n_remaining = len(population) - len(expected)
            sample_means = rng.choice(times, size=(BOOTSTRAP_REPLICATES, len(times)), replace=True).mean(axis=1)
            real_bootstrap_work += n_remaining * sample_means
            mean_work += n_remaining * float(times.mean())
            # Endpoints are already processed. Show sensitivity to the4 hash
            # selected interior times; use all6 for the conservative decision.
            ordered = sorted(population, key=lambda r: (r["n_sites"], r["id"]))
            extremes = {ordered[0]["id"], ordered[-1]["id"]}
            interior = [_planning_seconds(r) for r in valid if r["id"] not in extremes]
            interior_mean = float(np.mean(interior)) if interior else float(times.mean())
            interior_sensitivity_work += n_remaining * interior_mean
            strata.append({
                "split": split, "size_bin": label, "population": len(population),
                "population_weight": len(population) / len(real),
                "calibration_n": len(expected), "valid_timing_n": len(valid),
                "remaining_n": n_remaining,
                "population_sites_min": ordered[0]["n_sites"],
                "population_sites_max": ordered[-1]["n_sites"],
                "calibration_sites": sorted(population_by_id[r["id"]]["n_sites"] for r in valid),
                "mean_worker_wall_seconds": float(times.mean()),
                "mean_planning_worker_seconds": float(times.mean()),
                "mean_raw_relax_seconds": float(np.mean(raw_times)) if raw_times else None,
                "planning_normalization_applied_n": sum(_positive(r.get("planning_worker_seconds")) is not None for r in valid),
                "calibration_effective_planning_worker_seconds": times.tolist(),
                "interior_mean_worker_wall_seconds_sensitivity": interior_mean,
                "mean_remaining_worker_work_seconds": n_remaining * float(times.mean()),
                "bootstrap_mean_worker_wall_seconds_CI95": [float(x) for x in np.percentile(sample_means, [2.5, 97.5])],
            })
    result["real_strata"] = strata
    result["strata"] = strata
    result["real_population_covered"] = sum(s["population"] for s in strata)
    result["real_calibration_count"] = len(rows)
    result["real_remaining_count"] = len(real) - len(calibration_ids)
    if result["real_population_covered"] != len(real):
        block.append("Real forecast has a missing population stratum")
    ideal_calibration_wall = math.fsum(all_times) / workers if all_times else 0.
    measured_calibration_wall = _positive(calibration_wall_seconds)
    wall_source = "full calibration batch including startup/shutdown"
    if measured_calibration_wall is None:
        starts = [r.get("task_started_perf_counter") for r in rows]
        stops = [r.get("task_finished_perf_counter") for r in rows]
        if starts and all(isinstance(x, (int, float)) and math.isfinite(x) for x in starts + stops):
            measured_calibration_wall = _positive(max(stops) - min(starts))
            wall_source = "task compute span; startup/shutdown unavailable"
        if measured_calibration_wall is None:
            wall_source = "unavailable; conservative1.10 fallback"
        warnings.append("Pass calibration_wall_seconds for a measured startup/makespan correction; fallback excludes some batch overhead.")
    makespan_factor = max(1., measured_calibration_wall / ideal_calibration_wall) if measured_calibration_wall and ideal_calibration_wall > 0 else 1.10
    result["real_makespan"] = {
        "measured_calibration_wall_seconds": measured_calibration_wall,
        "ideal_sum_worker_times_divided_by_workers_seconds": ideal_calibration_wall,
        "factor": makespan_factor, "source": wall_source,
        "timing_basis": "effective planning_worker_seconds; raw relax_seconds fallback",
    }
    real_mean = mean_work / workers * makespan_factor
    real_bootstrap_seconds = real_bootstrap_work / workers * makespan_factor
    real_ci = [float(x) for x in np.percentile(real_bootstrap_seconds, [2.5, 97.5])]
    result["real_remaining_mean_seconds"] = real_mean
    result["real_remaining_CI95_seconds"] = real_ci
    result["real_interior_only_sensitivity_seconds"] = interior_sensitivity_work / workers * makespan_factor
    synth_mean = synth_upper = None
    if selected is not None and serial is not None:
        n_serial = int(serial.get("n", 0))
        accepted = int(serial.get("n_accepted_physics", 0))
        throughput = _positive(selected.get("throughput_per_hour"))
        if n_serial != len(plan["benchmark_candidates"]) or n_serial != 40:
            block.append("Serial yield benchmark does not contain the frozen40 candidates")
        if throughput is None:
            block.append("Selected benchmark has no positive measured throughput")
        if n_serial > 0 and 0 <= accepted <= n_serial:
            low, high = _wilson(accepted, n_serial)
            result["synthetic_physics_yield"] = {"accepted": accepted, "n": n_serial,
                                                 "point": accepted / n_serial,
                                                 "Wilson_CI95": [low, high]}
            if low <= 0 or accepted <= 0:
                block.append("Strict-physics benchmark has no positive conservative yield")
            elif throughput:
                targets = info.get("target_retained_sizes", {"TEST": 1000, "POOL": 5000})
                counts = {}
                for split in ("TEST", "POOL"):
                    target = int(targets[split])
                    reserve = len(plan["synthetic_candidates"][split])
                    point_raw = math.ceil(target / (accepted / n_serial))
                    upper_raw = math.ceil(target / low)
                    counts[split] = {"target_retained": target, "available_reserve": reserve,
                                     "raw_needed_point_yield": point_raw,
                                     "raw_needed_Wilson_lower_yield": upper_raw,
                                     "remaining_compute_cap": min(upper_raw, reserve),
                                     "reserve_sufficient_for_conservative_yield": upper_raw <= reserve}
                    if upper_raw > reserve:
                        warnings.append(f"{split} conservative-yield request{upper_raw} exceeds frozen reserve{reserve}; compute forecast uses reserve ceiling, so target completion is not guaranteed by this bound.")
                result["synthetic_candidate_counts"] = counts
                result["per_split_candidates_needed_upper"] = {s: v["raw_needed_Wilson_lower_yield"] for s, v in counts.items()}
                result["reserve_sufficient_for_conservative_yield"] = all(v["reserve_sufficient_for_conservative_yield"] for v in counts.values())
                candidate_mean = 3600. / throughput
                b_rows = _benchmark_rows(benchmark_results, workers, selected)
                b_times = [_positive(r.get("relax_seconds")) for r in b_rows]
                if len(b_times) == int(selected["n"]) and all(t is not None for t in b_times):
                    times = np.asarray(b_times, dtype=float)
                    wall = float(selected["wall_seconds_including_startup_shutdown"])
                    factor = max(1., wall / (float(times.sum()) / workers))
                    candidate_boot = rng.choice(times, size=(BOOTSTRAP_REPLICATES, len(times)), replace=True).mean(axis=1) / workers * factor
                    candidate_upper = max(candidate_mean, float(np.percentile(candidate_boot, 97.5)))
                    synthetic_runtime_method = "2000-replicate selected-worker per-candidate runtime bootstrap, multiplied by measured batch/ideal-work makespan factor"
                else:
                    candidate_upper = candidate_mean * 1.10
                    synthetic_runtime_method = "measured end-to-end throughput, slowed10% because raw benchmark timings were not supplied"
                    warnings.append("Pass benchmark_results to bootstrap selected-worker runtime uncertainty; using10% slowdown fallback.")
                admission = {}
                for split, count in counts.items():
                    candidates = plan["synthetic_candidates"][split]
                    point_admission = _synthetic_admission_work(candidates, count["raw_needed_point_yield"], workers)
                    upper_admission = _synthetic_admission_work(candidates, count["remaining_compute_cap"], workers)
                    count["point_prefix_admission"] = point_admission
                    count["upper_prefix_admission"] = upper_admission
                    admission[split] = {"point": point_admission, "upper": upper_admission}
                point_total = sum(v["point"]["available_prefix_n"] for v in admission.values())
                upper_total = sum(v["upper"]["available_prefix_n"] for v in admission.values())
                point_units = math.fsum(v["point"]["effective_work_units"] for v in admission.values())
                upper_units = math.fsum(v["upper"]["effective_work_units"] for v in admission.values())
                synth_mean = point_units * candidate_mean
                synth_upper = upper_units * candidate_upper
                result["synthetic_runtime"] = {
                    "method": synthetic_runtime_method,
                    "measured_candidate_mean_seconds": candidate_mean,
                    "conservative_candidate_seconds": candidate_upper,
                    "remaining_mean_seconds": synth_mean, "remaining_upper_seconds": synth_upper,
                    "admission_adjustment": {
                        "per_split": admission,
                        "point_prefix_candidates": point_total,
                        "upper_prefix_candidates": upper_total,
                        "point_effective_work_units": point_units,
                        "upper_effective_work_units": upper_units,
                        "point_extra_work_units": point_units - point_total,
                        "upper_extra_work_units": upper_units - upper_total,
                        "point_work_multiplier": point_units / point_total if point_total else 1.,
                        "upper_work_multiplier": upper_units / upper_total if upper_total else 1.,
                        "point_extra_seconds": (point_units - point_total) * candidate_mean,
                        "upper_extra_seconds": (upper_units - upper_total) * candidate_upper,
                        "policy": "Apply h2_admission.group_candidates separately to each split's requested point/upper prefix. Sum selected_workers/admission_workers per candidate and multiply by measured end-to-end benchmark seconds/item; no second division by workers. Use configured admission workers even for short groups.",
                        "limitation": "Concurrency correction cannot estimate additional per-structure runtime of unusually large or dense graphs absent from the40-case benchmark. Formal observed batch-wall updates remain necessary. When a requested prefix exceeds its frozen reserve, this computes reserve exhaustion cost, not guaranteed target completion.",
                    },
                    "benchmark_candidates_discarded_from_formal": True,
                    "dedup_note": "Frozen H2 candidates have globally unique reduced compositions; species-respecting StructureMatcher cannot merge different compositions. Physics yield remains uncertain across unused template groups.",
                }
        else:
            block.append("Invalid serial benchmark physics success counts")
    if block or synth_mean is None or synth_upper is None:
        result.update(status="blocked_forecast", fits_budget=False,
                      remaining_total_mean_seconds=None, remaining_total_upper_seconds=None,
                      upper_parts_seconds=None)
        return result
    # Sum the two marginal97.5% runtime upper bounds plus Wilson's lower yield
    # for planning; this is not a jointly calibrated overall95% confidence band.
    upper_parts = {"real": max(real_mean, real_ci[1]), "synthetic": synth_upper,
                   "evaluation": EVALUATION_RESERVE_SECONDS}
    contingency = CONTINGENCY_FRACTION * math.fsum(upper_parts.values())
    upper_parts["contingency"] = contingency
    upper = math.fsum(upper_parts.values())
    point = real_mean + synth_mean + EVALUATION_RESERVE_SECONDS
    result.update(status="forecast_ready", upper_parts_seconds=upper_parts,
                  remaining_real_seconds_upper=upper_parts["real"],
                  remaining_synthetic_seconds_upper=upper_parts["synthetic"],
                  remaining_total_seconds_upper=upper,
                  remaining_total_seconds_point=point,
                  remaining_total_mean_seconds=point,
                  remaining_total_upper_seconds=upper,
                  remaining_total_mean_hours=point / 3600.,
                  remaining_total_upper_hours=upper / 3600.,
                  fits_budget=upper <= remaining,
                  budget_margin_seconds=remaining - upper,
                  decision="proceed_under_current_forecast" if upper <= remaining else "save_progress_and_stop_before_formal_dataset",
                  overall_interval_note="Sum of separate conditional97.5% runtime upper bounds and a Wilson95% yield lower bound, plus5% contingency and30min evaluation reserve; a conservative planning rule, not a calibrated joint95% CI.")
    result["details"] = {
        "real_formula": "sum_h((N_h-n_calibration_h)*mean(effective_planning_worker_seconds_h))/selected_max_workers * max(1,calibration_wall/(sum_all_effective_planning_worker_seconds/selected_max_workers))",
        "real_upper_formula": "97.5percentile of2000 independent within-stratum replacement bootstraps, with frozen population counts and measured makespan correction",
        "synthetic_formula": "For each split, take prefix ceil(target/Wilson95yield_lower) capped by frozen reserve; sum(selected_workers/admission_workers) across that prefix; multiply effective units by selected-worker end-to-end seconds/item upper, without another worker division",
        "budget_formula": "1.05*(real_upper+synthetic_upper+1800sec)",
        "completed_calibration_excluded_from_remaining_work": True,
    }
    return result
