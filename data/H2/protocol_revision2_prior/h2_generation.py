"""Deterministic, label-blind inputs for finite H2; no writes, GPU or LLM.

prepare_plan(log=None) returns (info, plan). The plan contains 40 fresh benchmark
candidates, exactly 4000 real records, and generous, ordered TEST/POOL reserves.
Only the runner decides which physically accepted reserve records meet the
final quotas; this module never reads a relaxed result or a learned prediction.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import Counter, defaultdict
import copy
from fractions import Fraction
import functools
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability

import h1_generation as h1
import p1_data
from p1_verifier import elemental_references

ROOT = Path(__file__).resolve().parent
SEED = "H2-20261023-v1"
REAL_TRAIN_SEED = 20261023
RESERVE_SIZES = {"TEST": 2000, "POOL": 8000}
REAL_SIZE_BINS = ((1, 20), (21, 40), (41, 80), (81, 160), (161, None))
REVISED_TARGET_EXCLUSIONS = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"})
REVISED_MIN_VOLUME_PER_ATOM_A3 = 4.0
REVISED_MAX_ANGLE_PAIRS_3A = 20000
REVISED_GENERATION_POLICY = "H2-generation-protocol1-20261023"


def _say(log, value):
    if log is not None:
        log(value)


def _order(purpose, value):
    return hashlib.sha256(f"{SEED}|{purpose}|{value}".encode()).hexdigest()


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _composition_key(amounts):
    amounts = {symbol: Fraction(str(n)).limit_denominator(1000000)
               for symbol, n in amounts.items() if n > 0}
    lcm = math.lcm(*(x.denominator for x in amounts.values()))
    integer = {symbol: int(n * lcm) for symbol, n in amounts.items()}
    gcd = functools.reduce(math.gcd, integer.values())
    return "|".join(f"{symbol}:{integer[symbol] // gcd}" for symbol in sorted(integer))


def _replacement_key(structure, option):
    """Compute novelty before constructing or prescaling the candidate."""
    amounts = structure.composition.get_el_amt_dict()
    amounts[option["to"]] = amounts.pop(option["from"])
    return _composition_key(amounts)


def _options(record, structure, probability, available, counters):
    ionic, oxidation = h1._ionic_options(structure, probability, available,
                                        record["id"], counters)
    periodic = h1._periodic_options(structure, available, record["id"])
    # H2 has independent hashes while retaining the H1 eligibility definitions.
    ionic.sort(key=lambda x: _order("ionic-substitution",
                                   f"{record['id']}|{x['from']}|{x['to']}"))
    ionic_pairs = {(x["from"], x["to"]) for x in ionic}
    periodic = [x for x in periodic if (x["from"], x["to"]) not in ionic_pairs]
    periodic.sort(key=lambda x: _order("periodic-substitution",
                                      f"{record['id']}|{x['from']}|{x['to']}"))
    return {"ionic_probability": ionic, "periodic": periodic}, oxidation


def _make_candidate(template, option, method, split, index):
    record = h1._candidate(template["record"], template["structure"],
                           option, method, index)
    record.update({
        "id": f"H2-{'B' if split == 'BENCH' else ('T' if split == 'TEST' else 'P')}-{index:06d}",
        "dataset_kind": "synthetic", "split": split,
        "reduced_composition": _replacement_key(template["structure"], option),
        "novelty_prefilter_passed": True,
    })
    if p1_data.reduced_composition(record["initial_structure"]) != record["reduced_composition"]:
        raise AssertionError("Candidate novelty key differs after substitution")
    return record


def _select_candidates(templates, split, n, known, seen, counters, log=None,
                       candidate_prefilter=None):
    """Choose close to 70/30 accepted proposals with round-robin templates.

    Exhaust every eligible single-element substitution if necessary. A pair
    eligible for ionic substitution is classified as ionic even when it also
    meets the periodic rule, so candidate identity never depends on route.
    """
    queues = {method: [] for method in ("ionic_probability", "periodic")}
    for method in queues:
        level = 0
        while True:
            added = False
            for template in templates:
                options = template["options"][method]
                if level < len(options):
                    queues[method].append((template, options[level]))
                    added = True
            if not added:
                break
            level += 1
    positions = {method: 0 for method in queues}
    chosen = []

    def next_novel(method):
        while positions[method] < len(queues[method]):
            template, option = queues[method][positions[method]]
            positions[method] += 1
            counters[f"{split}_proposals_considered"] += 1
            counters[f"{split}_{method}_proposals_considered"] += 1
            key = _replacement_key(template["structure"], option)
            if key in known:
                counters[f"{split}_rejected_matbench_before_relaxation"] += 1
                continue
            if key in seen:
                counters[f"{split}_rejected_prior_or_duplicate_synthetic_composition"] += 1
                continue
            candidate = None
            if candidate_prefilter is not None:
                candidate = _make_candidate(template, option, method, split, len(chosen))
                if not candidate_prefilter(candidate):
                    continue
            seen.add(key)
            return template, option, candidate
        return None

    while len(chosen) < n:
        # 28/12 for benchmark and 1400/600 or5600/2400 for formal reserves.
        method = "ionic_probability" if len(chosen) % 10 < 7 else "periodic"
        item = next_novel(method)
        if item is None:
            method = "periodic" if method == "ionic_probability" else "ionic_probability"
            item = next_novel(method)
        if item is None:
            raise RuntimeError(f"Exhausted label-blind {split} proposals at {len(chosen)}/{n}")
        chosen.append(item[2] if item[2] is not None else
                      _make_candidate(item[0], item[1], method, split, len(chosen)))
        counters[f"{split}_{method}_selected"] += 1
        if len(chosen) % 2000 == 0:
            _say(log, f"H2 novel reserve {split}: {len(chosen)}/{n}")
    counters[f"{split}_selected"] = len(chosen)
    counters[f"{split}_unexamined_ionic_options"] = len(queues["ionic_probability"]) - positions["ionic_probability"]
    counters[f"{split}_unexamined_periodic_options"] = len(queues["periodic"]) - positions["periodic"]
    return chosen


def _real_candidates(records):
    """Exactly 2000 fixed random TRAIN rows plus all2000 frozen HOLDOUT rows."""
    train = [r for r in records if r["split"] == "TRAIN"]
    holdout = [r for r in records if r["split"] == "HOLDOUT"]
    chosen_ids = set(random.Random(REAL_TRAIN_SEED).sample(sorted(r["id"] for r in train), 2000))
    selected = sorted((r for r in train if r["id"] in chosen_ids), key=lambda r: r["id"])
    selected += sorted(holdout, key=lambda r: r["id"])
    if len(selected) != 4000 or len(holdout) != 2000:
        raise ValueError("Unexpected real TRAIN/HOLDOUT size")
    result = []
    for record in selected:
        structure = h1.clean_structure(record["structure"])
        result.append({
            "id": f"H2-R{record['split']}-{record['id']}",
            "original_id": record["id"], "idOriginal": record["id"],
            "source_index": record["source_index"],
            "dataset_kind": "real", "split": record["split"],
            "group": record["group"], "reduced_composition": record["group"],
            "template_id": record["id"], "template_group": record["group"],
            "template_formula": structure.composition.reduced_formula,
            "formula": structure.composition.reduced_formula, "n_sites": len(structure),
            "template_volume_A3": float(structure.volume),
            "initial_volume_A3": float(structure.volume), "lattice_scale_factor": 1.0,
            "initial_structure": structure.as_dict(),
            "dft_energy_eV_per_atom": float(record["target"]),
            "dft_target_policy": "Reference metadata for final real-data report only; never selection, prompts or feature selection",
        })
    return result


def _real_calibration(real):
    """Fixed size/split strata, including extremes; all labels count toward4000."""
    ids, strata = [], []
    for split in ("TRAIN", "HOLDOUT"):
        for low, high in REAL_SIZE_BINS:
            rows = [r for r in real if r["split"] == split and
                    r["n_sites"] >= low and (high is None or r["n_sites"] <= high)]
            label = f"{low}-{high if high is not None else 'inf'}"
            if not rows:
                strata.append({"split": split, "size_bin": label,
                               "population": 0, "calibration_ids": []})
                continue
            ordered = sorted(rows, key=lambda r: (r["n_sites"], r["id"]))
            selected = [ordered[0]]
            if ordered[-1]["id"] != selected[0]["id"]:
                selected.append(ordered[-1])
            seen = {r["id"] for r in selected}
            for row in sorted(rows, key=lambda r: _order("real-calibration", r["id"])):
                if len(selected) == min(6, len(rows)):
                    break
                if row["id"] not in seen:
                    selected.append(row)
                    seen.add(row["id"])
            chosen = [r["id"] for r in selected]
            ids.extend(chosen)
            strata.append({"split": split, "size_bin": label,
                           "population": len(rows), "n_sites_min": ordered[0]["n_sites"],
                           "n_sites_max": ordered[-1]["n_sites"], "calibration_ids": chosen})
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate real calibration record")
    return ids, strata


def prepare_plan(log=None):
    """Prepare all H2 inputs without modifying any file or using any labels.

    Real DFT targets are copied into separate candidate metadata only after
    TRAIN sampling and template/substitution choices have been frozen.
    """
    pilot_hash = h1._digest(p1_data.PILOT_PATH)
    if pilot_hash != h1.PILOT_SHA256:
        raise ValueError("Frozen P1 pilot SHA256 mismatch")
    h1_config_path = ROOT / "H1_synthetic_pilot" / "config.json"
    h1_inputs_path = ROOT / "work" / "h-data" / "h1_frozen_inputs.json.gz"
    known_path = ROOT / "work" / "h-data" / "matbench_compositions.json.gz"
    h1_config = _read_json(h1_config_path)
    if h1._digest(known_path) != h1_config["known_compositions_sha256"]:
        raise ValueError("Frozen full Matbench composition inventory SHA256 mismatch")
    if h1._digest(h1_inputs_path) != h1_config["frozen_inputs_sha256"]:
        raise ValueError("Frozen H1 inputs SHA256 mismatch")
    known = set(_read_gz(known_path))
    if len(known) != 95118 or len(known) != h1_config["n_known_matbench_compositions"]:
        raise ValueError("Unexpected full Matbench unique-composition count")
    h1_rows = _read_gz(h1_inputs_path)["candidates"]
    h1_template_ids = {r["template_id"] for r in h1_rows}
    h1_template_groups = {r["template_group"] for r in h1_rows}
    prior_compositions = {p1_data.reduced_composition(r["initial_structure"]) for r in h1_rows}
    prohibited_compositions = known | prior_compositions
    payload = _read_gz(p1_data.PILOT_PATH)
    records = payload["records"]
    if len(records) != 10000 or sum(r["split"] == "TRAIN" for r in records) != 8000:
        raise ValueError("Unexpected frozen P1 population")
    train_groups = {r["group"] for r in records if r["split"] == "TRAIN"}
    holdout_groups = {r["group"] for r in records if r["split"] == "HOLDOUT"}
    if train_groups & holdout_groups:
        raise ValueError("Frozen P1 TRAIN/HOLDOUT composition overlap")
    # Strip every DFT target and original structure/site property from the
    # template path. Only the final real-output assembly below copies targets.
    train = [{k: r[k] for k in ("id", "group", "structure")}
             for r in records if r["split"] == "TRAIN"]
    available = set()
    for symbol in elemental_references():
        try:
            h1.covalent_radius(symbol)
            available.add(symbol)
        except ValueError:
            pass
    counters = Counter({"train_records": 8000})
    eligible = []
    for record in train:
        if record["id"] in h1_template_ids or record["group"] in h1_template_groups:
            counters["excluded_H1_template_or_composition_group"] += 1
            continue
        if len(record["structure"]["sites"]) > 20:
            counters["excluded_more_than_20_sites"] += 1
            continue
        try:
            structure = h1.clean_structure(record["structure"])
        except (ValueError, KeyError, TypeError):
            counters["excluded_invalid_or_disordered_structure"] += 1
            continue
        if {str(el) for el in structure.composition.elements} - available:
            counters["excluded_missing_reference_or_covalent_radius"] += 1
            continue
        if p1_data.reduced_composition(record["structure"]) != record["group"]:
            raise ValueError(f"P1 composition mismatch: {record['id']}")
        eligible.append((record, structure))
    eligible.sort(key=lambda x: _order("template", x[0]["id"]))
    counters["ordered_size_reference_eligible"] = len(eligible)
    probability = SubstitutionProbability()
    templates = []
    for index, (record, structure) in enumerate(eligible):
        counters["replacement_evaluated"] += 1
        options, oxidation = _options(record, structure, probability, available, counters)
        if len(options["ionic_probability"]) < 3:
            counters["excluded_fewer_than_3_distinct_ionic_replacements"] += 1
            continue
        if not options["periodic"]:
            counters["excluded_no_distinct_periodic_replacement"] += 1
            continue
        templates.append({"record": record, "structure": structure,
                          "options": options, "oxidation_guess": oxidation})
        if (index + 1) % 500 == 0:
            _say(log, f"H2 label-blind template options: {index + 1}/{len(eligible)} eligible examined")
    counters["templates_with_both_generation_routes"] = len(templates)
    # Reserve benchmark groups before any relaxation. Benchmark uses separate
    # template composition groups, so concurrency tuning cannot adapt TEST.
    bygroup = defaultdict(list)
    for template in templates:
        bygroup[template["record"]["group"]].append(template)
    bench_templates, bench_groups = [], set()
    for group in sorted(bygroup, key=lambda g: _order("benchmark-template-group", g)):
        template = bygroup[group][0]
        if not any(_replacement_key(template["structure"], option) not in prohibited_compositions
                   for options in template["options"].values() for option in options):
            continue
        bench_templates.append(template)
        bench_groups.add(group)
        if len(bench_templates) == 40:
            break
    if len(bench_templates) != 40:
        raise RuntimeError("Cannot reserve40 fresh benchmark template groups")
    split_templates = {"TEST": [], "POOL": []}
    group_assignment = {}
    for group in bygroup:
        if group in bench_groups:
            group_assignment[group] = "BENCH"
        else:
            # Fixed approximately1:5 split, wholly independent of labels.
            group_assignment[group] = "TEST" if int(_order("template-split", group), 16) % 6 == 0 else "POOL"
    for template in templates:
        split = group_assignment[template["record"]["group"]]
        if split != "BENCH":
            split_templates[split].append(template)
    seen = set(prior_compositions)
    benchmark = _select_candidates(bench_templates, "BENCH", 40, known, seen, counters, log)
    synthetics = {}
    for split in ("TEST", "POOL"):
        synthetics[split] = _select_candidates(split_templates[split], split, RESERVE_SIZES[split],
                                              known, seen, counters, log)
    real = _real_candidates(records)
    real_calibration_ids, real_calibration_strata = _real_calibration(real)
    template_manifest = [{
        "id": t["record"]["id"], "group": t["record"]["group"],
        "split": group_assignment[t["record"]["group"]],
        "n_sites": len(t["structure"]), "oxidation_guess": t["oxidation_guess"],
        "n_ionic_options": len(t["options"]["ionic_probability"]),
        "n_periodic_options": len(t["options"]["periodic"]),
    } for t in templates]
    all_synthetic = benchmark + synthetics["TEST"] + synthetics["POOL"]
    if len({r["reduced_composition"] for r in all_synthetic}) != len(all_synthetic):
        raise AssertionError("Synthetic reduced-composition duplicate")
    if any(r["reduced_composition"] in prohibited_compositions for r in all_synthetic):
        raise AssertionError("Synthetic Matbench or H1 composition overlap")
    plan = {"benchmark_candidates": benchmark, "real_candidates": real,
            "real_calibration_ids": real_calibration_ids,
            "synthetic_candidates": synthetics,
            "template_manifest": template_manifest,
            "known_composition_count": len(known),
            "known_compositions_sha256": h1._digest(known_path)}
    info = {
        "seed": SEED, "real_train_random_seed": REAL_TRAIN_SEED,
        "reserve_sizes": RESERVE_SIZES, "target_retained_sizes": {"TEST": 1000, "POOL": 5000},
        "benchmark_size": 40, "real_sizes": {"TRAIN": 2000, "HOLDOUT": 2000},
        "real_calibration_count": len(real_calibration_ids),
        "real_calibration_strata": real_calibration_strata,
        "real_calibration_policy": "Up to6 fixed records per TRAIN/HOLDOUT x n_sites bin1-20,21-40,41-80,81-160,161+; include each stratum's smallest/largest structure and fill by independent H2 hash. Their formal labels are reused. Estimate total real runtime by population-weighted mean measured elapsed seconds within strata.",
        "generation_counts": dict(counters),
        "template_counts_by_split": dict(Counter(t["split"] for t in template_manifest)),
        "template_group_counts_by_split": dict(Counter(group_assignment.values())),
        "template_manifest_sha256": h1._json_hash(template_manifest),
        "benchmark_candidates_sha256": h1._json_hash(benchmark),
        "synthetic_candidates_sha256": {s: h1._json_hash(c) for s, c in synthetics.items()},
        "real_candidates_sha256": h1._json_hash(real),
        "real_train_ids_sha256": h1._json_hash([r["original_id"] for r in real if r["split"] == "TRAIN"]),
        "p1_pilot_path": str(p1_data.PILOT_PATH), "p1_pilot_sha256": pilot_hash,
        "matbench_source_sha256": p1_data.EXPECTED_SHA256,
        "matbench_composition_inventory_path": str(known_path),
        "matbench_composition_inventory_sha256": h1._digest(known_path),
        "matbench_unique_compositions": len(known),
        "h1_frozen_inputs_sha256": h1._digest(h1_inputs_path),
        "h1_excluded_template_count": len(h1_template_ids),
        "h1_excluded_template_group_count": len(h1_template_groups),
        "h1_excluded_candidate_composition_count": len(prior_compositions),
        "selection": "Frozen P1 TRAIN only; ordered<=20sites, elemental references/radii, >=3 ionic options and distinct periodic route. Exclude every H1 template composition group. Independent H2 SHA256 ordering; no energy label or relaxed result used.",
        "split_policy": "Reserve40 fresh BENCH template groups before relaxation. Assign all remaining template composition groups via SHA256(seed|template-split|group)%6:0 TEST, others POOL. Every polymorph of a template composition shares a split.",
        "candidate_policy": "Single-element whole-species replacement, H1 ion probability threshold0.001 and periodic criterion, H1 covalent-radius lattice prescale. Round-robin templates before extra options. Prefer7ionic/3periodic novel selections per10, fallback only if a route exhausts. All eligible options may be used if needed.",
        "probability_reproducibility": "Pymatgen default probability normalization accumulates floating point marginals over a set. Preparation/launcher fixes PYTHONHASHSEED=0 and freezes exact candidate JSON; changing that environment may alter insignificant probability roundoff without changing candidate identity.",
        "novelty_policy": "Before constructing/prescaling and before GPU relaxation, exclude all95118 full-Matbench reduced compositions, all300 H1 candidate compositions, and any previously selected H2 synthetic composition across BENCH/TEST/POOL. Exact composition uniqueness also prevents StructureMatcher duplicates under its species-respecting H1 settings; runner retains post-relaxation dedup as an explicit check.",
        "real_selection": "Python random.Random(20261023).sample(sorted8000TRAIN ids,2000); all2000 frozen HOLDOUT; never filter real records by the synthetic size or novelty criteria.",
        "label_boundary": "No DFT/CHGNet energy participates in template/substitution/split/TRAIN-row choices. Original DFT targets copied only to real metadata after choices; labels remain excluded from prompts and feature selection.",
        "provenance": h1._provenance(),
    }
    _say(log, f"H2 prepared40 benchmark,2000 TEST reserve,8000 POOL reserve,4000 real; novelty prefiltered; counts={dict(counters)}")
    return info, plan


def _initial_geometry_violations(volume_per_atom, angle_pairs):
    """Inclusive acceptance at exactly4 A3/atom and exactly20000 angle pairs."""
    return (volume_per_atom < REVISED_MIN_VOLUME_PER_ATOM_A3,
            angle_pairs > REVISED_MAX_ANGLE_PAIRS_3A)


def _initial_geometry_prefilter(candidate, counters):
    """Independent volume/angle checks on prescaled geometry, before labels.

    Compute both conditions even when volume fails, allowing their overlap and
    each independent rejection count to be reported without order ambiguity.
    """
    structure = h1.clean_structure(candidate["initial_structure"])
    per_atom = float(structure.volume) / len(structure)
    centers, _, _, distances = structure.get_neighbor_list(3.0, exclude_self=True)
    if not math.isfinite(per_atom) or not all(math.isfinite(float(d)) for d in distances):
        raise ValueError("Nonfinite prescaled geometry in revised generation")
    degree = Counter(map(int, centers))
    angles = sum(int(d) * (int(d) - 1) for d in degree.values())
    low_volume, high_angles = _initial_geometry_violations(per_atom, angles)
    split = candidate["split"]
    counters[f"{split}_initial_geometry_evaluated"] += 1
    counters[f"{split}_initial_volume_per_atom_below_4_A3"] += int(low_volume)
    counters[f"{split}_initial_angle_pairs_3A_above_20000"] += int(high_angles)
    counters[f"{split}_initial_geometry_both_conditions"] += int(low_volume and high_angles)
    counters[f"{split}_initial_geometry_union_rejected"] += int(low_volume or high_angles)
    if low_volume or high_angles:
        first = "initial_volume_per_atom_below_4_A3" if low_volume else "initial_angle_pairs_3A_above_20000"
        counters[f"{split}_first_rejection_{first}"] += 1
        return False
    candidate.update({
        "generation_prefilter_revision": REVISED_GENERATION_POLICY,
        "generation_prefilter_passed": True,
        "initial_volume_per_atom_A3": per_atom,
        "initial_angle_pairs_3A": int(angles),
        "initial_directed_neighbors_3A": int(len(centers)),
        "initial_max_degree_3A": max(degree.values(), default=0),
    })
    counters[f"{split}_initial_geometry_passed"] += 1
    return True


def prepare_revised_plan(previous_plan, log=None):
    """Regenerate formal reserves under the authorized label-blind screens.

    Preserve benchmark structures, real records, calibration IDs and the full
    template manifest exactly. Only formal replacement targets and prescaled
    geometries receive the new screens. No relaxed outputs or labels are read;
    original real DFT metadata is copied opaquely from previous_plan.
    """
    preserved_keys = ("benchmark_candidates", "real_candidates",
                      "real_calibration_ids", "template_manifest")
    preserved_hashes = {key: h1._json_hash(previous_plan[key]) for key in preserved_keys}
    if len(previous_plan["benchmark_candidates"]) != 40:
        raise ValueError("Revised generation requires the exact old40 benchmark candidates")
    if len(previous_plan["real_candidates"]) != 4000 or len(previous_plan["real_calibration_ids"]) != 60:
        raise ValueError("Revised generation requires the frozen4000 real/60 calibration identities")
    pilot_hash = h1._digest(p1_data.PILOT_PATH)
    if pilot_hash != h1.PILOT_SHA256:
        raise ValueError("Frozen P1 pilot SHA256 mismatch")
    h1_config = _read_json(ROOT / "H1_synthetic_pilot" / "config.json")
    known_path = ROOT / "work" / "h-data" / "matbench_compositions.json.gz"
    known_hash = h1._digest(known_path)
    if known_hash != h1_config["known_compositions_sha256"] or known_hash != previous_plan["known_compositions_sha256"]:
        raise ValueError("Frozen complete Matbench composition inventory changed")
    known = set(_read_gz(known_path))
    if len(known) != 95118 or len(known) != previous_plan["known_composition_count"]:
        raise ValueError("Unexpected Matbench composition inventory size")
    h1_inputs_path = ROOT / "work" / "h-data" / "h1_frozen_inputs.json.gz"
    if h1._digest(h1_inputs_path) != h1_config["frozen_inputs_sha256"]:
        raise ValueError("Frozen H1 inputs SHA256 mismatch")
    h1_rows = _read_gz(h1_inputs_path)["candidates"]
    prior_compositions = {p1_data.reduced_composition(r["initial_structure"]) for r in h1_rows}
    prohibited_compositions = known | prior_compositions
    h1_template_ids = {r["template_id"] for r in h1_rows}
    h1_template_groups = {r["template_group"] for r in h1_rows}
    payload = _read_gz(p1_data.PILOT_PATH)
    # Read only IDs/splits/composition groups/geometries from the frozen source;
    # no target is accessed or copied into the new synthetic generation path.
    train = {r["id"]: {k: r[k] for k in ("id", "group", "structure")}
             for r in payload["records"] if r["split"] == "TRAIN"}
    del payload
    available = set()
    for symbol in elemental_references():
        try:
            h1.covalent_radius(symbol)
            available.add(symbol)
        except ValueError:
            pass
    counters = Counter()
    probability = SubstitutionProbability()
    templates = {"TEST": [], "POOL": []}
    assignment = {}
    manifest_ids = set()
    for entry in previous_plan["template_manifest"]:
        if entry["id"] in manifest_ids:
            raise ValueError("Duplicate frozen template ID")
        manifest_ids.add(entry["id"])
        if entry["split"] not in ("BENCH", "TEST", "POOL"):
            raise ValueError("Unknown frozen template split")
        if entry["group"] in assignment and assignment[entry["group"]] != entry["split"]:
            raise ValueError("Frozen template composition group crosses splits")
        assignment[entry["group"]] = entry["split"]
        if entry["split"] == "BENCH":
            continue
        split = entry["split"]
        record = train.get(entry["id"])
        if record is None or record["group"] != entry["group"]:
            raise ValueError(f"Frozen template not found in TRAIN: {entry['id']}")
        if record["id"] in h1_template_ids or record["group"] in h1_template_groups:
            raise ValueError("Frozen formal template overlaps H1 template composition")
        structure = h1.clean_structure(record["structure"])
        if len(structure) != entry["n_sites"] or len(structure) > 20:
            raise ValueError("Frozen template size differs")
        options, oxidation = _options(record, structure, probability, available, counters)
        retained_options = {}
        for method, values in options.items():
            counters[f"{split}_{method}_options_before_target_exclusion"] += len(values)
            retained_options[method] = []
            for option in values:
                if option["to"] in REVISED_TARGET_EXCLUSIONS:
                    counters[f"{split}_excluded_target_noble_gas"] += 1
                    counters[f"{split}_{method}_excluded_target_noble_gas"] += 1
                    counters[f"{split}_excluded_target_noble_gas_{option['to']}"] += 1
                else:
                    retained_options[method].append(option)
            counters[f"{split}_{method}_options_after_target_exclusion"] += len(retained_options[method])
        # Do not rerun the old >=3 ionic/periodic eligibility rule after the
        # new target exclusion: all old template identities/splits stay fixed.
        templates[split].append({"record": record, "structure": structure,
                                 "options": retained_options, "oxidation_guess": oxidation})
        counters[f"{split}_frozen_templates_retained"] += 1
        if len(retained_options["ionic_probability"]) < 3:
            counters[f"{split}_retained_templates_now_fewer_than_3_ionic_options"] += 1
        if not any(retained_options.values()):
            counters[f"{split}_retained_templates_with_no_remaining_options"] += 1
    benchmark = previous_plan["benchmark_candidates"]
    bench_compositions = {p1_data.reduced_composition(r["initial_structure"]) for r in benchmark}
    if len(bench_compositions) != 40:
        raise ValueError("Frozen benchmark composition identities are not unique")
    if {r["template_group"] for r in benchmark} - {g for g, s in assignment.items() if s == "BENCH"}:
        raise ValueError("Frozen benchmark templates disagree with old manifest")
    seen = prior_compositions | bench_compositions
    synthetics = {}
    for split in ("TEST", "POOL"):
        synthetics[split] = _select_candidates(
            templates[split], split, RESERVE_SIZES[split], known, seen, counters, log,
            candidate_prefilter=lambda row: _initial_geometry_prefilter(row, counters))
    real_calibration_ids, strata = _real_calibration(previous_plan["real_candidates"])
    if real_calibration_ids != previous_plan["real_calibration_ids"]:
        raise ValueError("Frozen calibration IDs differ from the original deterministic selection")
    plan = copy.deepcopy(previous_plan)
    plan["synthetic_candidates"] = synthetics
    for key, digest in preserved_hashes.items():
        if h1._json_hash(plan[key]) != digest or h1._json_hash(previous_plan[key]) != digest:
            raise AssertionError(f"Revised generation changed immutable section {key}")
    all_synthetic = benchmark + synthetics["TEST"] + synthetics["POOL"]
    if len({r["reduced_composition"] for r in all_synthetic}) != len(all_synthetic):
        raise AssertionError("Revised synthetic composition duplicate")
    if any(r["reduced_composition"] in prohibited_compositions for r in all_synthetic):
        raise AssertionError("Revised synthetic overlaps Matbench or H1 composition")
    for split, rows in synthetics.items():
        if any(assignment[r["template_group"]] != split or r["substitution_to"] in REVISED_TARGET_EXCLUSIONS for r in rows):
            raise AssertionError("Revised formal split/target screen violated")
    prefilter_counts = {
        key: sum(counters[f"{split}_{key}"] for split in ("TEST", "POOL"))
        for key in ("excluded_target_noble_gas", "initial_volume_per_atom_below_4_A3",
                    "initial_angle_pairs_3A_above_20000", "initial_geometry_both_conditions",
                    "initial_geometry_union_rejected")}
    prefilter_counts["first_rejection"] = {
        key: sum(counters[f"{split}_first_rejection_{key}"] for split in ("TEST", "POOL"))
        for key in ("initial_volume_per_atom_below_4_A3", "initial_angle_pairs_3A_above_20000")}
    info = {
        "seed": SEED, "generation_revision": REVISED_GENERATION_POLICY,
        "real_train_random_seed": REAL_TRAIN_SEED,
        "reserve_sizes": RESERVE_SIZES, "target_retained_sizes": {"TEST": 1000, "POOL": 5000},
        "benchmark_size": 40, "real_sizes": {"TRAIN": 2000, "HOLDOUT": 2000},
        "real_calibration_count": 60, "real_calibration_strata": strata,
        "real_calibration_policy": "Reuse the exact previous60 real calibration IDs and4000 real rows without applying new target/geometry screens.",
        "generation_counts": dict(counters), "prefilter_counts": prefilter_counts,
        "prefilter_policy": {
            "excluded_replacement_targets": sorted(REVISED_TARGET_EXCLUSIONS),
            "minimum_initial_volume_per_atom_A3": REVISED_MIN_VOLUME_PER_ATOM_A3,
            "maximum_initial_angle_pairs_3A": REVISED_MAX_ANGLE_PAIRS_3A,
            "angle_definition": "sum_i degree_3A(i)*(degree_3A(i)-1); periodic directed neighbor list, exclude_self=True, cutoff3.0A, prescaled unrelaxed candidate",
            "strict_rejections": "target in He/Ne/Ar/Kr/Xe/Rn OR initial_volume_per_atom<4A3 OR initial_angle_pairs_3A>20000; exactly4 and20000 pass",
            "target_count_unit": "Replacement options excluded in the fixed formal template inventory before novelty/geometry evaluation; only replacement target is screened, existing template elements are not additionally restricted.",
            "geometry_count_unit": "Novel nonduplicate template/replacement proposals examined to refill formal reserves; volume/angle counts are independent, overlap+union recorded, first reason volume before angle. A failed geometry does not reserve its composition, allowing another template geometry to be considered.",
            "order": "Target exclusion on options, then full-Matbench/prior/duplicate composition exclusion, then both initial geometry checks, all before any relaxed label.",
            "benchmark_exception": "Preserve old same40 benchmark structures unchanged; no retrospective target/geometry exclusion.",
            "real_exception": "Preserve all4000 real structures/DFT metadata and60 calibration IDs unchanged; no new generation screen applies.",
        },
        "template_counts_by_split": dict(Counter(t["split"] for t in plan["template_manifest"])),
        "template_group_counts_by_split": dict(Counter(assignment.values())),
        "template_manifest_sha256": preserved_hashes["template_manifest"],
        "preserved_previous_section_sha256": preserved_hashes,
        "benchmark_candidates_sha256": preserved_hashes["benchmark_candidates"],
        "synthetic_candidates_sha256": {s: h1._json_hash(c) for s, c in synthetics.items()},
        "real_candidates_sha256": preserved_hashes["real_candidates"],
        "real_train_ids_sha256": h1._json_hash([r["original_id"] for r in plan["real_candidates"] if r["split"] == "TRAIN"]),
        "p1_pilot_path": str(p1_data.PILOT_PATH), "p1_pilot_sha256": pilot_hash,
        "matbench_source_sha256": p1_data.EXPECTED_SHA256,
        "matbench_composition_inventory_path": str(known_path),
        "matbench_composition_inventory_sha256": known_hash,
        "matbench_unique_compositions": len(known),
        "h1_frozen_inputs_sha256": h1._digest(h1_inputs_path),
        "h1_excluded_template_count": len(h1_template_ids),
        "h1_excluded_template_group_count": len(h1_template_groups),
        "h1_excluded_candidate_composition_count": len(prior_compositions),
        "selection": "Only previous formal template identities in their previous manifest order; no new template or split assignment, no post-target-screen >=3-option eligibility rerun.",
        "split_policy": "Exact previous template manifest and BENCH groups preserved; all formal candidates retain their template composition group's old TEST/POOL split.",
        "candidate_policy": "Use original H2 per-method option hashes and round-robin template ordering, removing prohibited targets and initial geometries and refilling with subsequent label-blind proposals. Prefer7ionic/3periodic accepted proposals per10; fallback only if a route exhausts.",
        "probability_reproducibility": "Run preparation with PYTHONHASHSEED=0, then freeze exact JSON. Default pymatgen set-order probability normalization can otherwise alter floating roundoff.",
        "novelty_policy": "Full95118-composition Matbench inventory, every H1 candidate composition, unchanged40 BENCH compositions, and previously selected revised formal compositions excluded before any relaxation; global exact synthetic composition uniqueness.",
        "real_selection": "Preserved previous real4000 records opaquely; no target accessed for generation and no real row reselected.",
        "label_boundary": "No relaxed result, predicted/DFT energy, convergence flag, or benchmark outcome is used for template, replacement, geometry, or split choices. Original real metadata is copied without inspecting its target.",
        "provenance": h1._provenance(),
    }
    _say(log, f"H2 revised label-blind reserves TEST2000/POOL8000; immutable BENCH40/real4000/calibration60/templates verified; prefilter_counts={prefilter_counts}")
    return info, plan


def generate_candidates(plan, split, max_candidates=None):
    """Read the frozen deterministic reserve, never use result-driven generation."""
    rows = plan["synthetic_candidates"][split]
    return list(rows if max_candidates is None else rows[:max_candidates])


def select_s0(pool_records, seed, n=500):
    """500 labels from retained POOL, seed0/1/2, selected without label access."""
    if seed not in (0, 1, 2):
        raise ValueError("H2 only freezes seeds0/1/2")
    ids = sorted(r["id"] for r in pool_records)
    if len(set(ids)) != len(ids) or len(ids) < n:
        raise ValueError("Invalid retained POOL ids")
    return random.Random(seed).sample(ids, n)


if __name__ == "__main__":
    info, plan = prepare_plan(print)
    print(json.dumps({"self_check": "passed", "info": info}, ensure_ascii=False, sort_keys=True))
