"""Read-only, label-blind candidate generation for the finite H1 pilot.

Public API: prepare_candidates(log), full_composition_set(log),
covalent_radius(symbol), clean_structure(raw_or_struct). No GPU, LLM, cache,
download, or result-file writes occur in this module. Call with Python -B.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path

from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.structure_prediction import substitution_probability
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
from pymatgen.core import Element, Structure

import p1_data
from p1_verifier import (CHECKPOINT, CHECKPOINT_FILENAME, REF_PATH, REF_SHA256,
                         elemental_references)

ROOT = Path(__file__).resolve().parent
SEED = "H1-20261023-v1"
IONIC_THRESHOLD = 0.001
PILOT_SHA256 = "e1be621f7c67f3cc9537d112be94cd3ee228d2ae18aae48418e6fb8a6664e6d8"


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _order(purpose, value):
    return hashlib.sha256(f"{SEED}|{purpose}|{value}".encode()).hexdigest()


def _say(log, text):
    if log is not None:
        log(text)


@lru_cache(maxsize=None)
def covalent_radius(symbol):
    """Pymatgen's Cordero et al. covalent radius in Angstrom, no fallback."""
    value = CovalentRadius.radius.get(str(symbol))
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"No covalent radius for {symbol}")
    return float(value)


def clean_structure(raw_or_struct):
    """Fresh neutral-element Structure; discard all DFT/site/global properties."""
    raw = raw_or_struct.as_dict() if isinstance(raw_or_struct, Structure) else raw_or_struct
    lattice = raw["lattice"]["matrix"]
    species, coordinates = [], []
    for site in raw["sites"]:
        if len(site["species"]) != 1 or not math.isclose(
                float(site["species"][0].get("occu", 1)), 1.0, abs_tol=1e-10):
            raise ValueError("H1 requires fully occupied ordered sites")
        species.append(site["species"][0]["element"])
        coordinates.append(site["abc"])
    if not species:
        raise ValueError("Empty structure")
    result = Structure(lattice, species, coordinates, site_properties=None,
                       to_unit_cell=True)
    if not result.is_ordered or not math.isfinite(result.volume) or result.volume <= 0:
        raise ValueError("Invalid ordered structure")
    return result


def _ionic_options(structure, probability, available, record_id, counters):
    """Use first ranked charge-balanced guess; max reduced composition 20 atoms.

    The pymatgen default common/ICSD oxidation-state list and occurrence ranking
    are retained (all_oxi_states=False); no site magnetization or DFT label is
    read. Only an integral, nonzero source oxidation state may be substituted.
    Other elements can retain their inferred average mixed-valence states.
    """
    try:
        guesses = structure.composition.oxi_state_guesses(
            all_oxi_states=False, max_sites=-20)
    except (ValueError, TypeError, OverflowError) as error:
        counters["oxidation_inference_exception"] += 1
        counters[f"oxidation_exception_{type(error).__name__}"] += 1
        return [], None
    if not guesses:
        counters["oxidation_no_charge_balanced_guess"] += 1
        return [], None
    oxidation = {str(k): float(v) for k, v in guesses[0].items()}
    present = {str(el) for el in structure.composition.elements}
    options = []
    for source in sorted(present):
        state = oxidation[source]
        if state == 0 or not math.isclose(state, round(state), abs_tol=1e-8):
            continue
        source_ions = [ion for ion in probability.species
                       if ion.symbol == source and float(ion.oxi_state) == state]
        if not source_ions:
            continue
        source_ion = source_ions[0]
        for target in sorted(probability.species, key=str):
            if target.symbol in present or target.symbol not in available:
                continue
            if float(target.oxi_state) != state:
                continue
            # First argument is variable; second is conditioned-on/fixed ion.
            conditional = float(probability.cond_prob(target, source_ion))
            if conditional >= IONIC_THRESHOLD:
                options.append({"from": source, "to": target.symbol,
                                "oxidation_state": int(round(state)),
                                "conditional_probability": conditional})
    options.sort(key=lambda x: _order("ionic-substitution", f"{record_id}|{x['from']}|{x['to']}"))
    if not options:
        counters["oxidation_guess_but_no_threshold_ionic_replacement"] += 1
    return options, oxidation


def _periodic_options(structure, available, record_id):
    """Same IUPAC group OR |period difference| == 1; other groups unrestricted.

    This literal interpretation of 'same group or adjacent period' can generate
    chemically severe perturbations; H1 relaxation/geometry filters assess them.
    A target already present in the template is always excluded.
    """
    present = {str(el) for el in structure.composition.elements}
    options = []
    for source in sorted(present):
        old = Element(source)
        for target in sorted(available - present):
            new = Element(target)
            if new.group == old.group or abs(int(new.row) - int(old.row)) == 1:
                options.append({"from": source, "to": target,
                                "oxidation_state": None,
                                "conditional_probability": None})
    options.sort(key=lambda x: _order("periodic-substitution", f"{record_id}|{x['from']}|{x['to']}"))
    return options


def _candidate(template, structure, option, method, index):
    source, target = option["from"], option["to"]
    old_comp = structure.composition.get_el_amt_dict()
    if target in old_comp or source not in old_comp:
        raise AssertionError("Replacement must change a full element without species merging")
    old_mean = math.fsum(n * covalent_radius(el) for el, n in old_comp.items()) / len(structure)
    new_comp = dict(old_comp)
    new_comp[target] = new_comp.pop(source)
    new_mean = math.fsum(n * covalent_radius(el) for el, n in new_comp.items()) / len(structure)
    scale = new_mean / old_mean
    changed = structure.copy()
    changed.replace_species({source: target})
    changed.scale_lattice(structure.volume * scale ** 3)
    changed = clean_structure(changed)
    return {
        "id": f"H1-{index:03d}", "template_id": template["id"],
        "template_group": template["group"],
        "template_formula": structure.composition.reduced_formula,
        "substitution_method": method, "substitution_from": source,
        "substitution_to": target,
        "conditional_probability": option["conditional_probability"],
        "oxidation_state": option["oxidation_state"],
        "formula": changed.composition.reduced_formula, "n_sites": len(changed),
        "template_volume_A3": float(structure.volume),
        "initial_volume_A3": float(changed.volume),
        "lattice_scale_factor": float(scale),
        "initial_structure": changed.as_dict(),
    }


def _provenance():
    # New pymatgen releases retain an analysis.* compatibility re-export; the
    # actual implementation and its data are under core.structure_prediction.
    module_path = Path(inspect.getfile(SubstitutionProbability)).resolve()
    lambda_path = module_path.parent / "data" / "lambda.json"
    chgnet_spec = importlib.util.find_spec("chgnet")
    if chgnet_spec is None or not chgnet_spec.origin:
        raise RuntimeError("P1 CHGNet package is unavailable")
    checkpoint_path = (Path(chgnet_spec.origin).parent / "pretrained" /
                       CHECKPOINT / CHECKPOINT_FILENAME)
    return {
        "substitution_model": "pymatgen SubstitutionProbability default lambda table; alpha=-5",
        "substitution_model_source": "https://doi.org/10.1021/ic102031h",
        "pymatgen_probability_documentation": "https://pymatgen.org/pymatgen.analysis.structure_prediction.html",
        "lambda_path": str(lambda_path), "lambda_sha256": _digest(lambda_path),
        "substitution_implementation_sha256": _digest(module_path),
        "radii_source": "Cordero et al., Covalent radii revisited (2008), https://doi.org/10.1039/B801115J; installed pymatgen CovalentRadius.radius",
        "radii_table_sha256": _json_hash(dict(sorted(CovalentRadius.radius.items()))),
        "radii_implementation_sha256": _digest(inspect.getfile(CovalentRadius)),
        "radii_table_A": {k: float(v) for k, v in sorted(CovalentRadius.radius.items())},
        "element_reference_path": str(REF_PATH), "element_reference_sha256": REF_SHA256,
        "chgnet_checkpoint_version": CHECKPOINT,
        "chgnet_checkpoint_path": str(checkpoint_path),
        "chgnet_checkpoint_sha256": _digest(checkpoint_path),
        "pymatgen_version": importlib.metadata.version("pymatgen"),
        "python_executable": sys.executable,
    }


def prepare_candidates(log=None):
    """Return (frozen configuration, 300 raw candidates, 8000 TRAIN DFT rows).

    No novelty screening occurs here: all 300 must first be relaxed. Selection
    only uses TRAIN split, ordered sites, size, elemental coverage, and chemical
    replacement eligibility, followed by independent deterministic hashes.
    """
    if not p1_data.PILOT_PATH.is_file():
        raise FileNotFoundError("Frozen P1 pilot required; H1 never prepares/modifies P1")
    pilot_hash = _digest(p1_data.PILOT_PATH)
    if pilot_hash != PILOT_SHA256:
        raise ValueError("Frozen P1 pilot SHA256 differs from completed P1 experiment")
    with gzip.open(p1_data.PILOT_PATH, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    records = payload["records"]
    if len(records) != 10000:
        raise ValueError("Expected frozen 10000-row P1 pilot")
    train = [r for r in records if r["split"] == "TRAIN"]
    holdout_groups = {r["group"] for r in records if r["split"] == "HOLDOUT"}
    if len(train) != 8000 or len(records) - len(train) != 2000:
        raise ValueError("Unexpected TRAIN/HOLDOUT sizes")
    if {r["group"] for r in train} & holdout_groups:
        raise ValueError("TRAIN and HOLDOUT composition groups overlap")
    # Retain no HOLDOUT structures, IDs or labels beyond this boundary.
    del records, payload
    train_summary = [{"id": r["id"], "group": r["group"],
                      "formation_energy_eV_per_atom": float(r["target"])} for r in train]
    # Remove even TRAIN targets from the entire selection/generation path.
    train = [{k: r[k] for k in ("id", "group", "structure")} for r in train]
    references = elemental_references()
    available = set()
    for symbol in references:
        try:
            covalent_radius(symbol)
            available.add(symbol)
        except ValueError:
            pass
    counters = Counter({"train_records": 8000, "train_ordered_size_reference_eligible": 0,
                        "replacement_evaluated": 0, "selected_templates": 0})
    eligible = []
    for record in train:
        raw = record["structure"]
        if len(raw["sites"]) > 20:
            counters["excluded_more_than_20_sites"] += 1
            continue
        try:
            structure = clean_structure(raw)
        except (ValueError, KeyError, TypeError):
            counters["excluded_invalid_or_disordered_structure"] += 1
            continue
        if {str(el) for el in structure.composition.elements} - available:
            counters["excluded_missing_reference_or_covalent_radius"] += 1
            continue
        if p1_data.reduced_composition(raw) != record["group"]:
            raise ValueError(f"P1 composition group mismatch: {record['id']}")
        counters["train_ordered_size_reference_eligible"] += 1
        eligible.append(record)
    eligible.sort(key=lambda r: _order("template", r["id"]))
    probability = SubstitutionProbability()
    selected, candidates, template_details = [], [], []
    for record in eligible:
        counters["replacement_evaluated"] += 1
        structure = clean_structure(record["structure"])
        ionic, oxidation = _ionic_options(structure, probability, available,
                                          record["id"], counters)
        if len(ionic) < 3:
            counters["excluded_fewer_than_3_distinct_ionic_replacements"] += 1
            continue
        periodic = _periodic_options(structure, available, record["id"])
        first_two = {(x["from"], x["to"]) for x in ionic[:2]}
        periodic = [x for x in periodic if (x["from"], x["to"]) not in first_two]
        if not periodic:
            counters["excluded_no_distinct_periodic_replacement"] += 1
            continue
        template_index = len(selected)
        options = [(x, "ionic_probability") for x in ionic[:3 if template_index < 10 else 2]]
        if template_index >= 10:
            options.append((periodic[0], "periodic"))
        for option, method in options:
            candidates.append(_candidate(record, structure, option, method, len(candidates)))
        selected.append(record["id"])
        template_details.append({"id": record["id"], "group": record["group"],
                                 "n_sites": len(structure),
                                 "oxidation_guess": oxidation,
                                 "n_ionic_options": len(ionic),
                                 "n_periodic_options_excluding_first_two_ionic": len(periodic)})
        if len(selected) == 100:
            break
    counters["selected_templates"] = len(selected)
    counters["replacement_not_evaluated_after_fixed_100_selected"] = len(eligible) - counters["replacement_evaluated"]
    if len(selected) != 100 or len(candidates) != 300:
        raise RuntimeError(f"Insufficient eligible TRAIN templates: {dict(counters)}")
    counts = Counter(c["substitution_method"] for c in candidates)
    if counts != Counter({"ionic_probability": 210, "periodic": 90}):
        raise AssertionError(f"Unexpected generation mixture: {counts}")
    if len(set(selected)) != 100 or any(c["template_group"] in holdout_groups for c in candidates):
        raise AssertionError("Template independence or TRAIN-only boundary violated")
    config = {
        "seed": SEED, "n_templates": 100, "n_candidates": 300,
        "method_counts": dict(counts), "selected_template_ids": selected,
        "selected_templates": template_details, "eligibility_counts": dict(counters),
        "candidate_sha256": _json_hash(candidates),
        "selected_template_ids_sha256": _json_hash(selected),
        "p1_pilot_path": str(p1_data.PILOT_PATH), "p1_pilot_sha256": pilot_hash,
        "selection": "Ascending SHA256(seed|template|id) among ordered TRAIN structures <=20 sites with MP references/radii, >=3 ionic options, and periodic option distinct from first2 ionic. Stop at100; no energy label or novelty access.",
        "replacement_selection": "Within template, ascending independent SHA256(seed|ionic-substitution OR periodic-substitution|id|source|target); first10 templates use3ionic, remaining90 use2ionic+1periodic. No candidate composition duplicates within a template.",
        "ionic_threshold": IONIC_THRESHOLD,
        "ionic_probability_definition": "SubstitutionProbability.cond_prob(target_ion, source_ion) = p(target|source), same integral nonzero oxidation state; target absent from template.",
        "oxidation_inference": "First ranked neutral composition.oxi_state_guesses(all_oxi_states=False,max_sites=-20); inferred from composition alone; integral nonzero source required. Size bound limits reduced composition to20 atoms; no all-state search or DFT/site-property use.",
        "periodic_criterion": "target.group == source.group OR abs(target.row-source.row)==1 (pymatgen Element IUPAC group/period), target absent, available MP reference and covalent radius; adjacent-period branch has no extra group restriction.",
        "lattice_prescale": "Isotropic lattice-length multiplier = sum_i(n_i*r_new_i)/sum_i(n_i*r_old_i); volume multiplied by its cube. Every site of source element replaced; coordinates fixed before relaxation.",
        "novelty_prefilter": False,
        "novelty_policy": "All300 relax first; full132752 Matbench reduced-composition screening and synthetic StructureMatcher deduplication occur after relaxation/physical filters.",
        "label_boundary": "No HOLDOUT label read by selection; TRAIN DFT values returned separately solely for descriptive comparison, never template/substitution selection.",
        "candidate_metadata": [{k: v for k, v in c.items() if k != "initial_structure"} for c in candidates],
        "provenance": _provenance(),
    }
    _say(log, f"H1 frozen 100 TRAIN templates /300 candidates: ionic210,periodic90; evaluated {counters['replacement_evaluated']}/{len(eligible)} eligible; SHA256={config['candidate_sha256']}")
    return config, candidates, train_summary


def full_composition_set(log=None):
    """Read all 132752 source rows streaming; return only unique composition keys.

    The dataset contains repeated stoichiometries, so the set need not contain
    132752 elements. No source targets are retained or used for selection.
    """
    if not p1_data.RAW_PATH.is_file():
        raise FileNotFoundError(p1_data.RAW_PATH)
    source_hash = _digest(p1_data.RAW_PATH)
    if source_hash != p1_data.EXPECTED_SHA256:
        raise ValueError("Full Matbench source SHA256 mismatch")
    keys, count = set(), 0
    for _, raw_row in p1_data._iter_raw_rows(p1_data.RAW_PATH):
        keys.add(p1_data.reduced_composition(raw_row[0]))
        count += 1
        if count % 25000 == 0:
            _say(log, f"Full Matbench composition inventory: {count}/{p1_data.N_FULL} rows")
    if count != 132752:
        raise ValueError(f"Incomplete novelty inventory: {count}")
    _say(log, f"Full Matbench novelty inventory verified: {count} rows, {len(keys)} unique reduced compositions; source SHA256={source_hash}")
    return keys


if __name__ == "__main__":
    cfg, rows, summary = prepare_candidates(print)
    assert len(rows) == 300 and len(summary) == 8000
    assert Counter(r["template_id"] for r in rows) == Counter({k: 3 for k in cfg["selected_template_ids"]})
    for row in rows:
        structure = clean_structure(row["initial_structure"])
        assert structure.is_ordered and len(structure) <= 20
        assert not any(site.properties for site in structure)
        assert structure.charge == 0
        assert row["substitution_to"] not in {k.split(":")[0] for k in row["template_group"].split("|")}
    print(json.dumps({"self_check": "passed", "candidates": len(rows),
                      "templates": cfg["n_templates"], "method_counts": cfg["method_counts"],
                      "eligible": cfg["eligibility_counts"],
                      "candidate_sha256": cfg["candidate_sha256"]}, sort_keys=True))
