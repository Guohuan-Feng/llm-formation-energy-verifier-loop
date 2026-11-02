"""Stage4.2-only label-free metadata, POOL oracle and independent L0 TEST gate.

All earlier stage files remain read-only. The only preparation write is the
explicitly authorized POOL-final_structure space-group metadata file. Import,
load_bundle and configuration never reveal a target or execute an LLM/model.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from copy import deepcopy
import gzip
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

import h4_data as old

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "work/h-data/H4"
PHASE = DATA / "phase4_2"
SPACEGROUPS = PHASE / "metadata/spacegroups.json"
PRIOR_MARKER = DATA / "phase4_1/evaluation_started.json"
PARAMETERS = {"symprec": 0.01, "angle_tolerance": 5}
METADATA_VERSION = "H4-stage4.2-POOL-final-structure-spacegroups-v1"
_BUNDLE = None
_REVEALED = {}
_TRUTH_READ = False

sha = old.sha
json_hash = old.json_hash


def checked(path, *, writable=False, must_exist=True):
    path = Path(path).resolve(strict=must_exist)
    limit = PHASE.resolve() if writable else ROOT.resolve()
    if "onedrive" in str(path).casefold() or not path.is_relative_to(limit):
        raise ValueError("Stage4.2 path escapes its non-OneDrive permitted root")
    return path


def _read(path):
    return json.loads(checked(path).read_text(encoding="utf-8"))


def _metadata_payload_hash(payload):
    return json_hash({k: v for k, v in payload.items() if k != "content_sha256"})


def _validate_spacegroups(payload, bundle, source):
    if (payload.get("version") != METADATA_VERSION or payload.get("parameters") != PARAMETERS
            or payload.get("source_path") != str(source)
            or payload.get("source_sha256") != bundle["source_hashes"][str(source)]
            or payload.get("pool_ids_sha256") != json_hash(bundle["pool_ids"])
            or payload.get("content_sha256") != _metadata_payload_hash(payload)):
        raise ValueError("Frozen stage4.2 space-group metadata identity changed")
    records = payload.get("records")
    if not isinstance(records, list) or [r.get("id") for r in records] != bundle["pool_ids"]:
        raise ValueError("Space-group metadata must cover all frozen POOL5000 in order")
    for row in records:
        if (set(row) != {"id", "number", "symbol", "final_structure_sha256"}
                or type(row["number"]) is not int or not 1 <= row["number"] <= 230
                or type(row["symbol"]) is not str or not row["symbol"]
                or type(row["final_structure_sha256"]) is not str or len(row["final_structure_sha256"]) != 64):
            raise ValueError("Malformed POOL space-group record")
    if payload.get("versions") != {"pymatgen": version("pymatgen"), "spglib": version("spglib")}:
        raise ValueError("Space-group package version changed since metadata preparation")
    return payload


def prepare_spacegroups(*, bundle=None, progress=None):
    """Authorized CPU-only symmetry analysis of exactly5000 POOL final_structure.

    No P1/template geometry, initial geometry fallback, target or generated
    descriptor is used. Existing metadata is checked and never overwritten.
    """
    b = bundle if bundle is not None else old.load_bundle()
    source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
    if sha(source) != b["source_hashes"][str(source)]:
        raise ValueError("Frozen H2 POOL geometry source bytes changed")
    if SPACEGROUPS.exists():
        return _validate_spacegroups(_read(SPACEGROUPS), b, source)
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        mixed = json.load(stream)
    pool = set(b["pool_ids"])
    results = {}
    for item in mixed:
        row = item["row"]
        rid = row["id"]
        if rid not in pool:
            continue
        if row["split"] != "POOL" or rid in results:
            raise ValueError("POOL geometry identity changed")
        raw = item["final_structure"]
        if not isinstance(raw, dict):
            raise ValueError("POOL final_structure is missing; no initial/template fallback is permitted")
        analyzer = SpacegroupAnalyzer(Structure.from_dict(raw), **PARAMETERS)
        number = int(analyzer.get_space_group_number())
        symbol = analyzer.get_space_group_symbol()
        if not 1 <= number <= 230 or not symbol:
            raise ValueError("POOL symmetry analysis did not yield a valid space group")
        results[rid] = {"id": rid, "number": number, "symbol": symbol,
            "final_structure_sha256": json_hash(raw)}
        if progress is not None and len(results) % 100 == 0:
            progress({"phase": "POOL_spacegroup_metadata", "complete": len(results), "total": 5000})
    if set(results) != pool:
        raise ValueError("POOL final-structure symmetry coverage incomplete")
    payload = {"version": METADATA_VERSION, "stage": "4.2", "scope": "POOL5000 final_structure only",
        "parameters": dict(PARAMETERS), "versions": {"pymatgen": version("pymatgen"), "spglib": version("spglib")},
        "source_path": str(source), "source_sha256": b["source_hashes"][str(source)],
        "pool_ids_sha256": json_hash(b["pool_ids"]), "n_records": 5000,
        "target_fields_accessed": False, "P1_or_initial_structure_substitution": False,
        "query_field_name": "template_spacegroup",
        "query_field_semantics": "User-designated name; computed on each POOL sample final_structure, not its original template",
        "records": [results[rid] for rid in b["pool_ids"]]}
    payload["content_sha256"] = _metadata_payload_hash(payload)
    _validate_spacegroups(payload, b, source)
    destination = checked(SPACEGROUPS, writable=True, must_exist=False)
    checked(destination.parent, writable=True, must_exist=False).mkdir(parents=True, exist_ok=True)
    with checked(destination, writable=True, must_exist=False).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        __import__("os").fsync(stream.fileno())
    return payload


def _hash_group(seed, group):
    return (hashlib.sha256(f"H3-CGCNN-inner-v1|{seed}|{group}".encode()).hexdigest(), group)


def load_bundle():
    """Extend the old label-free cache, never its random-O target oracle."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    b = old.load_bundle()
    source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
    metadata = _validate_spacegroups(_read(SPACEGROUPS), b, source)
    prior_config = PHASE / "stage4_1_public_snapshot/config.json"
    if not prior_config.exists():
        prior_config = ROOT / "H4_feature_loop/config.json"
    previous = _read(prior_config)
    if previous.get("stage") != "4.1" or previous["stage_boundary"]["later_stages_prepared"] is not False:
        raise ValueError("Prior stage4.1 freeze identity differs")
    # Metadata dictionaries are copied so stage4.1's in-memory bundle is untouched.
    result = {key: value for key, value in b.items() if key not in ("perseed", "by_id", "records", "evaluation_truth_loaded")}
    result["by_id"] = {rid: dict(record) for rid, record in b["by_id"].items()}
    result["records"] = [result["by_id"][rid] for rid in b["ids"]]
    for row in metadata["records"]:
        result["by_id"][row["id"]].update(template_spacegroup=row["number"], spacegroup_symbol=row["symbol"])
    groups = {b["groups"][rid] for rid in b["pool_ids"]}
    perseed = {}
    for seed in (0, 1, 2):
        frozen = previous["data"]["perseed"][str(seed)]
        fit_groups = {b["groups"][rid] for rid in frozen["fit_ids"]}
        val_groups = {b["groups"][rid] for rid in frozen["val_ids"]}
        if not fit_groups or not val_groups or fit_groups & val_groups:
            raise ValueError("Prior stage4.1 template split is invalid")
        threshold = max(_hash_group(seed, group) for group in val_groups)
        mapping = {group: "val" if _hash_group(seed, group) <= threshold else "fit" for group in sorted(groups)}
        if any(mapping[group] != "fit" for group in fit_groups) or any(mapping[group] != "val" for group in val_groups):
            raise ValueError("Extended split changed an existing stage4.1 group")
        perseed[seed] = {"S0_ids": list(frozen["S0_ids"]), "fit_group_mapping": mapping,
            "cutoff": {"sha256": threshold[0], "group": threshold[1]},
            "fit_pool_ids": [rid for rid in b["pool_ids"] if mapping[b["groups"][rid]] == "fit"],
            "val_pool_ids": [rid for rid in b["pool_ids"] if mapping[b["groups"][rid]] == "val"],
            "old_groups_unchanged": True}
    result.update(perseed=perseed, evaluation_truth_loaded=False,
        spacegroup_metadata=metadata, spacegroup_metadata_path=str(checked(SPACEGROUPS)),
        prior_stage4_1_marker_sha256=sha(checked(PRIOR_MARKER)),
        prior_stage4_1_config_path=str(checked(prior_config)))
    extra = {str(checked(SPACEGROUPS)): sha(SPACEGROUPS),
        str(checked(prior_config)): sha(prior_config),
        str(checked(PRIOR_MARKER)): sha(PRIOR_MARKER)}
    result["source_hashes"] = {**b["source_hashes"], **extra}
    result["input_hashes"] = dict(result["source_hashes"])
    _BUNDLE = result
    return result


def _bundle(bundle):
    return bundle if bundle is not None else load_bundle()


def validate_labels(labels, seed, *, bundle=None, require_n=None):
    b = _bundle(bundle)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Seed must be0/1/2")
    if (not isinstance(labels, dict) or not 500 <= len(labels) <= 2000
            or not set(b["perseed"][seed]["S0_ids"]) <= set(labels) <= set(b["pool_ids"])):
        raise ValueError("L0 labels must be this seed's S0500 plus at most1500 revealed POOL labels")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in labels.values()):
        raise ValueError("L0 labels must be finite scalars")
    if require_n is not None and len(labels) != require_n:
        raise ValueError("L0 label budget cardinality differs")
    return True


def reveal(ids, seed, *, bundle=None):
    """Access only explicitly selected POOL labels, preserving a per-seed2000 cap."""
    b = _bundle(bundle)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("Seed must be0/1/2")
    ids = list(ids)
    if len(set(ids)) != len(ids) or not set(ids) <= set(b["pool_ids"]):
        raise ValueError("Reveal must name distinct POOL members only")
    observed = _REVEALED.setdefault(seed, {})
    authorized = set(observed) | set(ids)
    if len(authorized) > 2000 or not set(b["perseed"][seed]["S0_ids"]) <= authorized:
        raise ValueError("Reveal must initialize S0500 and remain within additional1500 budget")
    missing = set(ids) - set(observed)
    if missing:
        source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
        if sha(source) != b["source_hashes"][str(source)]:
            raise ValueError("Frozen H2 POOL label bytes changed")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            mixed = json.load(stream)
        fresh = {}
        for item in mixed:
            row = item["row"]
            rid = row["id"]
            if rid not in missing:
                continue
            if row["split"] != "POOL" or rid in fresh:
                raise ValueError("Selected POOL label identity differs")
            value = row["formation_energy_eV_per_atom"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Selected POOL label must be finite")
            fresh[rid] = float(value)
        if set(fresh) != missing:
            raise ValueError("Selected POOL label coverage incomplete")
        observed.update(fresh)
    return {rid: observed[rid] for rid in ids}


def initial_labels(seed, *, bundle=None):
    b = _bundle(bundle)
    return reveal(b["perseed"][seed]["S0_ids"], seed, bundle=b)


def score_truth(ids, *, evaluation_marker, bundle=None):
    """Independent L0 terminal TEST read; never rescoring old O predictions."""
    global _TRUTH_READ
    b = _bundle(bundle)
    ids = list(ids)
    if ids != b["test_ids"]:
        raise ValueError("L0 terminal scoring requires all frozen TEST1000 in order")
    marker_path = checked(evaluation_marker)
    if marker_path != checked(PHASE / "evaluation_started.json"):
        raise ValueError("Only the stage4.2 final-evaluation gate is accepted")
    marker = _read(marker_path)
    required = {"stage": "4.2", "condition": "L0", "scoring_pass": 1,
        "current_L0_scoring_pass": 1, "prior_H4_TEST_scoring_passes": 1,
        "TEST_targets_previously_read_for_this_condition": False,
        "prior_stage4_1_marker_sha256": sha(checked(PRIOR_MARKER))}
    if any(type(marker.get(k)) is not type(v) or marker.get(k) != v for k, v in required.items()):
        raise ValueError("Stage4.2 terminal marker identity or scoring-pass disclosure differs")
    if _TRUTH_READ or b.get("evaluation_truth_loaded"):
        raise RuntimeError("L0 TEST truth has already been read in this process")
    source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
    if sha(source) != b["source_hashes"][str(source)]:
        raise ValueError("Frozen TEST source bytes changed")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        mixed = json.load(stream)
    authorized = set(ids)
    truth = {}
    _TRUTH_READ = True
    b["evaluation_truth_loaded"] = True
    for item in mixed:
        row = item["row"]
        rid = row["id"]
        if rid not in authorized:
            continue
        if row["split"] != "TEST" or rid in truth:
            raise ValueError("TEST target identity changed")
        value = row["formation_energy_eV_per_atom"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("TEST target must be finite")
        truth[rid] = float(value)
    if set(truth) != authorized:
        raise ValueError("L0 TEST truth coverage incomplete")
    return {rid: truth[rid] for rid in ids}, {**required,
        "TEST_targets_read": 1000, "prior_condition": "O", "current_condition": "L0",
        "source_path": str(source), "source_sha256": b["source_hashes"][str(source)],
        "evaluation_marker": str(marker_path), "evaluation_marker_sha256": sha(marker_path),
        "test_ids_sha256": json_hash(ids), "HOLDOUT_target_fields_accessed": False}


def configuration(*, bundle=None):
    b = _bundle(bundle)
    seeds = {str(seed): deepcopy(b["perseed"][seed]) for seed in (0, 1, 2)}
    for item in seeds.values():
        item["sha256"] = {k: json_hash(v) for k, v in item.items()}
    return {"data_protocol": "H4-stage4.2-L0-frozen-inputs-v1", "stage": "4.2", "condition": "L0",
        "S0_size": 500, "additional_label_budget": 1500, "rounds": 20, "per_round_quota": 75,
        "perseed": seeds, "pool_ids": list(b["pool_ids"]), "test_ids": list(b["test_ids"]),
        "pool_ids_sha256": json_hash(b["pool_ids"]), "test_ids_sha256": json_hash(b["test_ids"]),
        "template_groups_sha256": json_hash(b["groups"]),
        "stable_split_method": "Freeze stage4.1 max(validation SHA256(H3-CGCNN-inner-v1|seed|template_id),template_id) cutoff; every POOL group <=cutoff is val, others fit; old assignments unchanged",
        "spacegroup_metadata_path": b["spacegroup_metadata_path"],
        "spacegroup_metadata_sha256": sha(b["spacegroup_metadata_path"]),
        "spacegroup_parameters": dict(PARAMETERS), "spacegroup_versions": dict(b["spacegroup_metadata"]["versions"]),
        "spacegroup_input": "Each POOL sample's final_structure, explicitly user-authorized; not original template/P1 geometry",
        "source_hashes": dict(b["source_hashes"]), "field_spec_sha256": b["field_spec_sha256"],
        "POOL_label_access": "S0500 plus this L0 seed's actually selected IDs only; never stage4.1 O2000 wholesale",
        "TEST_scoring": "One new L0 terminal pass after its own stage4.2 gate; old O pass disclosed and untouched",
        "prior_stage4_1_marker_sha256": b["prior_stage4_1_marker_sha256"],
        "prior_stage4_1_config_path": b["prior_stage4_1_config_path"]}
