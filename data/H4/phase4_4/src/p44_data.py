"""H4.4 frozen metadata, original-DFT geometry and separately gated label access.

Import and prepare_definitions never execute features, models or target access.
Geometry generation is explicit, after freeze, and writes only phase4_4.
Never call the old H3 load_h2_verified/strict_label_status convenience loaders:
those inspect held-out CHGNet label fields while checking historical coverage.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATA = ROOT / "work/h-data/H4"
PHASE = DATA / "phase4_4"
H2 = ROOT / "work/h-data/H2"
H3 = ROOT / "work/h-data/H3"
SEEDS = (0, 1, 2)
VERSION = "H4.4-data-originalDFT-and-separated-target-gates-v1"
REAL_CACHE = PHASE / "representations/real_original_DFT_structures.jsonl.gz"
REAL_META = PHASE / "representations/real_original_DFT_metadata.json"
_TRUTH_STARTED = {"synthetic": False, "real": False}


def safe(path, *, write=False, exists=False):
    path = Path(path).resolve(strict=exists)
    allowed = PHASE.resolve() if write else ROOT.resolve()
    if "onedrive" in str(path).casefold() or not path.is_relative_to(allowed):
        raise ValueError("H4.4 path escapes its non-OneDrive permitted root")
    return path


def sha(path):
    digest = hashlib.sha256()
    with safe(path, exists=True).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    return json.loads(safe(path, exists=True).read_text(encoding="utf-8-sig"))


def _sealed(path):
    value = read(path)
    body = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("record_sha256") != json_hash(body): raise ValueError("Old scientific metadata self-hash changed")
    return value


def _write(path, value):
    destination = safe(path, write=True)
    safe(destination.parent, write=True).mkdir(parents=True, exist_ok=True)
    temporary = safe(destination.with_name(destination.name + ".tmp"), write=True)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with safe(temporary, write=True).open("xb") as stream:
        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    os.replace(safe(temporary, write=True), safe(destination, write=True))


def _feature_sets():
    output, inputs = {}, {}
    for condition, phase in (("O", "phase4_1"), ("L0", "phase4_2"), ("L1", "phase4_3")):
        selected_path = DATA / phase / ("selected_candidates.json" if condition == "O" else "selected_rounds.json")
        selected = read(selected_path)
        if condition == "O": selected = selected["selected"]
        values = list(selected.values()) if isinstance(selected, dict) else selected
        output[condition] = {}
        inputs[str(safe(selected_path))] = sha(selected_path)
        for seed in SEEDS:
            matches = [x for x in values if x["seed"] == seed]
            if len(matches) != 1: raise ValueError("Old selected-feature seed coverage changed")
            item = matches[0]
            selected_metadata = _sealed(item["metadata_path"])
            if selected_metadata["record_sha256"] != item["record_sha256"]:
                raise ValueError("Selected source record differs from saved selection")
            final_path = DATA / phase / "final" / f"{condition}_seed{seed}" / "final_predictions.json"
            final = _sealed(final_path)
            key = "selected_candidate_record_sha256" if condition == "O" else "selected_round_record_sha256"
            if final[key] != item["record_sha256"]: raise ValueError("Final model does not bind the saved selected source")
            kept = final["full_train_features"]["kept_names"]
            features = [deepcopy(f) for f in item["features"] if f["name"] in set(kept)]
            if [f["name"] for f in features] != kept: raise ValueError("Final kept source/name/order changed")
            for feature in features:
                if set(feature) != {"name", "description", "source"} or not all(type(v) is str for v in feature.values()):
                    raise ValueError("Unexpected old feature format")
            artifacts = {"selection_document": {"path": str(safe(selected_path)), "sha256": sha(selected_path)},
                "selected_metadata": {"path": str(safe(item["metadata_path"])), "sha256": sha(item["metadata_path"])},
                "final_metadata": {"path": str(safe(final_path)), "sha256": sha(final_path)}}
            for artifact in artifacts.values(): inputs[artifact["path"]] = artifact["sha256"]
            output[condition][str(seed)] = {"features": features, "features_sha256": json_hash(features),
                "selected_metadata_path": str(safe(item["metadata_path"])), "selected_record_sha256": item["record_sha256"],
                "final_metadata_path": str(safe(final_path)), "final_record_sha256": final["record_sha256"],
                "selected_index": item.get("candidate_index", item.get("round_index")), "source_artifacts": artifacts,
                "selection_rule": "Exact original final fullTRAIN kept_names and original complete source; no synthetic reselection"}
    return output, inputs


def prepare_definitions():
    """Metadata only: IDs, saved sources, group mapping, hashes; no label fields."""
    import numpy as np
    manifest_path = H2 / "split_manifest.json"
    manifest = read(manifest_path)
    base_path = H3 / "base_features.npz"
    with np.load(safe(base_path, exists=True), allow_pickle=False) as cache: ids = cache["ids"].tolist()
    if len(ids) != 10000 or len(set(ids)) != 10000: raise ValueError("Frozen H3 base ID universe changed")
    splits = manifest["split_ids"]
    pool, test = list(splits["synthetic_POOL"]), list(splits["synthetic_TEST"])
    train, holdout = list(splits["real_TRAIN"]), list(splits["real_HOLDOUT"])
    if [len(x) for x in (pool, test, train, holdout)] != [5000, 1000, 2000, 2000]: raise ValueError("Frozen split sizes changed")
    if set(ids) != set(pool + test + train + holdout) or len(set(pool + test + train + holdout)) != 10000:
        raise ValueError("Frozen data splits overlap or miss cached IDs")
    group_path = ROOT / "work/h-data/H3b/cgcnn/plan.json"
    groups = read(group_path)["groups"]
    synthetic_groups = {rid: str(groups[rid]) for rid in pool + test}
    real_groups = {rid: str(groups[rid]) for rid in train + holdout}
    if {real_groups[r] for r in train} & {real_groups[r] for r in holdout}: raise ValueError("Real composition groups overlap")
    if {synthetic_groups[r] for r in pool} & {synthetic_groups[r] for r in test}: raise ValueError("Synthetic template groups overlap")
    feature_sets, inputs = _feature_sets()
    paths = [manifest_path, base_path, H3 / "label_free_structures.jsonl.gz", H3 / "representation_metadata.json", group_path,
        H2 / "synthetic_records_private.json.gz", H2 / "real_records_private.json.gz", ROOT / "work/p1-data/p1_10000.json.gz",
        ROOT / "h3_data.py", ROOT / "h1_generation.py", ROOT / "p1_data.py", ROOT / "h2_generation.py", Path(__file__)]
    inputs.update({str(safe(path)): sha(path) for path in paths})
    O = {str(seed): list(manifest["h2_rf_synthetic_training_ids_by_seed"][str(seed)]) for seed in SEEDS}
    S0 = {str(seed): list(manifest["S0"][str(seed)]) for seed in SEEDS}
    for seed in SEEDS:
        if len(O[str(seed)]) != 2000 or len(set(O[str(seed)])) != 2000 or len(S0[str(seed)]) != 500 or not set(S0[str(seed)]) <= set(O[str(seed)]) <= set(pool):
            raise ValueError("Frozen O2000/S0500 membership changed")
    return {"schema_version": VERSION, "seeds": list(SEEDS), "ids": ids, "synthetic_pool_ids": pool, "synthetic_test_ids": test,
        "real_train_ids": train, "real_holdout_ids": holdout, "perseed_O_ids": O, "perseed_S0_ids": S0,
        "synthetic_groups": synthetic_groups, "real_groups": real_groups, "group_source": str(safe(group_path)),
        "real_group_rule": "p1_data.reduced_composition: occupancy Fractions limit_denominator1000000, denominatorLCM, countGCD, sorted element:integer ratio",
        "frozen_feature_sets": feature_sets, "source_hashes": inputs,
        "real_representation": {"policy": "H2 initial_structure from cleaned original P1 DFT structure; never H3 CHGNet-relaxed geometry",
            "new_cache": str(safe(REAL_CACHE)), "new_metadata": str(safe(REAL_META)),
            "conversion": "Frozen h3_data.structure_to_dict({'final_structure': original_H2_initial_structure}); no CHGNet/model call",
            "Magpie_policy": "Reuse H3 composition-only Magpie132 only after original-vs-cached element/fraction equality check"},
        "target_access": {"TRAIN": "Separate explicit POOL5000 or realDFT TRAIN2000 accessors only",
            "synthetic_gate": str(safe(PHASE / "evaluation_started.json")), "real_gate": str(safe(PHASE / "holdout_evaluation_started.json")),
            "historical_HOLDOUT_field_access_count_known": False, "known_formal_prior_stage_evidence": ["H2", "H3", "H3b"],
            "gate_count_is_not_field_access_count": True}}


def _definitions(cfg):
    value = cfg.get("data", cfg)
    if value.get("schema_version") != VERSION: raise ValueError("Use the frozen H4.4 data definitions")
    return value


def _verify_frozen(cfg):
    disk = read(PHASE / "config.json"); frozen = read(PHASE / "frozen_config.json")
    if disk != cfg or frozen.get("sha256") != sha(PHASE / "config.json"):
        raise ValueError("H4.4 config must be frozen before geometry/labels are requested")
    definitions = _definitions(cfg)
    for path, expected in definitions["source_hashes"].items():
        if sha(path) != expected: raise ValueError("Frozen H4.4 source/input changed: " + Path(path).name)
    return definitions


def _private_records(filename):
    # Callers select allowed IDs/split BEFORE looking up any target field.
    with gzip.open(safe(H2 / filename, exists=True), "rt", encoding="utf-8") as stream: return json.load(stream)


def _old_structures(wanted):
    values = {}
    with gzip.open(safe(H3 / "label_free_structures.jsonl.gz", exists=True), "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in wanted:
                if row["id"] in values: raise ValueError("Duplicate H3 label-free cache identity")
                values[row["id"]] = row["s"]
    if set(values) != set(wanted): raise ValueError("H3 label-free cache coverage changed")
    return values


def _prepare_original_real(definitions, progress=None):
    if REAL_CACHE.exists() or REAL_META.exists() or REAL_CACHE.with_name(REAL_CACHE.name + ".tmp").exists():
        raise FileExistsError("An original-DFT cache or partial attempt exists; no automatic regeneration")
    import importlib.util
    spec = importlib.util.spec_from_file_location("h4p44_frozen_geometry_contract", safe(ROOT / "h3_data.py", exists=True))
    conversion = importlib.util.module_from_spec(spec); spec.loader.exec_module(conversion)
    wanted = definitions["real_train_ids"] + definitions["real_holdout_ids"]
    old = _old_structures(set(wanted))
    raw = {}
    for record in _private_records("real_records_private.json.gz"):
        row = record["row"]; rid = row["id"]
        if rid not in set(wanted): continue
        expected_split = "TRAIN" if rid in set(definitions["real_train_ids"]) else "HOLDOUT"
        if row["split"] != expected_split or str(row["reduced_composition"]) != definitions["real_groups"][rid]: raise ValueError("Original real identity/group changed")
        if record.get("initial_structure") is None or rid in raw: raise ValueError("Missing/duplicate original DFT structure")
        raw[rid] = {"structure": record["initial_structure"], "original_id": row["original_id"]}
    if set(raw) != set(wanted): raise ValueError("Original real4000 geometry coverage changed")
    safe(REAL_CACHE.parent, write=True).mkdir(parents=True, exist_ok=True)
    temporary = safe(REAL_CACHE.with_name(REAL_CACHE.name + ".tmp"), write=True)
    initial_hashes = {}; s_hashes = {}
    with safe(temporary, write=True).open("xb") as destination:
        with gzip.GzipFile(fileobj=destination, mode="wb", filename="", mtime=0) as compressed:
            for index, rid in enumerate(wanted, 1):
                s = conversion.structure_to_dict({"final_structure": raw[rid]["structure"]})
                prior = old[rid]["composition"]; current = s["composition"]
                if set(prior) != set(current) or any(not math.isclose(float(prior[e]), float(current[e]), rel_tol=0, abs_tol=1e-12) for e in current):
                    raise ValueError("Original DFT composition differs from cached Magpie composition")
                initial_hashes[rid] = json_hash(raw[rid]["structure"]); s_hashes[rid] = json_hash(s)
                compressed.write((json.dumps({"id": rid, "s": s}, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                if index % 100 == 0:
                    _write(PHASE / "representations/geometry_progress.json", {"completed": index, "total": 4000, "target_fields_accessed": False})
                    if progress: progress({"phase": "original_DFT_geometry", "completed": index, "total": 4000})
    os.replace(safe(temporary, write=True), safe(REAL_CACHE, write=True))
    _write(REAL_META, {"status": "complete", "schema_version": VERSION, "n_structures": 4000, "ids": wanted,
        "cache_path": str(safe(REAL_CACHE)), "cache_sha256": sha(REAL_CACHE), "source_hashes": definitions["source_hashes"],
        "initial_structure_sha256": initial_hashes, "whitelist_s_sha256": s_hashes, "Magpie_composition_check": "PASS",
        "representation": definitions["real_representation"], "target_fields_accessed": False, "CHGNet_calls": 0})


def load_bundle(cfg, prepare_real_geometry=False, progress=None):
    definitions = _verify_frozen(cfg)
    if not REAL_CACHE.exists() or not REAL_META.exists():
        if not prepare_real_geometry: raise FileNotFoundError("Original-DFT real geometry cache is not materialized; no relaxed fallback")
        _prepare_original_real(definitions, progress=progress)
    metadata = read(REAL_META)
    if (metadata.get("status") != "complete" or metadata.get("source_hashes") != definitions["source_hashes"]
            or metadata.get("cache_sha256") != sha(REAL_CACHE) or metadata.get("target_fields_accessed") is not False):
        raise ValueError("Owned original-DFT geometry cache binding changed")
    import numpy as np
    with np.load(safe(H3 / "base_features.npz", exists=True), allow_pickle=False) as cache:
        ids = cache["ids"].tolist(); matrix = cache["magpie132"].copy()
    if ids != definitions["ids"] or matrix.shape != (10000, 132): raise ValueError("Cached Magpie ID/order/shape changed")
    synthetic = set(definitions["synthetic_pool_ids"] + definitions["synthetic_test_ids"])
    structures = _old_structures(synthetic)
    with gzip.open(safe(REAL_CACHE, exists=True), "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line); rid = record["id"]
            if rid in structures or rid not in definitions["real_groups"] or json_hash(record["s"]) != metadata["whitelist_s_sha256"][rid]:
                raise ValueError("Owned original-DFT s identity/hash changed")
            structures[rid] = record["s"]
    if set(structures) != set(ids): raise ValueError("Expected all10000 label-free structures")
    sources = {**definitions["source_hashes"], str(safe(REAL_CACHE)): sha(REAL_CACHE), str(safe(REAL_META)): sha(REAL_META)}
    return {"ids": ids, "index": {rid: i for i, rid in enumerate(ids)}, "magpie132": matrix, "structures_by_id": structures,
        **{key: deepcopy(definitions[key]) for key in ("synthetic_pool_ids", "synthetic_test_ids", "real_train_ids", "real_holdout_ids", "synthetic_groups", "real_groups", "perseed_O_ids", "perseed_S0_ids")},
        "feature_sets": deepcopy(definitions["frozen_feature_sets"]), "source_hashes": sources, "definitions": definitions,
        "evaluation_truth_loaded": False, "real_geometry": "original P1 DFT via H2 initial_structure", "new_target_fields_in_bundle": False}


def _training(cfg, bundle, domain):
    definitions = _verify_frozen(cfg)
    ids = definitions["synthetic_pool_ids"] if domain == "synthetic" else definitions["real_train_ids"]
    expected = set(ids); output = {}; split = "POOL" if domain == "synthetic" else "TRAIN"
    filename = "synthetic_records_private.json.gz" if domain == "synthetic" else "real_records_private.json.gz"
    field = "formation_energy_eV_per_atom" if domain == "synthetic" else "dft_energy_eV_per_atom"
    for record in _private_records(filename):
        row = record["row"]; rid = row["id"]
        if rid not in expected: continue
        if row["split"] != split or rid in output: raise ValueError("TRAIN identity/split changed")
        value = row[field]  # First and only target-field lookup, after TRAIN-only membership gate.
        if type(value) not in (int, float) or not math.isfinite(value): raise ValueError("TRAIN target must be finite")
        output[rid] = float(value)
    if set(output) != expected: raise ValueError("TRAIN target coverage changed")
    return {rid: output[rid] for rid in ids}


def synthetic_training_targets(cfg, bundle): return _training(cfg, bundle, "synthetic")
def real_training_targets(cfg, bundle): return _training(cfg, bundle, "real")


def _truth(ids, marker, bundle, domain):
    definitions = bundle["definitions"]
    cfg = read(PHASE / "config.json")
    if _verify_frozen(cfg) != definitions: raise ValueError("Truth bundle differs from frozen data definitions")
    frozen_ids = definitions["synthetic_test_ids"] if domain == "synthetic" else definitions["real_holdout_ids"]
    if list(ids) != frozen_ids: raise ValueError("Terminal truth requires the complete ordered frozen domain")
    expected_path = PHASE / ("evaluation_started.json" if domain == "synthetic" else "holdout_evaluation_started.json")
    if safe(marker, exists=True) != safe(expected_path, exists=True): raise ValueError("Wrong canonical terminal gate")
    m = read(marker)
    if m.get("config_sha256") != sha(PHASE / "config.json"):
        raise ValueError("Terminal gate must bind the frozen H4.4 config")
    required = {"stage": "4.4", "domain": domain, "condition": "controls" if domain == "synthetic" else "transfer", "scoring_pass": 1,
        "targets_previously_read_for_this_phase_domain": False}
    if domain == "synthetic": required.update(prior_H4_synthetic_TEST_scoring_passes=3, current_H4_synthetic_TEST_scoring_pass=4)
    else: required.update(current_H4_real_transfer_HOLDOUT_scoring_pass=1, historical_HOLDOUT_pass_count_known=False)
    if any(type(m.get(k)) is not type(v) or m.get(k) != v for k, v in required.items()): raise ValueError("Terminal gate constants/types differ")
    if type(m.get("owner_pid")) is not int or m["owner_pid"] != os.getpid(): raise ValueError("Truth must be consumed by the live gate owner")
    if domain == "real":
        dependencies = {"synthetic_evaluation_marker_sha256": PHASE / "evaluation_started.json", "synthetic_terminal_sha256": PHASE / "synthetic_terminal_results.json", "synthetic_pairing_cache_sha256": PHASE / "synthetic_pairing_cache.npz"}
        if any(m.get(key) != sha(path) for key, path in dependencies.items()): raise ValueError("Real gate must follow the committed synthetic pass")
    if _TRUTH_STARTED[domain]: raise RuntimeError("This domain's target accessor has already started; no reread/retry")
    filename = "synthetic_records_private.json.gz" if domain == "synthetic" else "real_records_private.json.gz"
    source = H2 / filename
    if sha(source) != definitions["source_hashes"][str(safe(source))]: raise ValueError("Frozen private target source changed")
    _TRUTH_STARTED[domain] = True  # Failure after the first read also forbids retry.
    expected = set(frozen_ids); output = {}; split = "TEST" if domain == "synthetic" else "HOLDOUT"
    field = "formation_energy_eV_per_atom" if domain == "synthetic" else "dft_energy_eV_per_atom"
    for record in _private_records(filename):
        row = record["row"]; rid = row["id"]
        if rid not in expected: continue
        if row["split"] != split or rid in output: raise ValueError("Evaluation identity/split changed")
        value = row[field]
        if type(value) not in (int, float) or not math.isfinite(value): raise ValueError("Evaluation target must be finite")
        output[rid] = float(value)
    if set(output) != expected: raise ValueError("Evaluation target coverage changed")
    provenance = {"source_sha256": sha(source), "marker_sha256": sha(marker), "domain": domain, "n": len(output),
        "this_phase_domain_accessor_calls": 1, "historical_HOLDOUT_field_access_count_known": False,
        "prior_formal_stage_evidence": ["H2", "H3", "H3b"], "known_formal_stage_evidence_is_not_field_access_count": True,
        "target_values_returned_to_label_free_bundle": False}
    return {rid: output[rid] for rid in frozen_ids}, provenance


def synthetic_truth(ids, *, evaluation_marker, bundle): return _truth(ids, evaluation_marker, bundle, "synthetic")
def real_holdout_truth(ids, *, evaluation_marker, bundle): return _truth(ids, evaluation_marker, bundle, "real")
