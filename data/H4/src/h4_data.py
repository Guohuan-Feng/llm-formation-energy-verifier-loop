"""Frozen, read-only H4 inputs; POOL labels and final TEST truth have separate gates.

Loading the bundle never inspects a TEST/HOLDOUT target field. H4 reuses H3's
existing label-free geometry and Magpie cache without executing descriptors.
No scientific operation or filesystem write occurs on import or in this module.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "work/h-data/H4"
H2 = ROOT / "work/h-data/H2"
H3 = ROOT / "work/h-data/H3"
SEEDS = (0, 1, 2)
INNER_SPLIT_VERSION = "H3-CGCNN-inner-v1"
TRAIN_MEMBERSHIP_VERSION = "H2-S0-topup-v1"
_BUNDLE = None
_TRAIN_LABELS = {}
_TRUTH_ALREADY_READ = False


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def checked(path, *, h4_only=False, must_exist=True):
    resolved = Path(path).resolve(strict=must_exist)
    limit = CACHE.resolve() if h4_only else ROOT.resolve()
    if "onedrive" in str(resolved).casefold() or not resolved.is_relative_to(limit):
        raise ValueError("H4 paths must resolve below the non-OneDrive research root")
    return resolved


def _read(path):
    return json.loads(checked(path).read_text(encoding="utf-8"))


def _inner_split(ids, groups, seed):
    ordered = sorted(set(groups[rid] for rid in ids), key=lambda group: (
        hashlib.sha256(f"{INNER_SPLIT_VERSION}|{seed}|{group}".encode()).hexdigest(), group))
    if len(ordered) < 2:
        raise ValueError("At least two template groups are required")
    n_val = min(len(ordered) - 1, max(1, math.ceil(.2 * len(ordered))))
    validation_groups = set(ordered[:n_val])
    fit = sorted(rid for rid in ids if groups[rid] not in validation_groups)
    val = sorted(rid for rid in ids if groups[rid] in validation_groups)
    if not fit or not val or set(fit) & set(val) or set(fit + val) != set(ids):
        raise ValueError("Invalid fixed template split")
    if {groups[rid] for rid in fit} & {groups[rid] for rid in val}:
        raise ValueError("Template leakage in internal validation")
    return fit, val


def _formula(s):
    amounts = Counter()
    for site in s["sites"]:
        amounts[site["element"]] += site["occupancy"]
    return "".join(symbol + format(amounts[symbol], ".10g") for symbol in sorted(amounts))


def _pool_metadata(bundle, source):
    """Whitelist metadata only, before looking up any energy key in mixed records."""
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    pool = set(bundle["pool_ids"])
    seen = set()
    for item in records:
        row = item["row"]
        rid = row["id"]
        if rid not in pool:
            continue
        if row["split"] != "POOL" or rid in seen:
            raise ValueError("Frozen H2 POOL metadata identity changed")
        reduced = row["reduced_composition"]
        tokens = reduced.split("|")
        if any(":" not in token for token in tokens):
            raise ValueError("Malformed H2 reduced-composition metadata")
        elements = [token.split(":", 1)[0] for token in tokens]
        record = bundle["by_id"][rid]
        if (len(set(elements)) != len(elements)
                or set(elements) != set(record["composition"])
                or row["n_sites"] != record["n_sites"]):
            raise ValueError("H2 metadata differs from frozen H3 geometry")
        record.update(query_elements=sorted(elements), reduced_composition=reduced,
            template_id=row["template_id"], template_group=row["template_group"])
        seen.add(rid)
    if seen != pool:
        raise ValueError("POOL metadata coverage is incomplete")
    return {"status": "PASS", "n_POOL_checked": len(seen), "target_fields_accessed": False,
        "query_fields_available": ["include_all", "include_any", "exclude", "n_elements", "n_sites"],
        "template_spacegroup_available_in_H2_records": False,
        "template_spacegroup_note": "No template space-group field is stored in H2 row or cached H3 structure; stage4.1 makes no query and does not infer one."}


def load_bundle():
    """Load all synthetic POOL5000/TEST1000, with no target-field access."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE
    split_path = checked(H2 / "split_manifest.json")
    split = _read(split_path)
    if split["manifest_content_sha256"] != json_hash({
            key: value for key, value in split.items() if key != "manifest_content_sha256"}):
        raise ValueError("H2 split manifest self-hash changed")
    plan_path = checked(H3 / "plan.json")
    plan = _read(plan_path)
    if plan["field_spec_sha256"] != json_hash(plan["field_spec"]):
        raise ValueError("H3 feature dictionary contract changed")
    meta_path = checked(H3 / "representation_metadata.json")
    meta = _read(meta_path)
    if meta["status"] != "complete" or meta.get("target_values_in_cache") is not False:
        raise ValueError("H3 cache must be complete and label-free")
    base_path = checked(meta["base_features_path"])
    structure_path = checked(meta["structure_cache_path"])
    for name, path in (("base_features_path", base_path), ("structure_cache_path", structure_path)):
        if sha(path) != meta["file_sha256"][name]:
            raise ValueError("Frozen H3 label-free cache hash changed")
    pool = list(split["split_ids"]["synthetic_POOL"])
    test = list(split["split_ids"]["synthetic_TEST"])
    if len(pool) != 5000 or len(test) != 1000 or len(set(pool + test)) != 6000:
        raise ValueError("H4 requires frozen disjoint POOL5000 and TEST1000")
    keep = set(pool + test)
    with np.load(base_path, allow_pickle=False) as arrays:
        old_ids = arrays["ids"].tolist()
        if json_hash(old_ids) != meta["ids_sha256"]:
            raise ValueError("H3 feature cache ID order changed")
        positions = [position for position, rid in enumerate(old_ids) if rid in keep]
        ids = [old_ids[position] for position in positions]
        magpie = arrays["magpie132"][positions].copy()
    if len(ids) != 6000 or set(ids) != keep or magpie.shape != (6000, 132) or not np.isfinite(magpie).all():
        raise ValueError("Synthetic Magpie132 coverage or finite values changed")
    index = {rid: position for position, rid in enumerate(ids)}
    group_path = checked(H3 / "cgcnn/plan.json")
    all_groups = _read(group_path)["groups"]
    if not keep <= set(all_groups):
        raise ValueError("Frozen synthetic template groups are incomplete")
    groups = {rid: all_groups[rid] for rid in ids}
    structures = {}
    with gzip.open(structure_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            rid = item["id"]
            if rid in keep:
                if rid in structures:
                    raise ValueError("Duplicate label-free structure")
                structures[rid] = item["s"]
            if len(structures) == 6000:
                break
    if set(structures) != keep:
        raise ValueError("Label-free structure coverage is incomplete")
    records = []
    pool_set = set(pool)
    for rid in ids:
        s = structures[rid]
        if set(s) != set(plan["field_spec"]["top_level"]) | {"schema_version"}:
            raise ValueError("Cached structure no longer matches frozen label-free field whitelist")
        records.append({"id": rid, "split": "POOL" if rid in pool_set else "TEST",
            "formula": _formula(s), "composition": dict(s["composition"]),
            "n_sites": s["n_sites"], "n_elements": s["n_elements"],
            "structure_statistics": dict(s["basic_structure_statistics"]), "group": groups[rid]})
    perseed = {}
    for seed in SEEDS:
        old = plan["training_sets"][str(seed)]
        S0 = list(old["S0_ids"])
        O = list(old["synthetic_train_ids"])
        if (len(set(S0)) != 500 or len(set(O)) != 2000 or not set(S0) <= set(O) <= pool_set
                or set(S0) != set(split["S0"][str(seed)])
                or set(O) != set(split["h2_rf_synthetic_training_ids_by_seed"][str(seed)])):
            raise ValueError("H2/H3 fixed seed memberships differ")
        remaining = sorted(pool_set - set(S0), key=lambda rid: (
            hashlib.sha256(f"{TRAIN_MEMBERSHIP_VERSION}|{seed}|{rid}".encode()).hexdigest(), rid))
        if O != sorted(S0 + remaining[:1500]):
            raise ValueError("O2000 no longer matches frozen uniform hash-random topup")
        fit, val = _inner_split(O, groups, seed)
        perseed[seed] = {"S0_ids": S0, "O_ids": O, "fit_ids": fit, "val_ids": val,
            "additional_ids": sorted(set(O) - set(S0)),
            "fit_groups": sorted({groups[rid] for rid in fit}),
            "val_groups": sorted({groups[rid] for rid in val})}
    private_path = checked(H2 / "synthetic_records_private.json.gz")
    h2_metrics_path = checked(ROOT / "H2_formal_dataset/metrics.json")
    expected = _read(h2_metrics_path)["private_dataset_hashes"][private_path.name]
    if sha(private_path) != expected:
        raise ValueError("Frozen H2 mixed private records changed")
    paths = (split_path, plan_path, meta_path, base_path, structure_path, group_path,
        private_path, h2_metrics_path)
    hashes = {str(path): sha(path) for path in paths}
    magpie.setflags(write=False)
    bundle = {"ids": ids, "index": index, "magpie132": magpie,
        "structures_by_id": structures, "records": records,
        "by_id": {record["id"]: record for record in records},
        "groups": groups, "pool_ids": pool, "test_ids": test, "perseed": perseed,
        "source_hashes": hashes, "input_hashes": dict(hashes),
        "field_spec": plan["field_spec"], "field_spec_sha256": plan["field_spec_sha256"],
        "evaluation_truth_loaded": False,
        "provenance": {"label_free_geometry_reused": True, "generated_functions_executed": 0,
            "target_fields_accessed_by_load_bundle": False, "feature_dictionary": "original H3 cached s",
            "training_membership": "Exact frozen H2/H3 random2000 membership reused for every seed"}}
    bundle["query_metadata_audit"] = _pool_metadata(bundle, private_path)
    _BUNDLE = bundle
    return bundle


def _bundle(bundle):
    return bundle if bundle is not None else load_bundle()


def validate_training_labels(labels, seed, *, bundle=None):
    b = _bundle(bundle)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Seed must be 0, 1 or 2")
    if (not isinstance(labels, dict) or set(labels) != set(b["perseed"][seed]["O_ids"])
            or not set(labels) <= set(b["pool_ids"])):
        raise ValueError("Training labels must cover exactly this seed's frozen POOL2000")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in labels.values()):
        raise ValueError("Training labels must be finite scalars")
    return True


def reveal_training(seed, *, bundle=None):
    """Read label fields only for this seed's authorized 2000 POOL members."""
    b = _bundle(bundle)
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("Seed must be 0, 1 or 2")
    if seed not in _TRAIN_LABELS:
        path = checked(H2 / "synthetic_records_private.json.gz")
        if sha(path) != b["source_hashes"][str(path)]:
            raise ValueError("Frozen H2 label bytes changed")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = json.load(stream)
        authorized = set(b["perseed"][seed]["O_ids"])
        labels = {}
        for item in records:
            row = item["row"]
            rid = row["id"]
            if rid not in authorized:
                continue  # No unselected POOL or TEST target field is accessed.
            if row["split"] != "POOL" or rid in labels:
                raise ValueError("Authorized training target identity differs")
            value = row["formation_energy_eV_per_atom"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Authorized POOL target must be finite")
            labels[rid] = float(value)
        validate_training_labels(labels, seed, bundle=b)
        _TRAIN_LABELS[seed] = labels
    return {rid: _TRAIN_LABELS[seed][rid] for rid in b["perseed"][seed]["O_ids"]}


def score_truth(ids, *, evaluation_marker, bundle=None):
    """Single terminal read of all frozen TEST1000 after a committed H4 gate.

    One truth mapping serves all three seeds in the single terminal evaluation.
    The caller must have committed all complete predictions before making its
    exclusive evaluation_started marker. This accessor never creates the gate.
    """
    global _TRUTH_ALREADY_READ
    b = _bundle(bundle)
    if list(ids) != b["test_ids"]:
        raise ValueError("Terminal truth access requires all TEST1000 in frozen order")
    marker_path = checked(evaluation_marker, h4_only=True)
    expected_path = checked(CACHE / "phase4_1/evaluation_started.json")
    if marker_path != expected_path:
        raise ValueError("Only the stage4.1 exclusive final-evaluation marker is accepted")
    marker = _read(marker_path)
    if (marker.get("stage") != "4.1" or type(marker.get("scoring_pass")) is not int or marker.get("scoring_pass") != 1
            or marker.get("TEST_targets_previously_read") is not False):
        raise ValueError("Terminal evaluation marker is malformed or discloses prior TEST access")
    if _TRUTH_ALREADY_READ or b.get("evaluation_truth_loaded"):
        raise RuntimeError("TEST truth has already been read in this evaluation process")
    path = checked(H2 / "synthetic_records_private.json.gz")
    if sha(path) != b["source_hashes"][str(path)]:
        raise ValueError("Frozen TEST source bytes changed")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    authorized = set(ids)
    truth = {}
    _TRUTH_ALREADY_READ = True  # Any target-access attempt is terminal, including failure.
    b["evaluation_truth_loaded"] = True
    for item in records:
        row = item["row"]
        rid = row["id"]
        if rid not in authorized:
            continue
        if row["split"] != "TEST" or rid in truth:
            raise ValueError("Frozen TEST target identity differs")
        value = row["formation_energy_eV_per_atom"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Frozen TEST target must be finite")
        truth[rid] = float(value)
    if set(truth) != authorized:
        raise ValueError("Frozen TEST truth coverage is incomplete")
    return {rid: truth[rid] for rid in ids}, {
        "stage": "4.1", "scoring_pass": 1, "TEST_targets_read": 1000,
        "source_path": str(path), "source_sha256": b["source_hashes"][str(path)],
        "evaluation_marker": str(marker_path), "evaluation_marker_sha256": sha(marker_path),
        "test_ids_sha256": json_hash(list(ids)), "HOLDOUT_target_fields_accessed": False}


def configuration(*, bundle=None):
    b = _bundle(bundle)
    seeds = {}
    for seed in SEEDS:
        split = b["perseed"][seed]
        seeds[str(seed)] = {key: list(value) for key, value in split.items()}
        seeds[str(seed)]["sha256"] = {key: json_hash(value) for key, value in split.items()}
    return {"data_protocol": "H4-stage4.1-frozen-inputs-v1", "seeds": list(SEEDS),
        "pool_ids": list(b["pool_ids"]), "test_ids": list(b["test_ids"]),
        "pool_ids_sha256": json_hash(b["pool_ids"]), "test_ids_sha256": json_hash(b["test_ids"]),
        "feature_row_ids_sha256": json_hash(b["ids"]), "template_groups_sha256": json_hash(b["groups"]),
        "perseed": seeds, "source_hashes": dict(b["source_hashes"]),
        "field_spec": b["field_spec"], "field_spec_sha256": b["field_spec_sha256"],
        "membership_method": "Exact H2/H3 S0500 plus 1500 remaining IDs ordered by SHA256(H2-S0-topup-v1|seed|id), final IDs sorted",
        "inner_split_method": "Distinct template groups ordered SHA256(H3-CGCNN-inner-v1|seed|group); first ceil(20% groups) validation; no group overlap",
        "HGB_fit_weights": "unweighted, same as H3", "validation_selection": "row MAE, smaller is better",
        "prompt_examples_may_include_internal_validation_labels": True,
        "prompt_examples_scope": "this seed's frozen POOL2000 only; no TEST/HOLDOUT labels",
        "TEST_truth_access": "One terminal read for all three seeds after phase4_1/evaluation_started.json",
        "query_metadata_audit": dict(b["query_metadata_audit"])}
