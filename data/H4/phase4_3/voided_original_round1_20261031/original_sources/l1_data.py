"""Independent L1 POOL oracle; reuse the immutable L0 label-free inputs.

Import/loading never accesses TEST targets, trains or generates features. L1
has its own reveal registry and a distinct first terminal scoring gate.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
SRC = ROOT / "work/h-data/H4/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from copy import deepcopy
import gzip
import json
import math
import h4_l0_data as frozen

DATA = ROOT / "work/h-data/H4"
PHASE = DATA / "phase4_3"
PRIOR_MARKERS = {"4_1": DATA / "phase4_1/evaluation_started.json",
                 "4_2": DATA / "phase4_2/evaluation_started.json"}
sha = frozen.sha
json_hash = frozen.json_hash
_BUNDLE = None
_REVEALED = {}
_TRUTH_READ = False


def checked(path, *, writable=False, must_exist=True):
    value = Path(path).resolve(strict=must_exist)
    limit = PHASE.resolve() if writable else ROOT.resolve()
    if "onedrive" in str(value).casefold() or not value.is_relative_to(limit):
        raise ValueError("L1 path escapes its non-OneDrive permitted root")
    return value


def load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        previous = frozen.load_bundle()
        result = dict(previous)
        result["perseed"] = deepcopy(previous["perseed"])
        result["source_hashes"] = dict(previous["source_hashes"])
        for path in PRIOR_MARKERS.values():
            checked(path)
            result["source_hashes"][str(path)] = sha(path)
        result["input_hashes"] = dict(result["source_hashes"])
        result.update(evaluation_truth_loaded=False, current_condition="L1",
                      previous_L0_bundle_mutated=False)
        _BUNDLE = result
    return _BUNDLE


def _bundle(bundle):
    return load_bundle() if bundle is None else bundle


def validate_labels(labels, seed, *, bundle=None, require_n=None):
    b = _bundle(bundle)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("L1 seed must be0/1/2")
    if (not isinstance(labels, dict) or not 500 <= len(labels) <= 2000
            or not set(b["perseed"][seed]["S0_ids"]) <= set(labels) <= set(b["pool_ids"])):
        raise ValueError("L1 labels must be its S0500 plus actually selected POOL members")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in labels.values()):
        raise ValueError("L1 labels must be finite scalars")
    if require_n is not None and len(labels) != require_n:
        raise ValueError("L1 labelled cardinality differs")
    return True


def reveal(ids, seed, *, bundle=None):
    b = _bundle(bundle)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("L1 seed must be0/1/2")
    ids = list(ids)
    if len(ids) != len(set(ids)) or not set(ids) <= set(b["pool_ids"]):
        raise ValueError("L1 may reveal only distinct POOL identities")
    observed = _REVEALED.setdefault(seed, {})
    authorized = set(observed) | set(ids)
    if len(authorized) > 2000 or not set(b["perseed"][seed]["S0_ids"]) <= authorized:
        raise ValueError("Initialize S0500 before further L1 reveals; additional budget is1500")
    missing = set(ids) - set(observed)
    if missing:
        source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
        if sha(source) != b["source_hashes"][str(source)]:
            raise ValueError("Frozen POOL target source changed")
        with gzip.open(source, "rt", encoding="utf-8") as stream:
            mixed = json.load(stream)
        fresh = {}
        for item in mixed:
            row = item["row"]
            rid = row["id"]
            if rid not in missing:
                continue  # Never look up another condition's or TEST target field.
            if row["split"] != "POOL" or rid in fresh:
                raise ValueError("L1 POOL target identity differs")
            value = row["formation_energy_eV_per_atom"]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("L1 POOL target must be finite")
            fresh[rid] = float(value)
        if set(fresh) != missing:
            raise ValueError("L1 selected POOL target coverage incomplete")
        observed.update(fresh)
    return {rid: observed[rid] for rid in ids}


def initial_labels(seed, *, bundle=None):
    b = _bundle(bundle)
    return reveal(b["perseed"][seed]["S0_ids"], seed, bundle=b)


def score_truth(ids, *, evaluation_marker, bundle=None):
    """One L1-only terminal read for all three seeds; O/L0 are not rescored."""
    global _TRUTH_READ
    b = _bundle(bundle)
    ids = list(ids)
    if ids != b["test_ids"]:
        raise ValueError("L1 terminal scoring requires frozen ordered TEST1000")
    marker_path = checked(evaluation_marker)
    if marker_path != checked(PHASE / "evaluation_started.json"):
        raise ValueError("L1 accepts only its exclusive phase4_3 final gate")
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    required = {"stage": "4.3", "condition": "L1", "scoring_pass": 1,
        "current_L1_scoring_pass": 1, "prior_H4_TEST_scoring_passes": 2,
        "TEST_targets_previously_read_for_this_condition": False,
        **{"prior_stage" + stage + "_marker_sha256": sha(checked(path))
           for stage, path in PRIOR_MARKERS.items()}}
    if any(type(marker.get(key)) is not type(value) or marker.get(key) != value for key, value in required.items()):
        raise ValueError("L1 gate identity or prior O/L0 scoring-pass disclosure differs")
    if _TRUTH_READ or b.get("evaluation_truth_loaded"):
        raise RuntimeError("L1 TEST truth has already been read")
    source = checked(ROOT / "work/h-data/H2/synthetic_records_private.json.gz")
    if sha(source) != b["source_hashes"][str(source)]:
        raise ValueError("Frozen TEST source changed")
    with gzip.open(source, "rt", encoding="utf-8") as stream:
        mixed = json.load(stream)
    truth = {}
    _TRUTH_READ = True
    b["evaluation_truth_loaded"] = True
    authorized = set(ids)
    for item in mixed:
        row = item["row"]
        rid = row["id"]
        if rid not in authorized:
            continue
        if row["split"] != "TEST" or rid in truth:
            raise ValueError("L1 TEST target identity differs")
        value = row["formation_energy_eV_per_atom"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("L1 TEST target must be finite")
        truth[rid] = float(value)
    if set(truth) != authorized:
        raise ValueError("L1 TEST target coverage incomplete")
    return {rid: truth[rid] for rid in ids}, {**required, "TEST_targets_read": 1000,
        "current_condition": "L1", "prior_conditions": ["O", "L0"],
        "source_path": str(source), "source_sha256": b["source_hashes"][str(source)],
        "evaluation_marker": str(marker_path), "evaluation_marker_sha256": sha(marker_path),
        "test_ids_sha256": json_hash(ids), "HOLDOUT_target_fields_accessed": False}


def configuration(*, bundle=None):
    b = _bundle(bundle)
    cfg = frozen.configuration(bundle=b)
    cfg.update(data_protocol="H4-stage4.3-L1-frozen-inputs-v1", stage="4.3", condition="L1",
        POOL_label_access="L1's identical S0500 plus its own actually selected POOL members only",
        TEST_scoring="One separate L1 terminal pass; prior O and L0 markers preserved, no rescoring",
        query_selection_arm="L1", initialization="Reuse validated original L0 Magpie-only S0500 model/cache; exclude baseline from round candidates")
    cfg.update({"prior_stage" + stage + "_marker_sha256": sha(checked(path))
                for stage, path in PRIOR_MARKERS.items()})
    return cfg
