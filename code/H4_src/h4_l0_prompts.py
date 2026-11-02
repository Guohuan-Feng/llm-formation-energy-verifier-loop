"""New frozen stage4.2 L0 template; full active source, no budget truncation.

The current set is exactly the prior training model's retained LLM features.
The ordinary inner-fit >20% failure mask may remove features; input budgeting
never does. Initial Magpie-only state has an empty current feature list.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

import numpy as np

import h4_prompts as feature_contract
import h4_l0_data as data
import h4_l0_queries as queries

VERSION = "H4-stage4.2-L0-feedback-features-and-queries-v1"
INPUT_LIMIT = 20000
INSTRUCTIONS = feature_contract.INSTRUCTIONS
FORMAT_REPAIR = feature_contract.FORMAT_REPAIR
ACTIVE_FEATURE_POLICY = (
    "Current features are exactly those used by the immediately preceding training model after its ordinary "
    "inner-fit TRAIN failure-rate>20% mask. Their original name/description/source are included in full. "
    "All raw generated features remain in the ledger/round metadata; L0 does not replay dropped-feature history. "
    "No feature is removed, shortened, reformatted or selected to fit the input budget. "
    "Initial state is Magpie-only with current_features=[]. If the full subsequent payload exceeds the frozen "
    "20000 conservative input bound, stop without calling an LLM or changing the prompt.")
PREFIX = (
    "Improve HGB descriptors from current validation feedback/summaries. Return JSON {features,queries,rationale}: "
    "new complete1..20 {name,description,source}, not a patch; rationale<=200 Unicode chars. L0: no history/notes. "
    "Initial []=Magpie-only. Current set=previous model's retained inner-fit features (>20% failures dropped); "
    "original code is complete, never budget-truncated. No external held-out feedback.\n")

def _compact_numeric_contract():
    box = feature_contract.sandbox
    values = [("builtins", sorted(box.SAFE_BUILTINS)),
        ("math", sorted(box.MATH_FUNCTIONS | box.MATH_CONSTANTS)),
        ("np", sorted(box.NUMPY_FUNCTIONS | box.NUMPY_CONSTANTS)),
        ("np.linalg", sorted(box.LINALG_FUNCTIONS)),
        ("data attrs", sorted(box.DATA_ATTRIBUTES)), ("data methods", sorted(box.DATA_METHODS))]
    return "Only these preprovided names/operations: " + "; ".join(name + ":" + ",".join(items) for name, items in values) + "\n"

DICTIONARY_CONTRACT = """Per source: only def feature(s), one numeric scalar; no helpers/decorators/annotations/outer code. Distinct descriptors, no predictors/material lookups; handle empty/None.
Readonly original-H3 s:
lattice={matrix:3x3 Angstrom row vectors,lengths_A:[a,b,c],angles_deg:[alpha,beta,gamma],a_A,b_A,c_A,alpha_deg,beta_deg,gamma_deg,volume_A3}.
sites=[{element:str,frac_coords:3 floats in[0,1),occupancy:float}].
neighbors=[{i:int,j:int,image:3 ints,distance_A:float}]: all directed periodic6-A neighbors; image translates j; no zero-image self/truncation.
composition={element:atomic_fraction}; n_sites,n_elements:int; volume_per_atom_A3,density_g_cm3:float.
element_properties={element:{Z,group,row,eneg,covalent_radius_A,mass_amu,valence_electrons,pymatgen_valence_subshell,first_ionization_energy_eV,electron_affinity_eV,melting_point_K,boiling_point_K,atomic_radius_A,missing_fields}}. Missing numbers=None; valence:H3 convention (not oxidation); subshell:(l,occupancy) or None. Atomic energies allowed; target/IDs/split absent.
basic_structure_statistics={volume_per_atom_A3,density_g_cm3,min_neighbor_distance_6A_A,mean_neighbor_distance_6A_A,std_neighbor_distance_6A_A,mean_coordination_6A,std_coordination_6A,lattice_anisotropy,max_angle_deviation_from90_deg}. std:ddof0; coordination:images included; empty distances:0; anisotropy:max(a,b,c)/min(a,b,c).
Local variables/if/loops/comprehensions/list/tuple/dict allowed. No import,I/O,network,reflection,private/dunder attrs,eval/exec,dynamic calls,recursion,random,input mutation,object dtype. Limits:function20000chars/2500ASTnodes;50ms/structure. AST/runtime/timeout failure=>NaN. Drop only at>20% inner-fit TRAIN failures, never validation/TEST failures.
""" + _compact_numeric_contract()
NUMERIC_FORMAT = {"summary": ".6g", "feedback": ".9g"}


class PromptBudgetExceeded(ValueError):
    pass


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def response_schema(bundle, capacity):
    if type(capacity) is not int or not 0 <= capacity <= 1500:
        raise ValueError("Capacity must be a nonnegative integer no larger than1500")
    result = queries.schema(bundle, capacity=1500)
    result["properties"]["features"] = deepcopy(feature_contract.FEATURE_SCHEMA["properties"]["features"])
    conditions = result["properties"]["queries"]["items"]["properties"]
    common = deepcopy(conditions["include_all"])
    if any(conditions[key] != common for key in ("include_all", "include_any", "exclude")):
        raise ValueError("Only identical element filters may share a schema definition")
    result["$defs"] = {"element_filter": common}
    for key in ("include_all", "include_any", "exclude"):
        conditions[key] = {"$ref": "#/$defs/element_filter"}
    result["required"] = ["features", "queries", "rationale"]
    return result


def canonical_schema_bytes(schema):
    return (_json(schema) + "\n").encode("utf-8")


def wire_schema(schema):
    return queries.strict_schema(schema)


def audit_input_bound(prompt, schema, *, include_repair=False):
    """Account the actual strict wire schema/note and optional repair suffix."""
    if type(prompt) is not str:
        raise ValueError("Prompt must be text")
    wire = wire_schema(schema)
    changed = _json(wire) != _json(schema)
    text = prompt + (FORMAT_REPAIR if include_repair else "") + (queries.WIRE_NOTE if changed else "")
    fixed = feature_contract._global_instruction_bytes() + len(INSTRUCTIONS.encode("utf-8")) + len(canonical_schema_bytes(wire))
    bound = len(text.encode("utf-8")) + fixed + 1500
    result = {"method": "utf8_byte_upper_bound_plus_1500_framing", "input_limit": INPUT_LIMIT,
        "canonical_prompt_bytes": len(prompt.encode("utf-8")), "wire_prompt_bytes": len(text.encode("utf-8")),
        "known_fixed_context_bytes": fixed, "input_token_upper_bound": bound,
        "strict_wire_note_included": changed, "format_repair_included": include_repair,
        "within_budget": bound <= INPUT_LIMIT}
    if bound > INPUT_LIMIT:
        raise PromptBudgetExceeded(f"Full unchanged L0 prompt exceeds frozen20000 input bound ({bound}); no source truncation or call is permitted")
    return result


def _feature_block(features):
    if not isinstance(features, list):
        raise ValueError("Current features must be a list")
    if features:
        feature_contract.validator({"features": features})
    return deepcopy(features)


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Feedback/summary values must be finite numbers")
    return float(value)


def _number(value, fmt):
    return format(_finite(float(value)), fmt)


def labelled_summary(labels, seed, *, bundle):
    data.validate_labels(labels, seed, bundle=bundle)
    energies = np.asarray([labels[rid] for rid in sorted(labels)], dtype=float)
    elements = Counter()
    counts = Counter()
    site_counts = []
    for rid in sorted(labels):
        row = bundle["by_id"][rid]
        elements.update(row.get("query_elements", row["composition"]))
        counts[row["n_elements"]] += 1
        site_counts.append(row["n_sites"])
    fmt = NUMERIC_FORMAT["summary"]
    return {"labeled_n": len(labels), "energy_eV_per_atom": {
        "min": _number(energies.min(), fmt), "mean": _number(energies.mean(), fmt),
        "std_ddof0": _number(energies.std(ddof=0), fmt),
        "quantiles_0_25_50_75_100": [_number(v, fmt) for v in np.quantile(energies, [0, .25, .5, .75, 1])]},
        "element_material_counts": dict(sorted(elements.items())),
        "n_elements_distribution": {str(k): v for k, v in sorted(counts.items())},
        "n_sites_quantiles_0_25_50_75_100": [_number(v, fmt) for v in np.quantile(site_counts, [0, .25, .5, .75, 1])]}


def _worst_rows(rows, labels, seed, bundle):
    mapping = bundle["perseed"][seed]["fit_group_mapping"]
    n_val = sum(mapping[bundle["groups"][rid]] == "val" for rid in labels)
    if not isinstance(rows, list) or len(rows) != min(20, n_val):
        raise ValueError("Feedback must contain the current validation's worst20 rows (or all if fewer)")
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "label_eV_per_atom", "prediction_eV_per_atom", "absolute_error_eV_per_atom"}:
            raise ValueError("Unexpected fields in current validation feedback")
        rid = row["id"]
        if rid in seen or rid not in labels or mapping[bundle["groups"][rid]] != "val":
            raise ValueError("Feedback rows must be distinct current internally labeled validation rows")
        label = _finite(row["label_eV_per_atom"])
        prediction = _finite(row["prediction_eV_per_atom"])
        error = _finite(row["absolute_error_eV_per_atom"])
        if label != labels[rid] or not math.isclose(error, abs(prediction - label), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Validation feedback differs from this seed's revealed label or prediction error")
        seen.add(rid)
        result.append({"formula": bundle["by_id"][rid]["formula"], "label_eV_per_atom": label,
            "prediction_eV_per_atom": prediction, "absolute_error_eV_per_atom": error})
    return result


def _pairs(value):
    return " ".join(str(k) + ":" + str(v) for k, v in value.items())


def _current_text(features):
    if not features:
        return "[]"
    parts = []
    for index, feature in enumerate(features):
        parts.append("@" + str(index) + " n=" + _json(feature["name"]) + " d=" + _json(feature["description"]) +
            " B=" + str(len(feature["source"].encode("utf-8"))) + "\n" +
            feature["source"] + "\n@end")
    return "\n".join(parts)


def _worst_text(rows):
    lines = ["formula\tlabel_eV_per_atom\tprediction_eV_per_atom\tabs_error_eV_per_atom"]
    for row in rows:
        if any(c in row["formula"] for c in "\t\r\n"):
            raise ValueError("Formula contains a feedback table delimiter")
        lines.append(row["formula"] + "\t" + "\t".join(_number(row[key], NUMERIC_FORMAT["feedback"])
            for key in ("label_eV_per_atom", "prediction_eV_per_atom", "absolute_error_eV_per_atom")))
    return "\n".join(lines)


def _summary_text(value):
    e = value["energy_eV_per_atom"]
    return ("n=" + str(value["labeled_n"]) + "; energy_eV_per_atom min/mean/std_ddof0=" +
        "/".join(e[k] for k in ("min", "mean", "std_ddof0")) + "; quantiles0/25/50/75/100=" +
        "/".join(e["quantiles_0_25_50_75_100"]) + "\nelement:material_count " + _pairs(value["element_material_counts"]) +
        "\nn_elements:count " + _pairs(value["n_elements_distribution"]) +
        "\nn_sites_quantiles0/25/50/75/100 " + "/".join(value["n_sites_quantiles_0_25_50_75_100"]))


def _catalogue_text(value):
    return ("unlabeled_n=" + str(value["unlabeled_POOL_n"]) + "\nelement:material_count " + _pairs(value["element_material_counts"]) +
        "\nn_elements:count " + _pairs(value["n_elements_distribution"]) +
        "\nn_sites_quantiles0/25/50/75/100 " + "/".join(_number(v, NUMERIC_FORMAT["summary"])
            for v in value["n_sites_quantiles_0_25_50_75_100"]) +
        "\ntemplate_spacegroup_number:count " + _pairs(value["template_spacegroup_distribution"]))


def build_request(seed, round_index, features, validation_mae, worst20, labels, *, carry=0, bundle=None):
    b = data._bundle(bundle)
    data.validate_labels(labels, seed, bundle=b)
    if type(round_index) is not int or not 1 <= round_index <= 20:
        raise ValueError("L0 round must be1..20")
    if type(carry) is not int or carry < 0 or 75 + carry > 1500:
        raise ValueError("Invalid carried quota")
    current = _feature_block(features)
    val_mae = _finite(validation_mae)
    if val_mae < 0:
        raise ValueError("Validation MAE must be nonnegative")
    worst = _worst_rows(worst20, labels, seed, b)
    capacity = 75 + carry
    summary = labelled_summary(labels, seed, bundle=b)
    catalogue = queries.catalogue(labels, bundle=b)
    schema = response_schema(b, capacity)
    prompt = (PREFIX + DICTIONARY_CONTRACT + queries.grammar_text(capacity) + "\n"
        + "Active code: @i n=name d=description B=source_UTF8_bytes; full original source follows each header; @end closes:\n" + _current_text(current) + "\n"
        + "Current validation rowMAE eV/atom: " + _number(val_mae, NUMERIC_FORMAT["feedback"]) + "\n"
        + "Current worst validation rows (TSV; largest absolute errors first):\n" + _worst_text(worst) + "\n"
        + "Current labeled-set summary:\n" + _summary_text(summary) + "\n"
        + "Unlabeled POOL catalogue (no target values):\n" + _catalogue_text(catalogue) + "\n"
        + "Return only final JSON with features,queries,rationale.\n")
    if any(rid in prompt for rid in b["ids"]):
        raise ValueError("A private material identifier entered the L0 prompt")
    ordinary = audit_input_bound(prompt, schema)
    repaired = audit_input_bound(prompt, schema, include_repair=True)
    return {"request_id": f"L0_seed{seed}_round{round_index:02d}", "kind": "features_query",
        "stage": "4.2", "arm": "L0", "seed": seed, "round_index": round_index,
        "prompt": prompt, "schema": schema, "instructions": INSTRUCTIONS,
        "max_format_repairs": 1, "admissible_format_failure": True,
        "canonical_prompt_sha256": _sha(prompt), "schema_sha256": _sha(canonical_schema_bytes(schema)),
        "wire_schema_sha256": _sha(canonical_schema_bytes(wire_schema(schema))),
        "input_budget": ordinary, "format_repair_input_budget": repaired,
        "current_features_sha256": data.json_hash(current), "current_feature_count": len(current),
        "current_complete_source_bytes": sum(len(f["source"].encode("utf-8")) for f in current),
        "labeled_ids_sha256": data.json_hash(sorted(labels)), "worst20_ids": [r["id"] for r in worst20],
        "summary_sha256": data.json_hash(summary), "catalogue_sha256": data.json_hash(catalogue),
        "carry": carry, "capacity": capacity}


def validate_response(payload, *, bundle, capacity):
    if not isinstance(payload, dict) or set(payload) != {"features", "queries", "rationale"}:
        raise ValueError("L0 response must contain exactly features,queries,rationale")
    feature_contract.validator({"features": payload["features"]})
    queries.validate({"queries": payload["queries"], "rationale": payload["rationale"]}, bundle=bundle, capacity=capacity)
    return True


def validator(bundle, capacity):
    return lambda payload: validate_response(payload, bundle=bundle, capacity=capacity)


def configuration(*, bundle=None):
    b = data._bundle(bundle)
    schema = response_schema(b, 75)
    return {"protocol_version": VERSION, "stage": "4.2", "condition": "L0",
        "prefix": PREFIX, "prefix_sha256": _sha(PREFIX),
        "dictionary_contract": DICTIONARY_CONTRACT, "dictionary_contract_sha256": _sha(DICTIONARY_CONTRACT),
        "active_feature_policy": ACTIVE_FEATURE_POLICY, "initial_current_features": [],
        "feature_schema": deepcopy(feature_contract.FEATURE_SCHEMA),
        "round1_schema_example": schema, "round1_wire_schema_example": wire_schema(schema),
        "query_element_schema_deduplication": "Three identical full element-filter schemas share local $defs/element_filter; enum/uniqueItems and all validation semantics unchanged",
        "round_schema_rule": "Schema n maximum is always1500 and its hash is identical for all rounds; semantic validator enforces actual total requested n<=75+carry, capacity<=1500",
        "query_grammar_version": queries.GRAMMAR_VERSION, "numeric_format": NUMERIC_FORMAT,
        "current_features_format": "@i n=full JSON name d=full JSON description B=source UTF8 bytecount; exact original source follows verbatim; @end delimiter",
        "worst20_format": "TSV formula,label_eV_per_atom,prediction_eV_per_atom,abs_error_eV_per_atom; 9 significant digits, all20 rows",
        "summary_catalogue_format": "All counts retained as ordered key:count pairs; all five declared quantiles retained",
        "instructions": INSTRUCTIONS, "instructions_sha256": _sha(INSTRUCTIONS),
        "format_repair_suffix": FORMAT_REPAIR, "format_repair_suffix_sha256": _sha(FORMAT_REPAIR),
        "wire_note": queries.WIRE_NOTE, "wire_note_sha256": _sha(queries.WIRE_NOTE),
        "input_limit": INPUT_LIMIT, "input_budget_method": "Actual strict wire schema+wire note+global instructions+UTF8 bytes+1500 framing; also check repair suffix",
        "over_budget_action": "Stop unchanged and report; never truncate/drop current active source for budgeting",
        "feedback_fields": ["formula", "label_eV_per_atom", "prediction_eV_per_atom", "absolute_error_eV_per_atom"],
        "feedback_scope": "Current seed's internally labeled validation rows only; no TEST/HOLDOUT labels or results",
        "rationale_limit_unicode_characters": 200, "max_queries": 3,
        "no_memory_notes_or_failed_feature_history": True}


round_request = build_request
