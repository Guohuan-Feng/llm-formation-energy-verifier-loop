"""Frozen compact stage4.1 requests and strict response-format validation.

This module never calls an LLM, fits a model, executes generated code or reads
TEST truth. Every one of 60 requests is prepared and budgeted before launch.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import h4_data as data

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import h3_sandbox as sandbox

PROTOCOL_VERSION = "H4-stage4.1-independent-features-v1"
INPUT_LIMIT = 20000
N_CANDIDATES = 20
N_EXAMPLES = 300
FORMAT_REPAIR = "\nFORMAT REPAIR ONLY: Return exactly the requested JSON schema with all required fields. No tools, files, examples, or additional information are available."
INSTRUCTIONS = (
    "Return only final JSON matching the response schema. Tools are forbidden. "
    "Do not use files, shell, network, or external services. "
    "Do not include chain-of-thought or reasoning steps.\n")
FEATURE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"features": {"type": "array", "minItems": 1, "maxItems": 20,
        "items": {"type": "object", "additionalProperties": False,
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 80},
                "description": {"type": "string", "minLength": 1, "maxLength": 800},
                "source": {"type": "string", "minLength": 1, "maxLength": sandbox.MAX_SOURCE_CHARS,
                    "description": "Exactly one top-level def feature(s); no other definitions or top-level code."}},
            "required": ["name", "description", "source"]}}},
    "required": ["features"]}

# This exact text, row format, schema and all request payload hashes are frozen
# in configuration before the first scientific call. Full original field_spec
# also lives in the public data config; no adaptive prompt shortening is used.
FEATURE_PREFIX = """Write 1 to 20 distinct scalar material descriptors to complement Magpie composition features in an HGB formation-energy regressor. This is an independent one-shot candidate: no other candidate, validation result, or feedback is supplied. Use only the labeled examples and summary below. Return exactly {"features":[{"name":...,"description":...,"source":...},...]}.
Each source must define exactly def feature(s), returning one finite float when possible; no helpers, decorators, annotations or outside code. Do not return predictions, fitted models, material lookup tables or duplicate/scaled copies. The label is absent from s; formula/example identifiers and experimental metadata are unavailable to code. Handle empty sequences and missing None elemental values.
s is the original recursively readonly H3 dictionary, with only:
lattice: matrix (3x3 row vectors Angstrom), lengths_A=[a,b,c], angles_deg=[alpha,beta,gamma], a_A,b_A,c_A,alpha_deg,beta_deg,gamma_deg,volume_A3.
sites: sequence of {element:str,frac_coords:[x,y,z] wrapped [0,1),occupancy:float}.
neighbors: all directed periodic neighbors within 6 Angstrom, {i:int,j:int,image:[int,int,int],distance_A:float}; image translates j; zero-image self excluded, no truncation.
composition: element->atomic fraction. n_sites:int; n_elements:int; volume_per_atom_A3; density_g_cm3.
element_properties: element->{Z,group,row,eneg,covalent_radius_A,mass_amu,valence_electrons,pymatgen_valence_subshell,first_ionization_energy_eV,electron_affinity_eV,melting_point_K,boiling_point_K,atomic_radius_A,missing_fields}. Missing numeric values are None. Valence counts outer period-shell electrons plus (n-1)d for d/f and (n-2)f for f-block; not oxidation states. Subshell is (l,occupancy) or None. Atomic tabulated ionization energy/affinity are permitted, never formation-energy labels.
basic_structure_statistics: volume_per_atom_A3,density_g_cm3,min_neighbor_distance_6A_A,mean_neighbor_distance_6A_A,std_neighbor_distance_6A_A,mean_coordination_6A,std_coordination_6A,lattice_anisotropy,max_angle_deviation_from90_deg. Distance std and per-site coordination std use ddof=0; coordination includes images. Empty distances give zero. Anisotropy=max(a,b,c)/min(a,b,c).
Use Python numeric expressions/local variables, if/for/while, lists/tuples/dicts and comprehensions. Only preprovided math, np and pure builtins below. No import, I/O, filesystem/network, reflection, private/double-underscore attributes, eval/exec, dynamic calls, recursion, randomness, mutation of inputs, or object dtype. Per function: 20000 source characters, 2500 AST nodes; per structure execution limit 50 ms. AST-invalid functions are recorded as failed, not repaired. Runtime failures/timeouts become NaN. Features failing on more than 20% of training rows are removed; held-out failures never choose features.
"""
EXAMPLE_COLUMNS = ("formula", "E_eV_per_atom", "n_sites", "volume_per_atom_A3", "mean_neighbor_distance_6A_A")
SUMMARY_POLICY = "All frozen POOL2000; labels may include internal validation rows; no TEST/HOLDOUT target or feedback"
NUMERIC_FORMAT = {"energy": ".6g", "structure": ".4g", "summary": ".5g"}


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def schema_bytes():
    return (json.dumps(FEATURE_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _numeric_contract():
    return "Numeric allowlists (only these operations): " + _json({
        "builtins": sorted(sandbox.SAFE_BUILTINS),
        "math": sorted(sandbox.MATH_FUNCTIONS | sandbox.MATH_CONSTANTS),
        "np": sorted(sandbox.NUMPY_FUNCTIONS | sandbox.NUMPY_CONSTANTS),
        "np.linalg": sorted(sandbox.LINALG_FUNCTIONS),
        "data_attributes": sorted(sandbox.DATA_ATTRIBUTES),
        "data_methods": sorted(sandbox.DATA_METHODS)}) + "\n"


def _global_instruction_bytes():
    # Same read-only storage guidance accounting as frozen H4A/e2 transport.
    import e2_transport
    return e2_transport._global_instruction_bytes()


def audit_input_bound(prompt, *, protocol_files=None):
    """Use the transport's byte bound when protocol files have been written."""
    if type(prompt) is not str:
        raise ValueError("Prompt must be a string")
    if protocol_files is None:
        fixed = _global_instruction_bytes() + len(INSTRUCTIONS.encode("utf-8")) + len(schema_bytes())
        result = {"method": "utf8_byte_upper_bound_plus_1500_framing", "prompt_bytes": len(prompt.encode("utf-8")),
            "known_fixed_context_bytes": fixed,
            "input_token_upper_bound": len(prompt.encode("utf-8")) + fixed + 1500}
    else:
        schema_path = data.checked(protocol_files["features_schema_path"], h4_only=True)
        instruction_path = data.checked(protocol_files["instructions_path"], h4_only=True)
        if schema_path.read_bytes() != schema_bytes() or instruction_path.read_bytes() != INSTRUCTIONS.encode("utf-8"):
            raise ValueError("Protocol schema/instructions differ from frozen stage4.1 template")
        import e2_transport
        result = e2_transport.input_budget(prompt, schema_path=schema_path,
            instructions_path=instruction_path)
        result["transport_input_limit"] = result["input_limit"]
    result.update(input_limit=INPUT_LIMIT, within_budget=result["input_token_upper_bound"] <= INPUT_LIMIT)
    if not result["within_budget"]:
        raise ValueError("H4 conservative 20000 input-token bound exceeded: " + str(result["input_token_upper_bound"]))
    return result


def sample_ids(seed, candidate_index, *, bundle=None):
    b = data._bundle(bundle)
    if type(seed) is not int or seed not in data.SEEDS:
        raise ValueError("Seed must be 0, 1 or 2")
    if type(candidate_index) is not int or candidate_index not in range(N_CANDIDATES):
        raise ValueError("Candidate index must be 0..19")
    # A deterministic hash-random order avoids any dependency on process RNG.
    members = b["perseed"][seed]["O_ids"]
    return sorted(members, key=lambda rid: (
        _sha(f"H4-O-example300-v1|{seed}|{candidate_index}|{rid}"), rid))[:N_EXAMPLES]


def _display_number(value, fmt):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Prompt numeric values must be finite")
    return format(value, fmt)


def _summary(seed, labels, bundle):
    ids = bundle["perseed"][seed]["O_ids"]
    energies = np.asarray([labels[rid] for rid in ids], dtype=float)
    elements = Counter()
    site_counts = []
    for rid in ids:
        record = bundle["by_id"][rid]
        elements.update(record["composition"].keys())
        site_counts.append(record["n_sites"])
    fmt = NUMERIC_FORMAT["summary"]
    return {"n_labeled": 2000, "energy_eV_per_atom": {
            "min": _display_number(energies.min(), fmt), "mean": _display_number(energies.mean(), fmt),
            "std_ddof0": _display_number(energies.std(ddof=0), fmt),
            "quantiles_0_25_50_75_100": [_display_number(v, fmt) for v in np.quantile(energies, [0, .25, .5, .75, 1])]},
        "n_sites_quantiles_0_25_50_75_100": [_display_number(v, fmt) for v in np.quantile(site_counts, [0, .25, .5, .75, 1])],
        "element_material_counts": dict(sorted(elements.items()))}


def _example_table(selected, labels, bundle):
    rows = ["\t".join(EXAMPLE_COLUMNS)]
    for rid in selected:
        record = bundle["by_id"][rid]
        formula = record["formula"]
        if any(char in formula for char in "\t\r\n"):
            raise ValueError("Formula cannot contain table delimiters")
        statistics = record["structure_statistics"]
        rows.append("\t".join((formula, _display_number(labels[rid], NUMERIC_FORMAT["energy"]),
            str(record["n_sites"]), _display_number(statistics["volume_per_atom_A3"], NUMERIC_FORMAT["structure"]),
            _display_number(statistics["mean_neighbor_distance_6A_A"], NUMERIC_FORMAT["structure"]))))
    if len(rows) != N_EXAMPLES + 1:
        raise ValueError("Every candidate must contain exactly 300 examples")
    return "\n".join(rows) + "\n"


def feature_request(seed, candidate_index, labels, *, bundle=None, protocol_files=None):
    b = data._bundle(bundle)
    data.validate_training_labels(labels, seed, bundle=b)
    selected = sample_ids(seed, candidate_index, bundle=b)
    summary = _summary(seed, labels, b)
    table = _example_table(selected, labels, b)
    prompt = (FEATURE_PREFIX + _numeric_contract() +
        "Examples are occupancy formulas of supplied cells (not reduced); geometry is the same frozen relaxed H3 cache. "
        "The 300 rows are a fixed uniform sample of this candidate's 2000 labeled training materials. "
        "The summary uses those 2000. They may contain internal-validation labels, as authorized; no external held-out labels are supplied.\n"
        "Summary:\n" + _json(summary) + "\nExamples (tab-separated):\n" + table +
        "Return only the final JSON feature set.\n")
    if any(rid in prompt for rid in b["ids"]):
        raise ValueError("A private material ID entered the prompt")
    bound = audit_input_bound(prompt, protocol_files=protocol_files)
    repair_bound = audit_input_bound(prompt + FORMAT_REPAIR, protocol_files=protocol_files)
    return {"request_id": f"O_seed{seed}_candidate{candidate_index:02d}",
        "kind": "features", "arm": "O", "seed": seed, "candidate_index": candidate_index,
        "prompt": prompt, "canonical_prompt_sha256": _sha(prompt),
        "schema_sha256": _sha(schema_bytes()), "sample_ids": selected,
        "sample_ids_sha256": data.json_hash(selected), "example_count": N_EXAMPLES,
        "summary_sha256": data.json_hash(summary), "input_budget": bound,
        "format_repair_input_budget": repair_bound,
        "sampling_method": "SHA256(H4-O-example300-v1|seed|candidate_index|id), take first300 without replacement"}


def build_requests(bundle=None, *, protocol_files=None):
    b = data._bundle(bundle)
    requests = []
    for seed in data.SEEDS:
        labels = data.reveal_training(seed, bundle=b)
        requests.extend(feature_request(seed, candidate, labels, bundle=b, protocol_files=protocol_files)
            for candidate in range(N_CANDIDATES))
    if len(requests) != 60 or len({request["request_id"] for request in requests}) != 60:
        raise ValueError("Exactly 60 independent feature requests are required")
    return requests


def configuration(requests, *, bundle=None):
    b = data._bundle(bundle)
    expected = [(seed, candidate) for seed in data.SEEDS for candidate in range(N_CANDIDATES)]
    if [(r["seed"], r["candidate_index"]) for r in requests] != expected:
        raise ValueError("Frozen request order must contain all 60 seed/candidate pairs")
    for request in requests:
        if request["canonical_prompt_sha256"] != _sha(request["prompt"]):
            raise ValueError("A prepared prompt changed before freeze")
        if request["sample_ids"] != sample_ids(request["seed"], request["candidate_index"], bundle=b):
            raise ValueError("Prepared example membership changed")
        audit_input_bound(request["prompt"])
        audit_input_bound(request["prompt"] + FORMAT_REPAIR)
    return {"protocol_version": PROTOCOL_VERSION, "candidate_indexing": "0..19 per seed",
        "requests": [{key: value for key, value in request.items() if key != "prompt"} for request in requests],
        "request_order_sha256": data.json_hash([r["request_id"] for r in requests]),
        "all_prompt_hashes_sha256": data.json_hash([r["canonical_prompt_sha256"] for r in requests]),
        "feature_prefix": FEATURE_PREFIX, "numeric_contract": _numeric_contract(),
        "feature_prefix_sha256": _sha(FEATURE_PREFIX), "numeric_contract_sha256": _sha(_numeric_contract()),
        "feature_schema": FEATURE_SCHEMA, "schema_file_sha256": _sha(schema_bytes()),
        "instructions": INSTRUCTIONS, "instructions_sha256": _sha(INSTRUCTIONS),
        "format_repair_suffix": FORMAT_REPAIR, "format_repair_suffix_sha256": _sha(FORMAT_REPAIR),
        "example_columns": list(EXAMPLE_COLUMNS), "numeric_format": NUMERIC_FORMAT,
        "summary_policy": SUMMARY_POLICY, "example_count_per_request": N_EXAMPLES,
        "independent_candidates_per_seed": N_CANDIDATES, "requests_total": 60,
        "input_limit": INPUT_LIMIT,
        "maximum_conservative_input_token_upper_bound": max(r["input_budget"]["input_token_upper_bound"] for r in requests),
        "maximum_format_repair_input_token_upper_bound": max(r["format_repair_input_budget"]["input_token_upper_bound"] for r in requests),
        "feature_dictionary_schema_sha256": b["field_spec_sha256"],
        "AST_invalid_features": "scientific failure records; never trigger a format-repair request",
        "no_TEST_or_HOLDOUT_labels_in_requests": True}


def validate_feature_payload(payload):
    """Format only: return True; generated-source AST validity is not a repair gate."""
    if not isinstance(payload, Mapping) or set(payload) != {"features"}:
        raise ValueError("Feature response must contain exactly features")
    features = payload["features"]
    if not isinstance(features, (list, tuple)) or not 1 <= len(features) <= 20:
        raise ValueError("Feature response requires 1 to 20 functions")
    for feature in features:
        if not isinstance(feature, Mapping) or set(feature) != {"name", "description", "source"}:
            raise ValueError("Each feature must contain exactly name,description,source")
        for key, limit in (("name", 80), ("description", 800), ("source", sandbox.MAX_SOURCE_CHARS)):
            if type(feature[key]) is not str or not 1 <= len(feature[key]) <= limit:
                raise ValueError("Invalid feature " + key)
    if len({f["name"] for f in features}) != len(features) or len({f["source"] for f in features}) != len(features):
        raise ValueError("Feature names and source strings must be distinct")
    return True


validator = validate_feature_payload
