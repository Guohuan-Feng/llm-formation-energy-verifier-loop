"""New stage4.2 label-free query grammar; H4A selection RNG is reused verbatim.

This module never reveals a label or calls a model. A query shortfall is carried
forward, never silently filled; the coordinator owns the post-round20 fill.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import random
import math
import numpy as np

FIELDS = {"include_all", "include_any", "exclude", "n_elements", "n_sites", "template_spacegroup", "n"}
MAX_QUERIES = 3
GRAMMAR_VERSION = "H4-stage4.2-L0-query-grammar-v1"


def allowed_elements(bundle):
    return sorted({e for rid in bundle["pool_ids"] for e in _elements(bundle["by_id"][rid])})


def _elements(record):
    return set(record.get("query_elements", record["composition"]))


def schema(bundle, capacity=75):
    if type(capacity) is not int or capacity < 0: raise ValueError("Capacity must be a nonnegative integer")
    elements = {"type": "array", "uniqueItems": True,
        "items": {"type": "string", "enum": allowed_elements(bundle)}}
    interval = {"type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "integer", "minimum": 1}}
    return {"type": "object", "additionalProperties": False, "required": ["queries", "rationale"],
        "properties": {"rationale": {"type": "string", "maxLength": 200},
            "queries": {"type": "array", "minItems": 0, "maxItems": MAX_QUERIES,
            "items": {"type": "object", "additionalProperties": False, "required": ["n"],
                "properties": {"include_all": elements, "include_any": elements, "exclude": elements,
                    "n_elements": interval, "n_sites": interval,
                    "template_spacegroup": {"type":"array","minItems":2,"maxItems":2,"items":{"type":"integer","minimum":1,"maximum":230}},
                    "n": {"type": "integer", "minimum": 1, "maximum": max(1, capacity)}}}}}}


def validate(value, *, bundle=None, allowed=None, capacity=75):
    if type(capacity) is not int or capacity < 0: raise ValueError("Invalid round capacity")
    valid_elements = set(allowed if allowed is not None else allowed_elements(bundle))
    if not isinstance(value, dict) or set(value) != {"queries", "rationale"} or not isinstance(value["queries"], list):
        raise ValueError("Query response must be exactly {queries:[...],rationale:string}")
    if type(value["rationale"]) is not str or len(value["rationale"]) > 200:
        raise ValueError("Root query rationale must have at most200 Unicode characters")
    if len(value["queries"]) > 3: raise ValueError("At most3 ordered queries per round")
    total = 0
    for query in value["queries"]:
        if not isinstance(query, dict) or not set(query) <= FIELDS or "n" not in query:
            raise ValueError("Unknown query field or missing n")
        if type(query["n"]) is not int or query["n"] < 1: raise ValueError("Query n must be a positive integer")
        total += query["n"]
        for field in ("include_all", "include_any", "exclude"):
            if field not in query: continue
            items = query[field]
            if not isinstance(items, list) or any(type(x) is not str or x not in valid_elements for x in items) or len(set(items)) != len(items):
                raise ValueError("Element query requires unique POOL-whitelist element symbols")
        for field in ("n_elements", "n_sites", "template_spacegroup"):
            if field not in query: continue
            interval = query[field]
            if not isinstance(interval, list) or len(interval) != 2 or any(type(x) is not int or x < 1 for x in interval) or interval[0] > interval[1]:
                raise ValueError("Query ranges must be closed positive-integer [lower,upper] intervals")
            if field == "template_spacegroup" and interval[1] > 230: raise ValueError("Space-group interval must be within1..230")
    if total > capacity: raise ValueError("Sum of requested n exceeds75 plus carried shortfall")
    return value


def matches(query, record):
    elements = _elements(record)
    if not set(query.get("include_all", [])) <= elements: return False
    if query.get("include_any") and not elements & set(query["include_any"]): return False
    if elements & set(query.get("exclude", [])): return False
    for field in ("n_elements", "n_sites", "template_spacegroup"):
        if field in query and not query[field][0] <= record[field] <= query[field][1]: return False
    return True


def _rng(arm, seed, round_index, query_index, purpose="query"):
    token = f"H4A-query-selection-v1|{purpose}|{arm}|{seed}|{round_index}|{query_index}"
    return random.Random(int(hashlib.sha256(token.encode()).hexdigest(), 16))


def select_only_no_reveal(value, *, arm, seed, round_index, labeled_ids, bundle, carry=0):
    if arm != "L0" or type(seed) is not int or seed not in (0, 1, 2): raise ValueError("Unknown arm/seed")
    if type(round_index) is not int or not 1 <= round_index <= 20: raise ValueError("Round must be1..20")
    if type(carry) is not int or carry < 0: raise ValueError("Carry must be nonnegative integer")
    labeled = list(labeled_ids)
    if len(set(labeled)) != len(labeled) or not set(labeled) <= set(bundle["pool_ids"]):
        raise ValueError("Already labeled identities must be distinct POOL only")
    capacity = 75 + carry
    validate(value, bundle=bundle, capacity=capacity)
    remaining = set(bundle["pool_ids"]) - set(labeled)
    selected = []; audit = []
    for index, query in enumerate(value["queries"]):
        candidates = sorted(rid for rid in remaining if matches(query, bundle["by_id"][rid]))
        count = min(query["n"], len(candidates))
        chosen = _rng(arm, seed, round_index, index).sample(candidates, count)
        selected.extend(chosen); remaining.difference_update(chosen)
        audit.append({"query_index": index, "requested_n": query["n"], "matched_unlabeled_n": len(candidates),
            "selected_n": count, "shortfall_n": query["n"] - count})
    if len(set(selected)) != len(selected) or len(selected) > capacity: raise AssertionError("Selection budget/duplicate error")
    return {"selected_ids": selected, "carry_in": carry, "capacity": capacity,
        "selected_n": len(selected), "carry_out": capacity - len(selected), "queries": audit,
        "automatic_topup": False, "labels_revealed": False, "grammar_version": GRAMMAR_VERSION}


def final_fill_only_no_reveal(*, arm, seed, labeled_ids, bundle, total_new_revealed):
    if type(total_new_revealed) is not int or not 0 <= total_new_revealed <= 1500: raise ValueError("Invalid revealed budget")
    if arm != "L0" or type(seed) is not int or seed not in (0, 1, 2): raise ValueError("Unknown arm/seed")
    labeled = list(labeled_ids)
    if len(set(labeled)) != len(labeled) or not set(labeled) <= set(bundle["pool_ids"]): raise ValueError("Final fill must remain inside POOL")
    if len(labeled) != 500 + total_new_revealed: raise ValueError("Final fill ledger cardinality differs from S0+reveals")
    count = 1500 - total_new_revealed
    remaining = sorted(set(bundle["pool_ids"]) - set(labeled))
    if len(remaining) < count: raise ValueError("Not enough remaining POOL for final fill")
    chosen = _rng(arm, seed, 21, 0, "final_uniform_fill").sample(remaining, count)
    return {"selected_ids": chosen, "selected_n": count, "labels_revealed": False,
        "after_round": 20, "extra_notes_update": False, "method": "uniform without replacement on remaining POOL; fixed arm/seed RNG"}


def catalogue(labeled_ids, *, bundle):
    labeled = list(labeled_ids)
    if len(set(labeled)) != len(labeled) or not set(labeled) <= set(bundle["pool_ids"]): raise ValueError("Catalogue labeled IDs must be POOL only")
    ids = sorted(set(bundle["pool_ids"]) - set(labeled))
    elements = Counter(); counts = Counter(); sites = []; spacegroups = Counter()
    for rid in ids:
        row = bundle["by_id"][rid]; elements.update(_elements(row)); counts[row["n_elements"]] += 1; sites.append(row["n_sites"]); spacegroups[row["template_spacegroup"]] += 1
    return {"unlabeled_POOL_n": len(ids), "element_material_counts": dict(sorted(elements.items())),
        "n_elements_distribution": {str(k): v for k, v in sorted(counts.items())},
        "n_sites_quantiles_0_25_50_75_100": np.quantile(sites, [0, .25, .5, .75, 1]).tolist() if sites else [],
        "template_spacegroup_distribution": {str(k):v for k,v in sorted(spacegroups.items())},
        "label_values_accessed": False, "TEST_or_HOLDOUT_accessed": False}


def grammar_text(capacity):
    return (f"queries:0..3 ordered; each n>=1 integer, sum<={capacity}. "
        "Optional unique POOL symbols:include_all=all,include_any=one if nonempty,exclude=none. "
        "n_elements,n_sites:closed positive[lo,hi]; template_spacegroup:closed1..230 from POOL final_structure. "
        "Conditions AND; unlabeled-only. Fixed arm/seed/round/query RNG draws min(n,matches), no repeats across queries. "
        "All unused quota carries, no round topup. After20 uniformly fill budget to2000, no extra feature/notes update. "
        "Invalid query:one correction then[] if invalid; root:features,queries,rationale<=200 Unicode chars.")


# Pure schema/null compatibility copied from frozen H4A; no transport import.
from copy import deepcopy

WIRE_NOTE = ("\nProvider JSON wire-format compatibility only: every field shown in the response schema is required "
             "on the wire. For an optional query condition you do not want to restrict, return null; null means "
             "omit that optional condition. Null does not change required fields or the query quota. "
             "Array items must still be unique when the original query grammar requires uniqueness.\n")

def strict_schema(canonical_schema):
    """Create a provider-only copy, without relaxing canonical validation."""
    if isinstance(canonical_schema, bool):
        return canonical_schema
    if not isinstance(canonical_schema, dict):
        raise ValueError("Canonical schema must be an object or boolean")
    result = {}
    for key, value in canonical_schema.items():
        if key == "uniqueItems":
            continue
        if key in ("properties", "$defs", "definitions"):
            result[key] = {name: strict_schema(child) for name, child in value.items()}
        elif key in ("items", "additionalProperties", "not") and isinstance(value, (dict, bool)):
            result[key] = strict_schema(value)
        elif key in ("anyOf", "oneOf", "allOf", "prefixItems"):
            result[key] = [strict_schema(child) for child in value]
        else:
            result[key] = deepcopy(value)
    if "properties" in canonical_schema:
        required = set(canonical_schema.get("required", []))
        optional = set(canonical_schema["properties"]) - required
        for name in canonical_schema["properties"]:
            if name not in required:
                result["properties"][name] = {"anyOf": [result["properties"][name], {"type": "null"}]}
        if optional:
            result["required"] = list(canonical_schema["properties"])
            result.setdefault("additionalProperties", False)
    return result


def normalize_optional_nulls(value, schema):
    """Only optional named fields may be omitted; required nulls stay invalid."""
    if not isinstance(schema, dict):
        return deepcopy(value)
    if isinstance(value, dict):
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        result = {}
        for name, child in value.items():
            if name in properties and name not in required and child is None:
                continue
            child_schema = properties.get(name, schema.get("additionalProperties", True))
            result[name] = normalize_optional_nulls(child, child_schema)
        return result
    if isinstance(value, list):
        prefix = schema.get("prefixItems", [])
        return [normalize_optional_nulls(child, prefix[i] if i < len(prefix) else schema.get("items", True))
                for i, child in enumerate(value)]
    return deepcopy(value)


