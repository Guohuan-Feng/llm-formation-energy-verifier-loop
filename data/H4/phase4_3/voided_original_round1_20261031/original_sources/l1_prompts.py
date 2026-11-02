"""New stage4.3 L1 prompt/schema; frozen L0 modules stay unchanged.

All L0 current-code, validation, summary and catalogue content is retained.
The canonical memory JSON and <=300-character response note are the additions.
This module never loads truth, executes a feature, fits or calls a model.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from copy import deepcopy
import hashlib
import json
import math

import h4_l0_prompts as l0
import h4_l0_queries as queries
import h4_prompts as feature_contract
import l1_counter as counter
import l1_memory as memory_module

VERSION = "H4-stage4.3-L1-memory-features-and-queries-v1"
INSTRUCTIONS = feature_contract.INSTRUCTIONS
FORMAT_REPAIR = feature_contract.FORMAT_REPAIR
PREFIX = (
    "Improve HGB descriptors from current validation feedback/summaries and the TRAIN/validation memory below. "
    "Return JSON {features,queries,rationale,notes}: new complete1..20 {name,description,source}, not a patch; "
    "rationale<=200 Unicode chars; notes<=300 Unicode chars, one concise scientific note, no reasoning steps. "
    "Initial []=Magpie-only. Current set=previous model's retained inner-fit features (>20% failures dropped); "
    "original code is complete, never budget-truncated. No external held-out feedback.\n")
MEMORY_LABEL = "TRAIN/validation memory (canonical JSON; complete definitions referenced by source SHA256):\n"
MEMORY_EXPLANATION = (
    "Memory delta_without_minus_full=validation rowMAE without that feature minus full-model validation rowMAE; "
    "positive means helpful. None means unavailable (including dropped features). Definitions contain each exact "
    "original source once; feature/failure records reference its SHA256. Query statistics and all historical notes "
    "are retained. If the frozen6000-token view was trimmed, it keeps the30 highest latest-contribution unique "
    "features plus every record from the latest3 rounds. Memory contains only this arm's TRAIN/validation feedback.\n")
DICTIONARY_CONTRACT = l0.DICTIONARY_CONTRACT
NUMERIC_FORMAT = deepcopy(l0.NUMERIC_FORMAT)


class PromptBudgetExceeded(ValueError):
    def __init__(self, budget):
        self.budget = budget
        super().__init__(f"Complete unchanged L1 request needs {budget['input_token_upper_bound']} conservative "
            f"input units; frozen input limit is {budget['input_limit']}; stop without shortening memory/source or calling a model")


def _json(value):
    return counter.canonical(value)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def response_schema(bundle, capacity=75):
    result = deepcopy(l0.response_schema(bundle, capacity))
    result["properties"]["notes"] = {"type": "string", "maxLength": memory_module.NOTES_LIMIT}
    result["required"] = ["features", "queries", "rationale", "notes"]
    return result


def canonical_schema_bytes(schema):
    return (_json(schema) + "\n").encode("utf-8")


def wire_schema(schema):
    return queries.strict_schema(schema)


def audit_input_bound(prompt, schema, *, input_limit, include_repair=False, measurement=False):
    """Keep the original whole-request conservative UTF8 guard, separately from memory tokens."""
    if type(prompt) is not str or type(input_limit) is not int or input_limit <= 0:
        raise ValueError("Prompt text and an explicitly frozen positive integer input_limit are required")
    wire = wire_schema(schema)
    changed = _json(wire) != _json(schema)
    actual = prompt + (FORMAT_REPAIR if include_repair else "") + (queries.WIRE_NOTE if changed else "")
    fixed = (feature_contract._global_instruction_bytes() + len(INSTRUCTIONS.encode("utf-8"))
        + len(canonical_schema_bytes(wire)))
    bound = len(actual.encode("utf-8")) + fixed + 1500
    result = {"method": "utf8_byte_upper_bound_plus_1500_framing", "input_limit": input_limit,
        "canonical_prompt_bytes": len(prompt.encode("utf-8")), "wire_prompt_bytes": len(actual.encode("utf-8")),
        "known_fixed_context_bytes": fixed, "input_token_upper_bound": bound,
        "strict_wire_note_included": changed, "format_repair_included": bool(include_repair),
        "within_budget": bound <= input_limit, "measurement_only": bool(measurement)}
    if not result["within_budget"] and not measurement: raise PromptBudgetExceeded(result)
    return result


def _checked_memory(value, *, private_ids, measurement=False):
    if isinstance(value, list):
        result = memory_module.build_memory(value, private_ids=private_ids, measurement=measurement)
    else:
        if not isinstance(value, dict) or "payload" not in value:
            raise ValueError("memory must be completed-round history or a build_memory result")
        payload = deepcopy(value["payload"])
        memory_module.validate_payload(payload, private_ids=private_ids)
        measured = counter.measure(payload)
        if "text" in value and value["text"] != measured["text"]:
            raise ValueError("Memory text differs from its canonical payload")
        if "token_count" in value and value["token_count"] != measured["token_count"]:
            raise ValueError("Memory token count differs from the frozen counter")
        if "UTF8_bytes" in value and value["UTF8_bytes"] != measured["UTF8_bytes"]:
            raise ValueError("Memory UTF8 byte count differs")
        result = {**deepcopy(value), **measured,
            "overflow": measured["token_count"] > memory_module.LIMIT,
            "limit_tokens": memory_module.LIMIT, "payload_sha256": _sha(measured["text"])}
    if result["overflow"] and not measurement: raise memory_module.MemoryBudgetExceeded(result)
    if not measurement and any("query_summary" not in row for row in result["payload"]["rounds"]):
        raise ValueError("Actual L1 memory needs every round's complete query/carry summary")
    return result


def build_request(seed, round_index, features, validation_mae, worst20, labels, memory, *,
        input_limit, carry=0, bundle=None, measurement=False):
    b = l0.data._bundle(bundle)
    l0.data.validate_labels(labels, seed, bundle=b)
    if type(round_index) is not int or not 1 <= round_index <= 20:
        raise ValueError("L1 round must be1..20")
    if type(carry) is not int or carry < 0 or 75 + carry > 1500:
        raise ValueError("Invalid carried quota")
    current = l0._feature_block(features)
    val_mae = l0._finite(validation_mae)
    if val_mae < 0: raise ValueError("Validation MAE must be nonnegative")
    worst = l0._worst_rows(worst20, labels, seed, b)
    capacity = 75 + carry
    summary = l0.labelled_summary(labels, seed, bundle=b)
    catalogue = queries.catalogue(labels, bundle=b)
    schema = response_schema(b, capacity)
    mem = _checked_memory(memory, private_ids=b["ids"], measurement=measurement)
    completed = mem["payload"]["rounds"]
    if len(completed) != round_index - 1 or [r["round"] for r in completed] != list(range(1, round_index)):
        raise ValueError("L1 prompt memory must contain exactly this seed's preceding completed rounds")
    grammar = queries.grammar_text(capacity).replace("root:features,queries,rationale<=200 Unicode chars.",
        "root:features,queries,rationale<=200 Unicode chars,notes<=300 Unicode chars.")
    prompt = (PREFIX + DICTIONARY_CONTRACT + grammar + "\n"
        + "Active code: @i n=name d=description B=source_UTF8_bytes; full original source follows each header; @end closes:\n"
        + l0._current_text(current) + "\n"
        + "Current validation rowMAE eV/atom: " + l0._number(val_mae, NUMERIC_FORMAT["feedback"]) + "\n"
        + "Current worst validation rows (TSV; largest absolute errors first):\n" + l0._worst_text(worst) + "\n"
        + "Current labeled-set summary:\n" + l0._summary_text(summary) + "\n"
        + "Unlabeled POOL catalogue (no target values):\n" + l0._catalogue_text(catalogue) + "\n"
        + MEMORY_EXPLANATION + MEMORY_LABEL + mem["text"] + "\n"
        + "Return only final JSON with features,queries,rationale,notes.\n")
    memory_module._no_private_ids(prompt, b["ids"])
    ordinary = audit_input_bound(prompt, schema, input_limit=input_limit, measurement=measurement)
    repaired = audit_input_bound(prompt, schema, input_limit=input_limit, include_repair=True, measurement=measurement)
    return {"request_id": f"L1_seed{seed}_round{round_index:02d}", "kind": "features_query_memory",
        "stage": "4.3", "arm": "L1", "seed": seed, "round_index": round_index,
        "prompt": prompt, "schema": schema, "instructions": INSTRUCTIONS,
        "max_format_repairs": 1, "admissible_format_failure": True,
        "canonical_prompt_sha256": _sha(prompt), "schema_sha256": _sha(canonical_schema_bytes(schema)),
        "wire_schema_sha256": _sha(canonical_schema_bytes(wire_schema(schema))),
        "input_budget": ordinary, "format_repair_input_budget": repaired,
        "memory_payload_sha256": mem["payload_sha256"], "memory_token_count": mem["token_count"],
        "memory_UTF8_bytes": mem["UTF8_bytes"], "memory_trimmed": mem.get("trimmed", False),
        "memory_overflow": mem["overflow"], "memory_history_rounds": len(completed),
        "current_features_sha256": l0.data.json_hash(current), "current_feature_count": len(current),
        "current_complete_source_bytes": sum(len(f["source"].encode("utf-8")) for f in current),
        "labeled_ids_sha256": l0.data.json_hash(sorted(labels)), "worst20_ids": [r["id"] for r in worst20],
        "summary_sha256": l0.data.json_hash(summary), "catalogue_sha256": l0.data.json_hash(catalogue),
        "carry": carry, "capacity": capacity, "measurement_only": bool(measurement)}


def _features_and_notes(value):
    if not isinstance(value, dict): raise ValueError("L1 response must be a JSON object")
    feature_contract.validator({"features": value.get("features")})
    notes = value.get("notes")
    if type(notes) is not str or len(notes) > memory_module.NOTES_LIMIT:
        raise ValueError("L1 note must be <=300 Unicode characters; never silently truncate it")
    memory_module._no_private_ids({"features": value["features"], "notes": notes})
    return deepcopy(value["features"]), notes


def validate_response(payload, *, bundle, capacity):
    if not isinstance(payload, dict) or set(payload) != {"features", "queries", "rationale", "notes"}:
        raise ValueError("L1 response must contain exactly features,queries,rationale,notes")
    _features_and_notes(payload)
    queries.validate({"queries": payload["queries"], "rationale": payload["rationale"]}, bundle=bundle, capacity=capacity)
    memory_module._no_private_ids(payload, bundle["ids"])
    return True


def validator(bundle, capacity):
    return lambda value: validate_response(value, bundle=bundle, capacity=capacity)


def resolve_response(result, *, bundle, capacity):
    """Preserve valid original features AND notes on final query-format failure.

    Result must explicitly identify a final format-invalid response before the
    empty-query fallback is allowed. A feature/note failure is a hard stop.
    """
    if not isinstance(result, dict): raise ValueError("Transport result must be an object")
    fallback_allowed = (result.get("last_status") == "format_invalid" and result.get("admissible_format_failure") is True)
    if result.get("status") != "ok" and not fallback_allowed:
        raise ValueError("L1 transport failure cannot use a query-only fallback")
    payload = result.get("response")
    if payload is None: payload = result.get("parsed_response", result.get("payload"))
    if payload is None:
        raw = result.get("raw_response_text", result.get("raw_response", result.get("response_text")))
        if not isinstance(raw, str): raise ValueError("No complete response JSON is available")
        def reject_constant(value): raise ValueError("Non-finite JSON constant: " + value)
        payload = json.loads(raw, parse_constant=reject_constant)
    payload = queries.normalize_optional_nulls(payload, response_schema(bundle, capacity))
    features, notes = _features_and_notes(payload)
    try:
        validate_response(payload, bundle=bundle, capacity=capacity)
        return {"features": features, "queries": deepcopy(payload["queries"]),
            "rationale": payload["rationale"], "notes": notes}, {"query_fallback": False, "notes_preserved": True}
    except ValueError as error:
        if not fallback_allowed:
            raise
        # Unrelated root-field additions are not relabelled as a query error.
        if not set(payload) <= {"features", "queries", "rationale", "notes"}:
            raise ValueError("Unexpected response fields cannot use query-only fallback") from error
        return {"features": features, "queries": [], "rationale": "", "notes": notes}, {
            "query_fallback": True, "query_error": str(error), "notes_preserved": True,
            "features_preserved": True, "admissible_format_failure": True}


def configuration(*, input_limit, bundle=None):
    if type(input_limit) is not int or input_limit <= 0: raise ValueError("Explicit frozen input_limit required")
    b = l0.data._bundle(bundle)
    schema = response_schema(b, 75)
    return {"protocol_version": VERSION, "stage": "4.3", "condition": "L1",
        "prefix": PREFIX, "prefix_sha256": _sha(PREFIX),
        "dictionary_contract": DICTIONARY_CONTRACT, "dictionary_contract_sha256": _sha(DICTIONARY_CONTRACT),
        "active_feature_policy": l0.ACTIVE_FEATURE_POLICY.replace("L0 does not replay dropped-feature history", "L1 replays prescribed memory history")
            .replace("20000 conservative input bound", "configured conservative input bound"),
        "initial_current_features": [], "feature_schema": deepcopy(feature_contract.FEATURE_SCHEMA),
        "schema": schema, "wire_schema": wire_schema(schema),
        "schema_sha256": _sha(canonical_schema_bytes(schema)),
        "wire_schema_sha256": _sha(canonical_schema_bytes(wire_schema(schema))),
        "round_schema_rule": "n maximum1500 is constant; semantic query total<=75+carry; notes maxLength300; other H3/H4 feature and H4A query grammar unchanged",
        "memory": memory_module.configuration(), "memory_label": MEMORY_LABEL, "memory_explanation": MEMORY_EXPLANATION,
        "memory_bytes_are_part_of_whole_request_guard": True,
        "whole_request_counter_is_not_the_memory_token_counter": True,
        "input_limit": input_limit, "input_budget_method": "Actual UTF8 strict wire schema+wire note+global/model instructions+1500 framing; ordinary and repair checked",
        "format_repair_suffix": FORMAT_REPAIR, "format_repair_suffix_sha256": _sha(FORMAT_REPAIR),
        "wire_note": queries.WIRE_NOTE, "wire_note_sha256": _sha(queries.WIRE_NOTE),
        "instructions": INSTRUCTIONS, "instructions_sha256": _sha(INSTRUCTIONS),
        "numeric_format": NUMERIC_FORMAT, "notes_max_unicode_characters": memory_module.NOTES_LIMIT,
        "max_format_repairs": 1, "query_only_final_invalid_fallback": "Keep exact valid features and notes; queries=[],rationale=''; raw response stays in ledger",
        "feature_or_note_invalid_after_repair": "Stop; never replace with empty feature set or truncate notes",
        "memory_or_full_input_overflow_action": "Stop entire stage unchanged before model call; report; no silent truncation",
        "feedback_scope": "This seed's labeled TRAIN/internal validation only; no TEST/HOLDOUT feedback"}


round_request = build_request
