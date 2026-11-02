"""Stage4.3 TRAIN/validation memory: lossless archive and deterministic prompt view.

This module never loads a dataset, calls a model, fits or executes a feature.
Only the explicitly requested private archive writer performs filesystem writes.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

import l1_counter as counter

VERSION = "H4-stage4.3-L1-memory-v1"
LIMIT = 6000
NOTES_LIMIT = 300
TOP_FEATURES = 30
RECENT_ROUNDS = 3
SCOPE = "TRAIN_validation_only"
PHASE = Path(__file__).resolve().parent.parent
POLICY = (
    "Archive every complete original definition, LOFO contribution, aggregated TRAIN/validation failure, "
    "ordered query payload/statistics and <=300-Unicode-character note. Prompt JSON stores each original "
    "source once under its SHA256 and references it from records. Measure the entire canonical JSON using "
    "the frozen local counter, excluding only the outer prompt label. If it exceeds6000 tokens retain the "
    "30 unique source definitions ranked by latest delta_without_minus_full descending (None last; ties "
    "first_seen_round,first_seen_feature_index,source_sha256), plus all four record types from the latest3 "
    "rounds and every historical note/query statistic. Never truncate source or notes. If that single "
    "deterministic view still exceeds6000 tokens, stop the entire stage before a model call.")
_PRIVATE_PATTERN = re.compile(r"\bH2-[A-Za-z]-\d+\b")


class MemoryBudgetExceeded(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Complete deterministic L1 memory needs {result['token_count']} tokens after "
            f"the prescribed trim; frozen limit is {LIMIT}; no further truncation or call is permitted")


def source_hash(source):
    if type(source) is not str:
        raise ValueError("Feature source must be Unicode text")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _integer(value, name, lower=0):
    if type(value) is not int or value < lower:
        raise ValueError(name + " must be an integer >=" + str(lower))
    return value


def _finite_or_none(value):
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("LOFO contribution must be a finite scalar or None")
    return float(value)


def _no_private_ids(value, private_ids=()):
    """Check content, never silently remove or shorten it."""
    forbidden = tuple(private_ids)
    if any(type(x) is not str or not x for x in forbidden):
        raise ValueError("Private identities must be nonempty strings")
    exact_pattern = re.compile("|".join(re.escape(rid) for rid in sorted(set(forbidden), key=lambda x: (-len(x), x)))) if forbidden else None
    def walk(item):
        if isinstance(item, dict):
            for key, child in item.items():
                walk(key); walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item: walk(child)
        elif isinstance(item, str):
            if _PRIVATE_PATTERN.search(item) or (exact_pattern is not None and exact_pattern.search(item)):
                raise ValueError("A private material identifier entered L1 memory")
    walk(value)


def _features(value):
    # The frozen original format validator deliberately does not execute/repair ASTs.
    import h4_prompts
    h4_prompts.validator({"features": value})
    return deepcopy(value)


def _query(value):
    fields = {"n", "include_all", "include_any", "exclude", "n_elements", "n_sites", "template_spacegroup"}
    if not isinstance(value, dict) or "n" not in value or not set(value) <= fields:
        raise ValueError("Memory query contains a noncanonical field")
    _integer(value["n"], "query n", 1)
    if value["n"] > 1500:
        raise ValueError("Memory query exceeds the total1500 label budget")
    for field in ("include_all", "include_any", "exclude"):
        if field not in value: continue
        xs = value[field]
        if (not isinstance(xs, list) or any(type(x) is not str or not re.fullmatch(r"[A-Z][a-z]?", x) for x in xs)
                or len(xs) != len(set(xs))):
            raise ValueError("Memory query elements must be unique canonical symbols")
    for field in ("n_elements", "n_sites", "template_spacegroup"):
        if field not in value: continue
        xs = value[field]
        if (not isinstance(xs, list) or len(xs) != 2 or any(type(x) is not int or x < 1 for x in xs)
                or xs[0] > xs[1] or (field == "template_spacegroup" and xs[1] > 230)):
            raise ValueError("Memory query range is invalid")
    return deepcopy(value)


def normalize_history(history, *, private_ids=()):
    """Strict ID-free archive contract; extra/raw model fields cannot enter memory."""
    if not isinstance(history, list) or len(history) > 20:
        raise ValueError("Memory history must be a list of at most20 completed rounds")
    required = {"round_index", "features", "contributions", "failures", "queries", "notes"}
    optional = {"query_summary", "rationale"}
    result = []
    previous = 0
    for original in history:
        if not isinstance(original, dict) or not required <= set(original) or not set(original) <= required | optional:
            raise ValueError("Unexpected/missing memory history fields")
        rnd = _integer(original["round_index"], "round_index", 1)
        if rnd != previous + 1 or rnd > 20:
            raise ValueError("Memory history must contain consecutive completed rounds starting at1")
        previous = rnd
        features = _features(original["features"])
        count = len(features)
        contributions = original["contributions"]
        if not isinstance(contributions, list) or len(contributions) != count:
            raise ValueError("Every raw feature must have one LOFO contribution (None if unavailable)")
        by_index = {}
        for item in contributions:
            if not isinstance(item, dict) or set(item) != {"feature_index", "delta_without_minus_full"}:
                raise ValueError("Unexpected LOFO fields")
            index = _integer(item["feature_index"], "feature_index")
            if index >= count or index in by_index: raise ValueError("Duplicate/unknown LOFO feature")
            by_index[index] = _finite_or_none(item["delta_without_minus_full"])
        failures = []
        if not isinstance(original["failures"], list): raise ValueError("Failure records must be a list")
        for item in original["failures"]:
            if not isinstance(item, dict) or set(item) != {"feature_index", "phase", "failed_count", "total_count", "reasons"}:
                raise ValueError("Unexpected failure fields; sample rows/identities are not allowed")
            index = _integer(item["feature_index"], "failed feature index")
            failed = _integer(item["failed_count"], "failed count")
            total = _integer(item["total_count"], "total count", 1)
            if index >= count or failed > total or item["phase"] not in {"inner_fit", "validation", "full_train"}:
                raise ValueError("Failure memory may contain only this arm's TRAIN/validation aggregates")
            reasons = item["reasons"]
            if not isinstance(reasons, dict) or any(type(k) is not str or not k for k in reasons):
                raise ValueError("Failure reasons must be aggregated reason strings")
            if any(type(v) is not int or v < 0 for v in reasons.values()) or sum(reasons.values()) != failed:
                raise ValueError("Failure reason counts must sum to failed_count")
            failures.append(deepcopy(item))
        queries = []
        if not isinstance(original["queries"], list) or len(original["queries"]) > 3:
            raise ValueError("A completed round has at most3 ordered queries")
        qfields = {"query", "requested_count", "matched_count", "revealed_count", "carry_after"}
        for item in original["queries"]:
            if not isinstance(item, dict) or set(item) != qfields:
                raise ValueError("Memory queries contain only payload and aggregate match/reveal/carry statistics")
            clean = {"query": _query(item["query"])}
            for key in qfields - {"query"}: clean[key] = _integer(item[key], key)
            if (clean["requested_count"] != clean["query"]["n"]
                    or clean["revealed_count"] > min(clean["requested_count"], clean["matched_count"])):
                raise ValueError("Memory query statistics disagree with its payload")
            queries.append(clean)
        notes = original["notes"]
        if type(notes) is not str or len(notes) > NOTES_LIMIT:
            raise ValueError("Each complete L1 note must be <=300 Unicode characters; no truncation")
        clean = {"round_index": rnd, "features": features,
            "contributions": [{"feature_index": i, "delta_without_minus_full": by_index[i]} for i in range(count)],
            "failures": failures, "queries": queries, "notes": notes}
        if "rationale" in original:
            if type(original["rationale"]) is not str or len(original["rationale"]) > 200:
                raise ValueError("Memory rationale must respect the unchanged200-character limit")
            clean["rationale"] = original["rationale"]
        if "query_summary" in original:
            summary = original["query_summary"]
            keys = {"capacity", "carry_in", "carry_out", "requested_count", "matched_count", "revealed_count"}
            if not isinstance(summary, dict) or set(summary) != keys:
                raise ValueError("Unexpected query summary fields")
            summary = {k: _integer(summary[k], k) for k in keys}
            if (summary["capacity"] != 75 + summary["carry_in"] or summary["capacity"] > 1500
                    or summary["requested_count"] != sum(q["requested_count"] for q in queries)
                    or summary["matched_count"] != sum(q["matched_count"] for q in queries)
                    or summary["revealed_count"] != sum(q["revealed_count"] for q in queries)
                    or summary["requested_count"] > summary["capacity"]
                    or summary["carry_out"] != summary["capacity"] - summary["revealed_count"]):
                raise ValueError("Round query summary is inconsistent")
            clean["query_summary"] = summary
        result.append(clean)
    _no_private_ids(result, private_ids)
    return {"version": VERSION, "scope": SCOPE, "rounds": result}


def _view(archive):
    definitions = {}
    records = []
    first_seen = {}
    latest = {}
    for item in archive["rounds"]:
        rnd = item["round_index"]
        refs = [source_hash(f["source"]) for f in item["features"]]
        contributions = {x["feature_index"]: x["delta_without_minus_full"] for x in item["contributions"]}
        features = []
        for index, feature in enumerate(item["features"]):
            ref = refs[index]
            if ref in definitions and definitions[ref]["source"] != feature["source"]:
                raise ValueError("Feature source SHA256 collision")
            definitions[ref] = {"source": feature["source"]}
            first_seen.setdefault(ref, (rnd, index, ref))
            latest[ref] = contributions[index]
            features.append({"ref": ref, "name": feature["name"], "description": feature["description"],
                "delta_without_minus_full": contributions[index]})
        failures = [{"ref": refs[x["feature_index"]], **{k: deepcopy(v) for k, v in x.items() if k != "feature_index"}}
            for x in item["failures"]]
        record = {"round": rnd, "features": features, "failures": failures,
            "queries": deepcopy(item["queries"]), "notes": item["notes"]}
        for key in ("query_summary", "rationale"):
            if key in item: record[key] = deepcopy(item[key])
        records.append(record)
    ranking = sorted(definitions, key=lambda ref: (
        latest[ref] is None, -latest[ref] if latest[ref] is not None else 0, *first_seen[ref]))
    return {"scope": SCOPE, "definitions": definitions, "rounds": records}, ranking


def validate_payload(payload, *, private_ids=()):
    """Check an already-built prompt view without interpreting or executing code."""
    if not isinstance(payload, dict) or set(payload) != {"scope", "definitions", "rounds"} or payload["scope"] != SCOPE:
        raise ValueError("Unexpected memory payload/scope")
    if not isinstance(payload["definitions"], dict) or not isinstance(payload["rounds"], list):
        raise ValueError("Invalid memory definitions/round records")
    refs = set()
    for ref, definition in payload["definitions"].items():
        if (not isinstance(definition, dict) or set(definition) != {"source"}
                or source_hash(definition["source"]) != ref):
            raise ValueError("Memory definition source/hash differs")
    if len(payload["rounds"]) > 20: raise ValueError("Too many memory rounds")
    for position, record in enumerate(payload["rounds"], 1):
        if (not isinstance(record, dict) or not {"round", "features", "failures", "queries", "notes"} <= set(record)
                or not set(record) <= {"round", "features", "failures", "queries", "notes", "query_summary", "rationale"}):
            raise ValueError("Unexpected memory record fields")
        if type(record["round"]) is not int or record["round"] != position:
            raise ValueError("Memory rounds must be consecutive from1")
        if type(record["notes"]) is not str or len(record["notes"]) > NOTES_LIMIT:
            raise ValueError("Invalid historical note")
        if "rationale" in record and (type(record["rationale"]) is not str or len(record["rationale"]) > 200):
            raise ValueError("Invalid historical rationale")
        if not isinstance(record["features"], list) or len(record["features"]) > 20:
            raise ValueError("Invalid feature reference list")
        round_refs = set()
        names = set()
        for feature in record["features"]:
            if not isinstance(feature, dict) or set(feature) != {"ref", "name", "description", "delta_without_minus_full"}:
                raise ValueError("Unexpected feature-reference fields")
            if feature["ref"] not in payload["definitions"]: raise ValueError("Dangling memory definition reference")
            if feature["ref"] in round_refs or feature["name"] in names:
                raise ValueError("Duplicate current-round feature source or name")
            round_refs.add(feature["ref"]); names.add(feature["name"])
            refs.add(feature["ref"])
            _features([{k: feature[k] for k in ("name", "description")} | {"source": payload["definitions"][feature["ref"]]["source"]}])
            _finite_or_none(feature["delta_without_minus_full"])
        if not isinstance(record["failures"], list): raise ValueError("Invalid failure record list")
        for failure in record["failures"]:
            if not isinstance(failure, dict) or set(failure) != {"ref", "phase", "failed_count", "total_count", "reasons"}:
                raise ValueError("Unexpected failure-reference fields")
            if failure["ref"] not in round_refs or failure["phase"] not in {"inner_fit", "validation", "full_train"}:
                raise ValueError("Unknown feature or disallowed failure scope")
            failed = _integer(failure["failed_count"], "failed_count")
            total = _integer(failure["total_count"], "total_count", 1)
            reasons = failure["reasons"]
            if (failed > total or not isinstance(reasons, dict) or any(type(k) is not str or not k for k in reasons)
                    or any(type(v) is not int or v < 0 for v in reasons.values()) or sum(reasons.values()) != failed):
                raise ValueError("Invalid aggregate failure statistics")
            refs.add(failure["ref"])
        if not isinstance(record["queries"], list) or len(record["queries"]) > 3:
            raise ValueError("Invalid query record list")
        for item in record["queries"]:
            if not isinstance(item, dict) or set(item) != {"query", "requested_count", "matched_count", "revealed_count", "carry_after"}:
                raise ValueError("Unexpected query record fields")
            _query(item["query"])
            for key in ("requested_count", "matched_count", "revealed_count", "carry_after"):
                _integer(item[key], key)
            if item["requested_count"] != item["query"]["n"] or item["revealed_count"] > min(item["requested_count"], item["matched_count"]):
                raise ValueError("Invalid query statistics")
        if "query_summary" in record:
            summary = record["query_summary"]
            keys = {"capacity", "carry_in", "carry_out", "requested_count", "matched_count", "revealed_count"}
            if not isinstance(summary, dict) or set(summary) != keys:
                raise ValueError("Unexpected query summary fields")
            for key in keys: _integer(summary[key], key)
            if (summary["capacity"] != 75 + summary["carry_in"] or summary["capacity"] > 1500
                    or summary["requested_count"] != sum(q["requested_count"] for q in record["queries"])
                    or summary["matched_count"] != sum(q["matched_count"] for q in record["queries"])
                    or summary["revealed_count"] != sum(q["revealed_count"] for q in record["queries"])
                    or summary["requested_count"] > summary["capacity"]
                    or summary["carry_out"] != summary["capacity"] - summary["revealed_count"]):
                raise ValueError("Inconsistent memory query summary")
    if refs != set(payload["definitions"]): raise ValueError("Unreferenced or missing memory definition")
    _no_private_ids(payload, private_ids)
    counter.canonical(payload)  # Reject NaN or non-JSON values, never coerce them.
    return True


def build_memory(history, *, measurement=False, private_ids=()):
    """Return a deterministic memory view plus its complete lossless private archive.

    measurement=True reports prescribed-trim overflow for a clearly labelled
    local surrogate; actual runtime defaults to raising MemoryBudgetExceeded.
    """
    archive = normalize_history(history, private_ids=private_ids)
    if not measurement and any("query_summary" not in r for r in archive["rounds"]):
        raise ValueError("Actual L1 memory requires every round's query summary, including empty-query carry")
    payload, ranking = _view(archive)
    full = counter.measure(payload)
    trimmed = full["token_count"] > LIMIT
    recent = {x["round"] for x in payload["rounds"][-RECENT_ROUNDS:]}
    top = set(ranking[:TOP_FEATURES])
    if trimmed:
        records = []
        refs = set()
        for old in payload["rounds"]:
            new = deepcopy(old)
            if old["round"] not in recent:
                new["features"] = [x for x in new["features"] if x["ref"] in top]
                new["failures"] = [x for x in new["failures"] if x["ref"] in top]
            refs.update(x["ref"] for x in new["features"])
            refs.update(x["ref"] for x in new["failures"])
            records.append(new)
        payload = {"scope": SCOPE, "definitions": {ref: payload["definitions"][ref] for ref in sorted(refs)}, "rounds": records}
    validate_payload(payload, private_ids=private_ids)
    measured = counter.measure(payload)
    result = {"payload": payload, **measured, "trimmed": trimmed,
        "overflow": measured["token_count"] > LIMIT, "limit_tokens": LIMIT,
        "archive": archive, "archive_sha256": source_hash(counter.canonical(archive)),
        "payload_sha256": source_hash(measured["text"]),
        "audit": {"history_rounds": len(history), "unique_source_definitions_full": len(ranking),
            "unique_source_definitions_retained": len(payload["definitions"]),
            "untrimmed_token_count": full["token_count"], "untrimmed_UTF8_bytes": full["UTF8_bytes"],
            "top_feature_hashes": ranking[:TOP_FEATURES] if trimmed else [],
            "recent_rounds_preserved": sorted(recent), "historical_notes_preserved": len(history),
            "historical_query_records_preserved": sum(len(x["queries"]) for x in history),
            "measurement_only": bool(measurement), "source_or_note_truncation": False}}
    if result["overflow"] and not measurement: raise MemoryBudgetExceeded(result)
    return result


def persist_archive(path, history, *, private_ids=()):
    """Explicit append-free, exclusive private archive; build_memory never writes."""
    destination = Path(path).resolve()
    if "onedrive" in str(destination).casefold() or not destination.is_relative_to(PHASE.resolve()):
        raise ValueError("Memory archive must stay inside non-OneDrive phase4_3")
    archive = normalize_history(history, private_ids=private_ids)
    encoded = (counter.canonical(archive) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir so a newly substituted directory link is rejected.
    if destination.resolve() != destination or "onedrive" in str(destination.resolve()).casefold():
        raise ValueError("Memory archive destination changed")
    with destination.open("xb") as stream: stream.write(encoded)
    return {"path": str(destination), "sha256": hashlib.sha256(encoded).hexdigest(), "rounds": len(history)}


def configuration():
    return {"version": VERSION, "scope": SCOPE, "limit_tokens": LIMIT, "notes_max_unicode_characters": NOTES_LIMIT,
        "canonical_serialization": "l1_counter.canonical; sorted compact ensure_asciiFalse JSON, no trailing newline",
        "token_counter": "l1_counter.measure: pinned tiktoken0.12.0/o200k_base.encode_ordinary",
        "policy": POLICY, "top_unique_features": TOP_FEATURES, "recent_complete_rounds": RECENT_ROUNDS,
        "source_identity": "SHA256(original source UTF8); one full definition per unique source; records reference its hash",
        "contribution_sign": "validation_rowMAE_without_this_feature minus validation_rowMAE_full; positive is helpful",
        "ranking": ["latest delta descending, None last", "first_seen_round", "first_seen_feature_index", "source_sha256"],
        "query_summary_required_for_actual_driver": True, "archive_lossless": True,
        "no_TEST_HOLDOUT_or_private_sample_IDs": True, "no_source_or_note_truncation": True,
        "remaining_overflow_action": "Stop entire stage before scientific model call; measurement mode reports without running L1"}
