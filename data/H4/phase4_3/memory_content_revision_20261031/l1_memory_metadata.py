"""Provisional L1 memory-content revision: metadata only, never feature text.

The deterministic format/retention policy is chosen before real measurement.
No dataset, model, generated feature, scorer or report is run on import/build.
The frozen original 6000-token counter is reused without modification.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

REVISION = Path(__file__).resolve().parent
PHASE = REVISION.parent
OLD_SRC = PHASE / "src"
if str(OLD_SRC) not in sys.path:
    sys.path.insert(0, str(OLD_SRC))
import l1_counter as counter
import l1_memory as original

VERSION = "H4-L1-memory-content-metadata-only-provisional-v1"
LIMIT = 6000
TOP_UNIQUE_FEATURES = 30
RECENT_ROUNDS = 3
NOTES_LIMIT = 300
FEATURE_FIELDS = {"sha256", "delta", "failed", "reasons"}
ROUND_FIELDS = {"round", "features", "queries", "stats", "note"}
QUERY_FIELDS = {"query", "requested", "matched", "revealed", "carry"}
STAT_FIELDS = {"capacity", "carry_in", "carry_out", "requested_count", "matched_count", "revealed_count"}
PHASES = {"inner_fit", "validation", "full_train"}
POLICY = (
    "Initially include all historical feature-instance metadata and ONLY the latest3 complete round records. A feature instance is "
    "identified by its original one-based round and zero-based feature index. Store only source SHA256, "
    "LOFO delta_without_minus_full, any-failure flag and aggregate failure reasons. Never put feature names, "
    "descriptions, code, code summaries, sample identities or held-out feedback in memory. Round records "
    "reference feature instances and contain ordered canonical queries, match/reveal/carry statistics and "
    "one complete note of at most300 Unicode characters. Count the entire canonical compact sorted JSON "
    "with the unchanged pinned6000-token counter. If over6000, retain the top30 UNIQUE SOURCE HASHES by "
    "their LATEST LOFO delta descending (None last; ties first_seen_round,first_seen_feature_index,sourceSHA256). "
    "Retain every historical feature-instance metadata item referencing those30 hashes, union every feature "
    "instance referenced by the latest3 complete round records. This is the original unique-hash ranking. "
    "Keep ONLY those latest3 round records after trimming; "
    "older notes/query records are not retained. Do not truncate notes, iterate trimming or change this "
    "format to fit a measured result. If this one prescribed retained view is still over6000, stop before "
    "a model request. The complete scientific source ledger remains separately preserved outside memory.")


class MemoryBudgetExceeded(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Required metadata-only memory uses {result['token_count']} tokens after the "
            f"single prescribed trim; unchanged limit is {LIMIT}; stop without further truncation")


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(key):
    if type(key) is not str or not re.fullmatch(r"(?:[1-9]|1[0-9]|20):(?:[0-9]|1[0-9])", key):
        raise ValueError("Feature identifier must be round:index with round1..20 and index0..19")
    return tuple(int(part) for part in key.split(":"))


def _int(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(field + " must be an integer >=" + str(minimum))
    return value


def _delta(value):
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError("LOFO delta must be finite or None")
    return None if value is None else float(value)


def _query_row(item):
    if not isinstance(item, dict) or set(item) != QUERY_FIELDS:
        raise ValueError("Unexpected metadata query fields")
    original._query(item["query"])
    for key in QUERY_FIELDS - {"query"}: _int(item[key], key)
    if item["requested"] != item["query"]["n"] or item["revealed"] > min(item["requested"], item["matched"]):
        raise ValueError("Query statistics disagree with the ordered canonical payload")


def validate_payload(payload, *, full_archive=False, private_ids=()):
    """Closed metadata grammar; source/name/description/rationale cannot survive."""
    if not isinstance(payload, dict) or set(payload) != {"features", "rounds"}:
        raise ValueError("Memory is exactly {features,rounds}; no source definitions or extra fields")
    features, rounds = payload["features"], payload["rounds"]
    if (not isinstance(features, dict) or not isinstance(rounds, list)
            or len(rounds) > (20 if full_archive else RECENT_ROUNDS)):
        raise ValueError("Invalid metadata feature table or round list")
    for identifier, value in features.items():
        _identity(identifier)
        if not isinstance(value, dict) or set(value) != FEATURE_FIELDS:
            raise ValueError("Historical feature permits only SHA256,delta,failed,reasons")
        if type(value["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"]):
            raise ValueError("Historical source SHA256 must be a complete lowercase64-hex hash")
        _delta(value["delta"])
        if type(value["failed"]) is not bool or not isinstance(value["reasons"], dict):
            raise ValueError("Failed must be a boolean and reasons a phase-to-counts mapping")
        failure_count = 0
        for phase, reasons in value["reasons"].items():
            if phase not in PHASES or not isinstance(reasons, dict):
                raise ValueError("Memory failure reasons are only TRAIN/validation aggregate categories")
            if any(type(k) is not str or not k for k in reasons):
                raise ValueError("Failure categories must be nonempty strings")
            for amount in reasons.values(): failure_count += _int(amount, "failure count")
        if value["failed"] != (failure_count > 0):
            raise ValueError("Failed flag differs from the aggregate failure reasons")
    previous = 0
    referenced = set()
    for record in rounds:
        if not isinstance(record, dict) or set(record) != ROUND_FIELDS:
            raise ValueError("Round permits only round,feature refs,queries,stats,note")
        rnd = _int(record["round"], "round", 1)
        if rnd > 20 or rnd <= previous or (full_archive and rnd != previous + 1):
            raise ValueError("Historical round ordering differs")
        previous = rnd
        refs = record["features"]
        if not isinstance(refs, list) or not 1 <= len(refs) <= 20 or len(set(refs)) != len(refs):
            raise ValueError("A detailed round must retain all1..20 original feature-instance references")
        for index, ref in enumerate(refs):
            if ref not in features or _identity(ref) != (rnd, index):
                raise ValueError("Round feature references differ from original round/index order")
            referenced.add(ref)
        if not isinstance(record["queries"], list) or len(record["queries"]) > 3:
            raise ValueError("At most3 ordered queries per detailed round")
        for item in record["queries"]: _query_row(item)
        stats = record["stats"]
        if not isinstance(stats, dict) or set(stats) != STAT_FIELDS:
            raise ValueError("Every round, including empty-query rounds, needs its complete aggregate ledger")
        for key in STAT_FIELDS: _int(stats[key], key)
        if (stats["capacity"] != 75 + stats["carry_in"] or stats["capacity"] > 1500
                or stats["requested_count"] != sum(q["requested"] for q in record["queries"])
                or stats["matched_count"] != sum(q["matched"] for q in record["queries"])
                or stats["revealed_count"] != sum(q["revealed"] for q in record["queries"])
                or stats["requested_count"] > stats["capacity"]
                or stats["carry_out"] != stats["capacity"] - stats["revealed_count"]):
            raise ValueError("Round match/reveal/carry statistics are inconsistent")
        if type(record["note"]) is not str or len(record["note"]) > NOTES_LIMIT:
            raise ValueError("A note must be <=300 Unicode characters, never truncated")
    if full_archive and referenced != set(features):
        raise ValueError("Complete metadata archive must account for every original feature instance")
    if not rounds and features:
        raise ValueError("Feature metadata without any completed round is invalid")
    original._no_private_ids(payload, private_ids)
    counter.canonical(payload)  # Reject NaN/non-JSON values without coercion.
    return True


def metadata_history(history, *, private_ids=()):
    """Read original scientific records transiently; return ONLY metadata.

    The full source/name/description are checked by the old format validator,
    used to compute identity, and are never retained in this returned archive.
    """
    raw = original.normalize_history(history, private_ids=private_ids)["rounds"]
    features, rounds = {}, []
    for record in raw:
        rnd = record["round_index"]
        if "query_summary" not in record:
            raise ValueError("Metadata memory requires even empty-query rounds' complete query summary")
        contributions = {x["feature_index"]: x["delta_without_minus_full"] for x in record["contributions"]}
        refs = []
        for index, feature in enumerate(record["features"]):
            identifier = str(rnd) + ":" + str(index)
            reasons = {}
            for failure in record["failures"]:
                if failure["feature_index"] != index: continue
                phase = reasons.setdefault(failure["phase"], {})
                for category, count in failure["reasons"].items(): phase[category] = phase.get(category, 0) + count
            features[identifier] = {"sha256": _sha(feature["source"]), "delta": _delta(contributions[index]),
                "failed": any(sum(values.values()) > 0 for values in reasons.values()), "reasons": reasons}
            refs.append(identifier)
        queries = [{"query": deepcopy(q["query"]), "requested": q["requested_count"], "matched": q["matched_count"],
            "revealed": q["revealed_count"], "carry": q["carry_after"]} for q in record["queries"]]
        rounds.append({"round": rnd, "features": refs, "queries": queries,
            "stats": deepcopy(record["query_summary"]), "note": record["notes"]})
    archive = {"features": features, "rounds": rounds}
    validate_payload(archive, full_archive=True, private_ids=private_ids)
    return archive


def component_breakdown(payload):
    """Exact fixed-order increments plus clearly non-additive standalone parts.

    Canonical JSON/BPE boundary merges can produce signed increments. The
    fixed-order increment sum, unlike standalone counts, equals the final count.
    """
    validate_payload(payload)
    stages = []
    snapshots = [
        ("empty_JSON_container", {"features": {}, "rounds": []}),
        ("feature_metadata", {"features": deepcopy(payload["features"]), "rounds": []}),
        ("rounds_and_feature_references", {"features": deepcopy(payload["features"]), "rounds": [
            {"round": r["round"], "features": deepcopy(r["features"])} for r in payload["rounds"]]}),
        ("ordered_queries_and_aggregate_stats", {"features": deepcopy(payload["features"]), "rounds": [
            {k: deepcopy(v) for k, v in r.items() if k != "note"} for r in payload["rounds"]]}),
        ("complete_notes", deepcopy(payload))]
    last_tokens = last_bytes = 0
    for name, value in snapshots:
        measured = counter.measure(value)
        stages.append({"component": name, "token_increment": measured["token_count"] - last_tokens,
            "UTF8_byte_increment": measured["UTF8_bytes"] - last_bytes,
            "cumulative_token_count": measured["token_count"], "cumulative_UTF8_bytes": measured["UTF8_bytes"]})
        last_tokens, last_bytes = measured["token_count"], measured["UTF8_bytes"]
    standalone = {
        "feature_identifiers_and_source_SHA256": {key: {"sha256": value["sha256"]} for key, value in payload["features"].items()},
        "feature_LOFO_deltas": {key: {"delta": value["delta"]} for key, value in payload["features"].items()},
        "feature_failure_flags_and_reasons": {key: {"failed": value["failed"], "reasons": value["reasons"]} for key, value in payload["features"].items()},
        "round_identifiers_and_feature_references": [{"round": r["round"], "features": r["features"]} for r in payload["rounds"]],
        "ordered_queries_and_aggregate_stats": [{"round": r["round"], "queries": r["queries"], "stats": r["stats"]} for r in payload["rounds"]],
        "complete_notes": [{"round": r["round"], "note": r["note"]} for r in payload["rounds"]]}
    alone = {name: {k: v for k, v in counter.measure(value).items() if k != "text"} for name, value in standalone.items()}
    return {"method": "Fixed predeclared assembly order, exact signed token increments; standalone sections are non-additive",
        "ordered_increments": stages, "standalone_nonadditive_components": alone,
        "sum_token_increments": sum(s["token_increment"] for s in stages),
        "sum_UTF8_byte_increments": sum(s["UTF8_byte_increment"] for s in stages),
        "final_token_count": last_tokens, "final_UTF8_bytes": last_bytes}


def build_memory(history, *, measurement=False, private_ids=()):
    """Pure build/measure; accept raw scientific history OR a metadata-only archive."""
    if isinstance(history, list):
        archive = metadata_history(history, private_ids=private_ids)
    elif isinstance(history, dict):
        archive = deepcopy(history)
        validate_payload(archive, full_archive=True, private_ids=private_ids)
    else:
        raise ValueError("History must be original round records or the complete metadata-only archive")
    payload = {"features": deepcopy(archive["features"]), "rounds": deepcopy(archive["rounds"][-RECENT_ROUNDS:])}
    full = counter.measure(payload)
    trimmed = full["token_count"] > LIMIT
    first_seen, latest_delta = {}, {}
    for identifier in sorted(archive["features"], key=_identity):
        value = archive["features"][identifier]
        source_sha = value["sha256"]
        first_seen.setdefault(source_sha, (*_identity(identifier), source_sha))
        latest_delta[source_sha] = value["delta"]
    ranking = sorted(first_seen, key=lambda source_sha: (
        latest_delta[source_sha] is None,
        -latest_delta[source_sha] if latest_delta[source_sha] is not None else 0,
        *first_seen[source_sha]))
    if trimmed:
        recent = deepcopy(archive["rounds"][-RECENT_ROUNDS:])
        top_hashes = set(ranking[:TOP_UNIQUE_FEATURES])
        keep = {identifier for identifier, value in archive["features"].items() if value["sha256"] in top_hashes}
        keep |= {ref for record in recent for ref in record["features"]}
        payload = {"features": {key: deepcopy(archive["features"][key]) for key in sorted(keep, key=_identity)}, "rounds": recent}
    validate_payload(payload, private_ids=private_ids)
    measured = counter.measure(payload)
    breakdown = component_breakdown(payload)
    if breakdown["final_token_count"] != measured["token_count"] or breakdown["final_UTF8_bytes"] != measured["UTF8_bytes"]:
        raise AssertionError("Component accounting differs from the exact serialized memory")
    result = {"payload": payload, **measured, "trimmed": trimmed,
        "overflow": measured["token_count"] > LIMIT, "limit_tokens": LIMIT,
        "payload_sha256": _sha(measured["text"]), "archive": archive,
        "archive_sha256": _sha(counter.canonical(archive)), "component_breakdown": breakdown,
        "audit": {"version": VERSION, "history_rounds": len(archive["rounds"]),
            "feature_instances_full": len(archive["features"]), "feature_instances_retained": len(payload["features"]),
            "round_records_full": len(archive["rounds"]), "round_records_retained": len(payload["rounds"]),
            "untrimmed_token_count": full["token_count"], "untrimmed_UTF8_bytes": full["UTF8_bytes"],
            "unique_source_hashes_full": len(ranking),
            "unique_source_hashes_retained": len({value["sha256"] for value in payload["features"].values()}),
            "top_source_hashes": ranking[:TOP_UNIQUE_FEATURES] if trimmed else [],
            "recent_rounds_preserved": [r["round"] for r in payload["rounds"][-RECENT_ROUNDS:]],
            "older_round_notes_queries_retained_after_trim": False,
            "older_round_notes_queries_retained_in_any_counted_view": False,
            "name_description_source_or_source_summary_retained": False,
            "source_or_note_truncation": False, "measurement_only": bool(measurement)}}
    if result["overflow"] and not measurement: raise MemoryBudgetExceeded(result)
    return result


def persist_archive(path, history, *, private_ids=()):
    """Explicit exclusive write of the metadata-only archive, never scientific source."""
    destination = Path(path).resolve()
    if "onedrive" in str(destination).casefold() or not destination.is_relative_to(REVISION.resolve()):
        raise ValueError("Revised memory archive must stay under its non-OneDrive revision directory")
    if isinstance(history, list): archive = metadata_history(history, private_ids=private_ids)
    else:
        archive = deepcopy(history)
        validate_payload(archive, full_archive=True, private_ids=private_ids)
    encoded = (counter.canonical(archive) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination or "onedrive" in str(destination.resolve()).casefold():
        raise ValueError("Memory archive path changed")
    with destination.open("xb") as stream: stream.write(encoded)
    return {"path": str(destination), "sha256": hashlib.sha256(encoded).hexdigest(), "rounds": len(archive["rounds"])}


def configuration():
    return {"version": VERSION, "status": "provisional_not_frozen", "policy": POLICY,
        "payload_keys": ["features", "rounds"], "feature_identifier": "one-based-round:zero-based-index",
        "feature_fields": sorted(FEATURE_FIELDS), "round_fields": sorted(ROUND_FIELDS),
        "query_record_fields": sorted(QUERY_FIELDS), "round_stat_fields": sorted(STAT_FIELDS),
        "failure_flag_definition": "True iff any observed TRAIN/validation failure category count is positive",
        "failure_reasons": "phase -> category -> count; no row IDs, tracebacks or source summaries added",
        "top_unique_source_hashes": TOP_UNIQUE_FEATURES, "recent_complete_rounds": RECENT_ROUNDS,
        "tie_rule": "LATEST LOFO delta for each unique source hash descending; None last; first_seen_round,first_seen_feature_index,sourceSHA256",
        "trim_action": "Original unique-hash ranking; one pass all historical instances of top30hashes union latest3 feature refs; onlylatest3 detailed records/notes/queries remain",
        "initial_view_round_records": "Only latest3 complete query/note/feature-reference records from the start; all historical feature metadata until6000 trim",
        "counter": "Unchanged l1_counter.measure (pinned tiktoken0.12.0/o200k_base.encode_ordinary)",
        "limit_tokens": LIMIT, "notes_limit_unicode_characters": NOTES_LIMIT,
        "scientific_sources_stored_inside_memory": False,
        "metadata_archive_only": True, "remaining_overflow_action": "Stop before any model call, no iterative truncation",
        "component_breakdown": "Exact fixed-order signed increments plus explicitly non-additive standalone component counts"}
