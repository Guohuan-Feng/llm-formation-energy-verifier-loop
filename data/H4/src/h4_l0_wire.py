"""Private H4A provider-schema compatibility; canonical science stays frozen.

No call or write occurs on import. Optional canonical object fields travel as
required nullable wire fields. Null is normalized to omission before the frozen
transport's canonical validation; uniqueness remains canonically enforced.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h4_transport as base

VERSION = "H4-L0-strict-wire-optional-null-1.0"
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


class StrictSchemaTransport(base.FiniteTransport):
    """Same constructor/cache identity as FiniteTransport, strict provider wire.

    Production:
      StrictSchemaTransport(runtime, DATA/'transport', OUT, log=log, ...)
    Existing first-valid responses and original request fingerprints are reused.
    For a corrected failed query the coordinator supplies one new request_id.
    ``call_once_fn`` is inherited solely as the CPU mock seam.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canonical_owned_call_once = self._call_once
        self._call_once = self._strict_owned_call_once

    def _strict_owned_call_once(self, prompt, *, schema_path, instructions_path, catalog_path,
                                stop_event, metadata=None, deadline_seconds=None):
        canonical_path = base.safe(schema_path, must_exist=True)
        original_bytes = canonical_path.read_bytes()
        schema = json.loads(original_bytes.decode("utf-8-sig"))
        wire = strict_schema(schema)
        changed = base.canonical(wire) != base.canonical(schema)
        # A fully-required evaluation schema has exactly the old bytes/prompt.
        if changed:
            wire_bytes = (base.canonical(wire) + "\n").encode("utf-8")
            with self.lock:
                wire_path = base._immutable(self.data_dir / "wire_schemas" / (base.digest(wire_bytes) + ".json"), wire_bytes)
            wire_prompt = prompt + WIRE_NOTE
        else:
            wire_path = canonical_path
            wire_prompt = prompt
        record = self._canonical_owned_call_once(wire_prompt, schema_path=wire_path,
            instructions_path=instructions_path, catalog_path=catalog_path, stop_event=stop_event,
            metadata=metadata, deadline_seconds=deadline_seconds)
        if not isinstance(record, dict):
            return record
        record.update(wire_transport_version=VERSION, wire_schema_changed=changed,
            canonical_schema_path=str(canonical_path), canonical_schema_sha256=base.digest(original_bytes),
            wire_schema_path=str(wire_path), wire_schema_sha256=base.file_sha(wire_path),
            canonical_prompt_sha256=base.digest(prompt.encode("utf-8")),
            wire_prompt_sha256=base.digest(wire_prompt.encode("utf-8")))
        if changed:
            record["provider_response_text"] = record.get("response_text", "")
            record["provider_event_log_path"] = record.get("event_log_path")
            if record.get("status") == "ok":
                try:
                    raw = json.loads(record["provider_response_text"], parse_constant=base._reject_constant)
                    normalized = normalize_optional_nulls(raw, schema)
                    record["response_text"] = base.canonical(normalized)
                    record["optional_nulls_normalized_to_omission"] = True
                except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
                    # Keep an invalid response invalid. Frozen canonical validation
                    # supplies its usual format error and exactly-once policy.
                    record["wire_normalization_error"] = f"{type(exc).__name__}: {exc}"
        return record
