"""Finite H4 local CLI transport. Importing this module never makes a call.

Scientific orchestration supplies prompts and semantic validators. This adapter
owns only immutable request identity, transport retries, first-valid caching,
durable accounting, and the frozen H3 owned-process CLI implementation. H3
source/configuration/cache and the H3 sandbox are never modified or executed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[4]
VERSION = "H4-finite-owned-CLI-1.0"
MODEL = "gpt-6.1-sol"
PINNED_SHA256 = "5c1b55966bf9f757dcbf00065c9a6cb862ead55d34b015f1be1939ff18e43943"
PINNED_CLI_VERSION = "codex-cli 0.165.0-alpha.1.2"
PINNED_EXE = ROOT / "work/h-data/H3/runtime_pinned/codex-0.165.0-alpha.1.2.exe"
INPUT_LIMIT = 20_000
DELAYS = (60, 120, 240)
INTEGRITY = {"tool_activity_rejected", "model_mismatch_rejected", "backend_protocol_rejected",
             "preflight_rejected", "actual_input_budget_violation"}
FORMAT_REPAIR = "\nFORMAT REPAIR ONLY: Return exactly the requested JSON schema with all required fields. No tools, files, examples, or additional information are available."
DEFAULT_INSTRUCTIONS = "Return only final JSON matching the response schema. Tools are forbidden. Do not use files, shell, network, or external services. Do not include chain-of-thought or reasoning steps.\n"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe(path, *, within=None, must_exist=False) -> Path:
    """Resolve symlinks/junctions before every task-controlled file operation."""
    value = Path(path).resolve(strict=must_exist)
    if any("onedrive" in part.casefold() for part in value.parts):
        raise ValueError("OneDrive path is forbidden")
    if within is not None and not value.is_relative_to(Path(within).resolve(strict=True)):
        raise ValueError("Path escapes the H4 destination")
    return value


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path) -> str:
    hasher = hashlib.sha256()
    with safe(path, must_exist=True).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON value")
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)


def _reject_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def _schema_validate(value, schema, *, root=None, location="$", depth=0):
    """Dependency-free validator for the JSON Schema vocabulary used in H4.

    Unsupported validation keywords fail closed rather than silently accepting
    a schema. Local $ref, combinations, object/array bounds, scalar ranges and
    enums are supported; no remote reference or executable validator is loaded.
    """
    if depth > 100:
        raise ValueError("JSON Schema nesting exceeds 100")
    if isinstance(schema, bool):
        if not schema:
            raise ValueError(location + ": forbidden by schema")
        return
    if not isinstance(schema, dict):
        raise ValueError("Schema must be an object or boolean")
    root = schema if root is None else root
    supported = {"$schema", "$id", "$defs", "definitions", "$ref", "title", "description", "default", "examples",
                 "type", "enum", "const", "properties", "required", "additionalProperties", "items", "prefixItems",
                 "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties", "minimum", "maximum",
                 "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern",
                 "anyOf", "oneOf", "allOf", "not", "format"}
    unknown = set(schema) - supported
    if unknown:
        raise ValueError("Unsupported schema keyword: " + ", ".join(sorted(unknown)))
    if "$ref" in schema:
        ref = schema["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            raise ValueError("Only local JSON schema references are allowed")
        target = root
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        _schema_validate(value, target, root=root, location=location, depth=depth + 1)
    if "format" in schema:
        raise ValueError("Schema format validators are not enabled")
    for key in ("allOf", "anyOf", "oneOf"):
        if key in schema:
            matches = 0
            for child in schema[key]:
                try:
                    _schema_validate(value, child, root=root, location=location, depth=depth + 1)
                    matches += 1
                except (ValueError, TypeError, KeyError):
                    pass
            if ((key == "allOf" and matches != len(schema[key])) or
                    (key == "anyOf" and not matches) or (key == "oneOf" and matches != 1)):
                raise ValueError(location + ": fails " + key)
    if "not" in schema:
        try:
            _schema_validate(value, schema["not"], root=root, location=location, depth=depth + 1)
        except (ValueError, TypeError, KeyError):
            pass
        else:
            raise ValueError(location + ": fails not")
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "boolean": lambda x: isinstance(x, bool),
              "null": lambda x: x is None,
              "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
              "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)}
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if any(kind not in checks for kind in allowed) or not any(checks[kind](value) for kind in allowed):
            raise ValueError(location + ": type mismatch")
    if "enum" in schema and not any(canonical(value) == canonical(item) for item in schema["enum"]):
        raise ValueError(location + ": invalid enum value")
    if "const" in schema and canonical(value) != canonical(schema["const"]):
        raise ValueError(location + ": invalid const value")
    if isinstance(value, dict):
        missing = set(schema.get("required", [])) - set(value)
        if missing:
            raise ValueError(location + ": missing required fields " + ", ".join(sorted(missing)))
        props = schema.get("properties", {})
        for key, child in value.items():
            rule = props.get(key, schema.get("additionalProperties", True))
            _schema_validate(child, rule, root=root, location=location + "." + key, depth=depth + 1)
        for bound, op in (("minProperties", lambda n: len(value) < n), ("maxProperties", lambda n: len(value) > n)):
            if bound in schema and op(schema[bound]):
                raise ValueError(location + ": " + bound)
    elif isinstance(value, list):
        for bound, op in (("minItems", lambda n: len(value) < n), ("maxItems", lambda n: len(value) > n)):
            if bound in schema and op(schema[bound]):
                raise ValueError(location + ": " + bound)
        if schema.get("uniqueItems") and len({canonical(x) for x in value}) != len(value):
            raise ValueError(location + ": duplicate items")
        prefix = schema.get("prefixItems", [])
        for i, child in enumerate(value):
            rule = prefix[i] if i < len(prefix) else schema.get("items", True)
            _schema_validate(child, rule, root=root, location=f"{location}[{i}]", depth=depth + 1)
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ValueError(location + ": string length")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(location + ": pattern mismatch")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        for bound, op in (("minimum", lambda n: value < n), ("maximum", lambda n: value > n),
                          ("exclusiveMinimum", lambda n: value <= n), ("exclusiveMaximum", lambda n: value >= n)):
            if bound in schema and op(schema[bound]):
                raise ValueError(location + ": " + bound)
        if "multipleOf" in schema and not math.isclose(value / schema["multipleOf"], round(value / schema["multipleOf"]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(location + ": multipleOf")


def validate_response(text, schema, validator=None):
    value = json.loads(text, parse_constant=_reject_constant)
    _finite(value)
    _schema_validate(value, schema)
    if validator is not None:
        checked = validator(value)
        if checked is False:
            raise ValueError("Semantic validator rejected response")
        if checked is not None and checked is not True:
            value = checked
        _finite(value)
        _schema_validate(value, schema)
    return value


def _read_lines(path):
    path = safe(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank or incomplete durable ledger line {index}")
            records.append(json.loads(line, parse_constant=_reject_constant))
    return records


def _append(path, record):
    path = safe(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _immutable(path, content: bytes):
    path = safe(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("Immutable protocol file changed: " + path.name)
    return path


def _load_fresh(path, name):
    definition = importlib.util.spec_from_file_location(name, safe(path, must_exist=True))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def _production_call_once(runtime, data_dir):
    """Reuse frozen owned-Job code with wholly separate H4 module globals."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    e2 = _load_fresh(ROOT / "e2_transport.py", "_h4_e2_private")
    base = _load_fresh(ROOT / "f2_transport_v2.py", "_h4_f2_private")
    original = _load_fresh(ROOT / "h3_resume_transport.py", "_h4_owned_source")
    e2.CODEX_EXE = Path(runtime["executable"])
    e2.CLI_VERSION = runtime["cli_version"]
    e2.MODEL = MODEL
    e2.REASONING_EFFORT = "high"
    e2.INPUT_LIMIT = INPUT_LIMIT
    e2.ROOT = safe(ROOT, must_exist=True)
    base._e2 = e2
    base.resolve_runtime = lambda: dict(runtime)
    original_environment = base._environment
    private_temp = safe(data_dir / "cli_tmp")
    private_temp.mkdir(parents=True, exist_ok=True)

    def environment():
        result = original_environment()
        result["TEMP"] = result["TMP"] = str(safe(private_temp, must_exist=True))
        safe(Path(result.get("CODEX_HOME", Path.home() / ".codex")), must_exist=True)
        return result

    base._environment = environment
    globals_copy = dict(original.call_once.__globals__)
    globals_copy.update(_e2=e2, _base=base,
                        c=types.SimpleNamespace(DATA=data_dir, safe=safe, now=now), TRANSPORT_VERSION=VERSION)
    call = types.FunctionType(original.call_once.__code__, globals_copy, "h4_owned_call_once",
                              original.call_once.__defaults__, original.call_once.__closure__)
    call.__kwdefaults__ = dict(original.call_once.__kwdefaults__ or {})
    return call, e2


def _verify_runtime(runtime):
    expected = {"executable": str(safe(PINNED_EXE, must_exist=True)), "sha256": PINNED_SHA256,
                "cli_version": PINNED_CLI_VERSION, "model": MODEL, "reasoning_effort": "high"}
    provided = dict(runtime)
    provided["executable"] = str(safe(provided["executable"], must_exist=True))
    if provided != expected or file_sha(expected["executable"]) != PINNED_SHA256:
        raise ValueError("Frozen H4 CLI runtime identity changed")
    version = subprocess.run([expected["executable"], "--version"], cwd=safe(ROOT, must_exist=True),
                             capture_output=True, text=True, encoding="utf-8", timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if version.returncode or version.stdout.strip() != PINNED_CLI_VERSION:
        raise ValueError("Frozen H4 CLI runtime version changed")
    return expected


class FiniteTransport:
    """One process, up to four owned CLI trees, append-only durable accounting.

    ``call_once_fn`` is an explicit CPU-test seam: when supplied no CLI is loaded
    or launched. Production always checks the exact pinned binary SHA/version.
    Different request IDs remain independent even when their prompts coincide.
    Each ID is permanently bound to its complete request fingerprint.
    """
    def __init__(self, runtime, data_dir, out_dir, protocol_paths=None, log=None, *,
                 max_calls=300, max_logical_requests=180, max_workers=4,
                 call_once_fn=None, wait_fn=None):
        if not 1 <= int(max_workers) <= 4:
            raise ValueError("H4 allows one to four CLI workers")
        if int(max_calls) <= 0 or int(max_logical_requests) <= 0:
            raise ValueError("Finite budgets must be positive")
        h4_root = safe(ROOT / "work/h-data/H4", must_exist=True)
        self.data_dir = safe(data_dir, within=h4_root)
        self.out_dir = safe(out_dir)
        # Production has exactly the designated experiment output; test mocks
        # may use their isolated software_* directories within H4.
        if call_once_fn is None and self.out_dir != safe(ROOT / "H4_feature_loop"):
            if not self.out_dir.is_relative_to(h4_root):
                raise ValueError("Unexpected H4 output destination")
        elif call_once_fn is not None and not self.out_dir.is_relative_to(h4_root):
            raise ValueError("Mock outputs must stay inside H4")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = deepcopy(runtime) if call_once_fn else _verify_runtime(runtime)
        self.max_calls = int(max_calls)
        self.max_logical_requests = int(max_logical_requests)
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self._semaphore = threading.BoundedSemaphore(int(max_workers))
        self._flight = {}
        self._identity = {}
        self._cache = {}
        self._history = []
        self._results = []
        self.n_calls = 0
        self.failed_streak = 0
        self.stop_reason = None
        self.started_utc = now()
        self.started_monotonic = time.monotonic()
        self._log_callback = log
        self._wait_fn = wait_fn
        self.ledger = safe(self.out_dir / "llm_calls.jsonl")
        self.results_path = safe(self.data_dir / "request_results.jsonl")
        self.events_path = safe(self.data_dir / "transport_events.jsonl")
        self._owner_handle = None
        self._claim_owner()
        try:
            self._restore()
            self.protocol_paths = dict(protocol_paths or {})
            catalog_source = self.protocol_paths.get("catalog", self.protocol_paths.get("catalog_path", ROOT / "work/h-data/H3/model_catalog.json"))
            catalog_bytes = safe(catalog_source, must_exist=True).read_bytes()
            self.catalog_path = _immutable(self.data_dir / "protocol" / (digest(catalog_bytes) + ".catalog.json"), catalog_bytes)
            instructions_source = self.protocol_paths.get("instructions", self.protocol_paths.get("instructions_path"))
            self.default_instructions = safe(instructions_source, must_exist=True).read_text(encoding="utf-8-sig") if instructions_source else DEFAULT_INSTRUCTIONS
            self._call_once, self._e2 = ((call_once_fn, None) if call_once_fn else _production_call_once(self.runtime, self.data_dir))
            self._event("initialized", cli_attempts=self.n_calls, first_valid_cached=len(self._cache), test_mode=call_once_fn is not None)
        except BaseException:
            self.close()
            raise

    def _claim_owner(self):
        """Windows byte lock prevents two transport owners using one ledger."""
        path = safe(self.data_dir / "transport_owner.lock")
        self._owner_handle = path.open("a+b")
        self._owner_handle.seek(0, 2)
        if self._owner_handle.tell() == 0:
            self._owner_handle.write(b"0")
            self._owner_handle.flush()
            os.fsync(self._owner_handle.fileno())
        self._owner_handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._owner_handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._owner_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner_handle.close()
            self._owner_handle = None
            raise RuntimeError("Another H4 transport owns this private ledger")

    def close(self):
        with self.lock:
            if self._flight:
                raise RuntimeError("Cannot close transport while requests are active")
            if self._owner_handle is not None:
                if os.name == "nt":
                    import msvcrt
                    self._owner_handle.seek(0)
                    msvcrt.locking(self._owner_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._owner_handle.fileno(), fcntl.LOCK_UN)
                self._owner_handle.close()
                self._owner_handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _event(self, event, **values):
        with self.lock:
            _append(self.events_path, {"utc": now(), "event": event, **values})
        if self._log_callback is not None:
            self._log_callback("H4 transport " + event + (" " + str(values["request_id"]) if "request_id" in values else ""))

    def _terminal(self, reason):
        with self.lock:
            self.stop_reason = self.stop_reason or reason
            self.stop.set()

    def _bind(self, request_id, request_hash):
        if request_id in self._identity and self._identity[request_id] != request_hash:
            raise ValueError("A request_id was reused with a different frozen payload")
        self._identity[request_id] = request_hash

    def _restore(self):
        reservations = {}
        for record in _read_lines(self.ledger):
            if record.get("record_type") not in {"reservation", "result"}:
                raise ValueError("Unknown durable CLI ledger record")
            index = record.get("call_index")
            if not isinstance(index, int) or index < 1:
                raise ValueError("Invalid durable call index")
            self.n_calls = max(self.n_calls, index)
            self._bind(record["request_id"], record["request_fingerprint"])
            if record["record_type"] == "reservation":
                if index in reservations:
                    raise ValueError("Duplicate CLI reservation")
                reservations[index] = record
            else:
                if index not in reservations or reservations[index].get("completed"):
                    raise ValueError("CLI result has no unique reservation")
                reservation = reservations[index]
                if any(record.get(key) != reservation.get(key) for key in ("request_id", "request_fingerprint", "kind")):
                    raise ValueError("CLI reservation/result identity mismatch")
                reservation["completed"] = True
                self._history.append(record)
                if record.get("status") == "ok":
                    if "validated_response" not in record or digest(canonical(record["validated_response"]).encode("utf-8")) != record.get("validated_response_sha256"):
                        raise ValueError("First-valid response hash mismatch")
                    rid = record["request_id"]
                    if rid in self._cache:
                        raise ValueError("Duplicate successful request in immutable ledger")
                    self._cache[rid] = record
        if self.n_calls and set(reservations) != set(range(1, self.n_calls + 1)):
            raise ValueError("CLI ledger reservation sequence has a gap")
        for record in _read_lines(self.results_path):
            self._bind(record["request_id"], record["request_fingerprint"])
            if record.get("status") == "ok":
                cached = self._cache.get(record["request_id"])
                if cached is None or record.get("response_sha256") != cached.get("validated_response_sha256"):
                    raise ValueError("Private success lacks identical public first-valid result")
            self._results.append(record)
        # A successful public result is sufficient to recover after a crash
        # before its private logical-result append. No second model call occurs.
        for record in self._results:
            if record.get("status") == "ok":
                self.failed_streak = 0
            elif not record.get("admissible_format_failure"):
                self.failed_streak += 1
        terminal = next((r for r in reversed(self._history) if r.get("status") in INTEGRITY | {"terminal_usage_limit"}), None)
        if terminal is not None:
            self._terminal({"status": terminal["status"], "request_id": terminal["request_id"]})
        elif self.failed_streak >= 3:
            self._terminal("three_consecutive_final_failed_requests")

    def _prepare(self, spec):
        if not isinstance(spec, dict) or not all(isinstance(spec.get(k), str) and spec[k] for k in ("request_id", "kind", "prompt")):
            raise ValueError("Spec requires non-empty request_id, kind and prompt")
        schema = spec.get("schema")
        if isinstance(schema, (str, Path)):
            schema = json.loads(safe(schema, must_exist=True).read_text(encoding="utf-8-sig"))
        if not isinstance(schema, (dict, bool)):
            raise ValueError("Spec requires a JSON schema")
        schema_bytes = (canonical(schema) + "\n").encode("utf-8")
        instructions = spec.get("instructions", self.default_instructions)
        if not isinstance(instructions, str):
            raise ValueError("Spec instructions must be text")
        instruction_bytes = instructions.encode("utf-8")
        max_repairs = spec.get("max_format_repairs", 1)
        if not isinstance(max_repairs, int) or isinstance(max_repairs, bool) or not 0 <= max_repairs <= 1:
            raise ValueError("At most one format-only repair is allowed")
        identity = {"kind": spec["kind"], "prompt": spec["prompt"], "schema": schema,
                    "instructions": instructions, "runtime": self.runtime,
                    "catalog_sha256": file_sha(self.catalog_path), "transport_version": VERSION,
                    "max_format_repairs": max_repairs, "admissible_format_failure": bool(spec.get("admissible_format_failure", False))}
        request_hash = digest(canonical(identity).encode("utf-8"))
        with self.lock:
            schema_path = _immutable(self.data_dir / "schemas" / (digest(schema_bytes) + ".json"), schema_bytes)
            instructions_path = _immutable(self.data_dir / "instructions" / (digest(instruction_bytes) + ".md"), instruction_bytes)
        return schema, request_hash, schema_path, instructions_path, max_repairs

    def _cached_result(self, spec, request_hash, schema, validator):
        cached = self._cache[spec["request_id"]]
        value = validate_response(cached["response_text"], schema, validator)
        if canonical(value) != canonical(cached["validated_response"]):
            raise ValueError("Cached semantic validator changed the first-valid response")
        return {"request_id": spec["request_id"], "kind": spec["kind"], "id": spec.get("id"), "seed": spec.get("seed"),
                "status": "ok", "response": deepcopy(value), "last_status": "ok", "last_call_index": cached["call_index"],
                "request_fingerprint": request_hash, "response_sha256": cached["validated_response_sha256"],
                "cache_hit": True, "logical_elapsed_seconds": 0.0, "cli_attempt_count": 0}

    def cached(self, spec, validator=None):
        """Read a validated first-valid cache entry without dispatching a call."""
        schema, request_hash, _, _, _ = self._prepare(spec)
        with self.lock:
            self._bind(spec["request_id"], request_hash)
            if spec["request_id"] not in self._cache:
                return None
            return self._cached_result(spec, request_hash, schema, validator)

    @property
    def responses(self):
        with self.lock:
            return {rid: deepcopy(record["validated_response"]) for rid, record in self._cache.items()}

    def request(self, spec, validator=None):
        """Execute or reuse one immutable logical request; never print its data."""
        if self._owner_handle is None:
            raise RuntimeError("Transport is closed")
        schema, request_hash, schema_path, instructions_path, max_repairs = self._prepare(spec)
        request_id = spec["request_id"]
        while True:
            with self.lock:
                try:
                    self._bind(request_id, request_hash)
                    if request_id in self._cache:
                        return self._cached_result(spec, request_hash, schema, validator)
                except BaseException as exc:
                    self._terminal("request_integrity_failure: " + str(exc))
                    raise
                if request_id not in self._flight:
                    owned_flight = {"event": threading.Event(), "result": None, "error": None}
                    self._flight[request_id] = owned_flight
                    break
                pending = self._flight[request_id]
            pending["event"].wait()
            if pending["error"] is not None:
                raise pending["error"]
            joined = deepcopy(pending["result"])
            joined["coalesced_inflight"] = True
            joined["cli_attempt_count"] = 0
            return joined
        try:
            result = self._run(spec, schema, request_hash, schema_path, instructions_path, max_repairs, validator)
            owned_flight["result"] = result
            return result
        except BaseException as exc:
            owned_flight["error"] = exc
            raise
        finally:
            with self.lock:
                self._flight.pop(request_id)["event"].set()

    def _reserve(self, spec, request_hash, format_round, rate_retry, prompt):
        with self.lock:
            if self.stop.is_set():
                return None
            if self.n_calls >= self.max_calls:
                self._terminal("finite_global_cli_call_cap")
                return None
            if len(self._identity) > self.max_logical_requests:
                self._terminal("finite_global_logical_request_cap")
                return None
            self.n_calls += 1
            reservation = {"record_type": "reservation", "call_index": self.n_calls, "request_id": spec["request_id"],
                           "kind": spec["kind"], "request_fingerprint": request_hash, "format_round": format_round,
                           "rate_limit_transport_retry": rate_retry, "reserved_utc": now(),
                           "prompt_sha256": digest(prompt.encode("utf-8")), "transport_version": VERSION}
            _append(self.ledger, reservation)
            return reservation

    def _run(self, spec, schema, request_hash, schema_path, instructions_path, max_repairs, validator):
        started = time.monotonic()
        final = None
        parsed = None
        rate_retries = 0
        attempts = 0
        for format_round in range(max_repairs + 1):
            prompt = spec["prompt"] + (FORMAT_REPAIR if format_round else "")
            while not self.stop.is_set():
                with self._semaphore:
                    reservation = self._reserve(spec, request_hash, format_round, rate_retries, prompt)
                    if reservation is None:
                        break
                    metadata = {**reservation, "record_type": "result", "started_utc": now(),
                                **{k: spec[k] for k in ("id", "seed", "arm", "round", "stage") if k in spec}}
                    try:
                        record = self._call_once(prompt, schema_path=schema_path, instructions_path=instructions_path,
                                                 catalog_path=self.catalog_path, stop_event=self.stop, metadata=metadata,
                                                 deadline_seconds=None)
                        if not isinstance(record, dict):
                            raise TypeError("call_once did not return a record")
                        record = {**metadata, **record}
                        # Caller/transport may never mutate reserved identity.
                        if any(record.get(k) != metadata[k] for k in ("call_index", "request_id", "kind", "request_fingerprint")):
                            raise ValueError("call_once changed reserved request identity")
                    except Exception as exc:
                        record = {**metadata, "status": "backend_protocol_rejected", "error": f"{type(exc).__name__}: {exc}",
                                  "response_text": "", "elapsed_seconds": 0.0}
                    attempts += 1
                    record["completed_utc"] = now()
                    if record.get("status") == "ok":
                        try:
                            parsed = validate_response(record.get("response_text", ""), schema, validator)
                            record["validated_response"] = parsed
                            record["validated_response_sha256"] = digest(canonical(parsed).encode("utf-8"))
                        except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
                            record["status"] = "format_invalid"
                            record["format_error"] = f"{type(exc).__name__}: {exc}"
                            parsed = None
                    with self.lock:
                        _append(self.ledger, record)
                        self._history.append(record)
                        if parsed is not None:
                            if spec["request_id"] in self._cache:
                                self._terminal("duplicate_first_valid_response")
                                raise ValueError("Duplicate first-valid response")
                            self._cache[spec["request_id"]] = record
                    final = record
                # Frozen call_once performs process.wait, joins and Job close
                # before the semaphore is released above.
                if record.get("status") in INTEGRITY | {"terminal_usage_limit"}:
                    self._terminal({k: record.get(k) for k in ("status", "request_id", "error", "recovery_time_message", "recovery_reset_fields")})
                    self._event("provider_or_integrity_stop", request_id=spec["request_id"], status=record["status"])
                    break
                if record.get("status") == "rate_limited_final" and rate_retries < 3:
                    delay = DELAYS[rate_retries]
                    rate_retries += 1
                    self._event("same_request_final429_backoff", request_id=spec["request_id"], retry=rate_retries, seconds=delay)
                    interrupted = self._wait_fn(delay) if self._wait_fn else self.stop.wait(delay)
                    if interrupted or self.stop.is_set():
                        break
                    continue
                break
            if parsed is not None or self.stop.is_set() or final is None or final.get("status") != "format_invalid":
                break
        success = parsed is not None
        admissible = bool(spec.get("admissible_format_failure")) and final is not None and final.get("status") == "format_invalid"
        result = {"request_id": spec["request_id"], "kind": spec["kind"], "id": spec.get("id"), "seed": spec.get("seed"),
                  "status": "ok" if success else "failed", "response": deepcopy(parsed) if success else None,
                  "last_call_index": final.get("call_index") if final else None, "last_status": final.get("status") if final else "undispatched",
                  "validation_error": final.get("format_error") if final else None,
                  "raw_response_text": final.get("response_text") if final and not success else None,
                  "request_fingerprint": request_hash, "response_sha256": final.get("validated_response_sha256") if success else None,
                  "cache_hit": False, "logical_elapsed_seconds": time.monotonic() - started,
                  "cli_attempt_count": attempts, "rate_limit_transport_retries": rate_retries,
                  "admissible_format_failure": admissible, "completed_utc": now()}
        with self.lock:
            if success:
                self.failed_streak = 0
            elif not admissible:
                self.failed_streak += 1
            if self.failed_streak >= 3:
                self._terminal("three_consecutive_final_failed_requests")
            _append(self.results_path, result)
            self._results.append(result)
        self._event("logical_complete", request_id=spec["request_id"], status=result["status"],
                    last_status=result["last_status"], cli_attempt_count=attempts)
        return result

    def summary(self):
        with self.lock:
            groups = defaultdict(list)
            for item in self._results:
                groups[item["kind"]].append(item)
            per_kind = {}
            for kind, logicals in groups.items():
                latencies = [r["logical_elapsed_seconds"] for r in logicals]
                calls = [r for r in self._history if r["kind"] == kind]
                per_kind[kind] = {"logical_requests": len(logicals), "logical_status_counts": dict(Counter(r["status"] for r in logicals)),
                                  "logical_latencies_seconds": latencies,
                                  "mean_logical_latency_seconds": statistics.mean(latencies) if latencies else None,
                                  "median_logical_latency_seconds": statistics.median(latencies) if latencies else None,
                                  "cli_attempt_count": len(calls),
                                  **{key: sum(r.get(key) or 0 for r in calls) for key in ("input_tokens", "output_tokens", "cached_input_tokens")}}
            return {"transport_version": VERSION, "started_utc": self.started_utc,
                    "wall_elapsed_seconds": time.monotonic() - self.started_monotonic,
                    "cli_attempt_count": self.n_calls, "completed_cli_attempt_count": len(self._history),
                    "n_calls": self.n_calls, "logical_results": len(self._results),
                    "logical_request_count": len(self._results), "unique_request_ids": len(self._identity),
                    "first_valid_cached": len(self._cache), "failed_streak": self.failed_streak,
                    "stopped": self.stop.is_set(), "stop_reason": deepcopy(self.stop_reason),
                    "max_calls": self.max_calls, "max_logical_requests": self.max_logical_requests, "per_kind": per_kind}
