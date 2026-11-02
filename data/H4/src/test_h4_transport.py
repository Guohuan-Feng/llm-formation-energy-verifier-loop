"""CPU-only transport tests. No real CLI request, fit, predict or API call."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
import unittest
import uuid


SOURCE = Path(__file__).resolve().with_name("h4_transport.py")
definition = importlib.util.spec_from_file_location("h4_transport_under_test", SOURCE)
t = importlib.util.module_from_spec(definition)
definition.loader.exec_module(t)
TEST_ROOT = t.safe(t.ROOT / "work/h-data/H4/software_transport", must_exist=True)
RUNTIME = {"executable": "CPU-MOCK-NOT-A-CLI", "sha256": "mock", "cli_version": "mock",
           "model": t.MODEL, "reasoning_effort": "high"}
SCHEMA = {"type": "object", "properties": {"value": {"type": "number"}},
          "required": ["value"], "additionalProperties": False}


def private_case(name):
    path = t.safe(TEST_ROOT / (name + "_" + uuid.uuid4().hex))
    path.mkdir(parents=True)
    return path


def spec(request_id="r1", **updates):
    result = {"request_id": request_id, "kind": "eval", "prompt": "CPU mock prompt", "schema": deepcopy(SCHEMA)}
    result.update(updates)
    return result


def good(prompt, **kwargs):
    return {**kwargs["metadata"], "status": "ok", "response_text": '{"value":1}', "elapsed_seconds": 0.01,
            "input_tokens": 20, "output_tokens": 5, "cached_input_tokens": 4}


def sequence(records):
    calls = []
    lock = threading.Lock()

    def call(prompt, **kwargs):
        with lock:
            index = len(calls)
            calls.append((prompt, kwargs))
            selected = records[min(index, len(records) - 1)]
        result = good(prompt, **kwargs)
        result.update(selected)
        return result

    return call, calls


class FiniteTests(unittest.TestCase):
    def make(self, name, call=good, **kwargs):
        case = private_case(name)
        transport = t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=call, **kwargs)
        self.addCleanup(lambda: transport.close() if transport._owner_handle is not None else None)
        return transport, case

    def test_valid_response_is_cached_and_only_one_call(self):
        call, calls = sequence([{}])
        transport, _ = self.make("cached", call)
        first = transport.request(spec())
        second = transport.request(spec())
        self.assertEqual(first["response"], {"value": 1})
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["cli_attempt_count"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(transport.cached(spec())["response"], {"value": 1})
        self.assertEqual(transport.summary()["logical_results"], 1)
        returned = transport.responses
        returned["r1"]["value"] = 99
        self.assertEqual(transport.responses["r1"]["value"], 1)

    def test_cached_request_fingerprint_collision_stops(self):
        transport, _ = self.make("identity")
        transport.request(spec())
        with self.assertRaisesRegex(ValueError, "different frozen payload"):
            transport.request(spec(prompt="different prompt"))
        self.assertTrue(transport.stop.is_set())
        self.assertEqual(transport.n_calls, 1)

    def test_same_prompt_different_id_remains_independent(self):
        transport, _ = self.make("independent")
        transport.request(spec("a"))
        transport.request(spec("b"))
        self.assertEqual(transport.n_calls, 2)

    def test_restart_uses_first_valid_public_ledger(self):
        call, calls = sequence([{}])
        transport, case = self.make("resume", call)
        first = transport.request(spec())
        transport.close()
        resumed = t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=call)
        self.addCleanup(resumed.close)
        self.assertTrue(resumed.request(spec())["cache_hit"])
        self.assertEqual(resumed.request(spec())["response_sha256"], first["response_sha256"])
        self.assertEqual(len(calls), 1)

    def test_recover_success_if_private_append_was_interrupted(self):
        call, calls = sequence([{}])
        transport, case = self.make("public_recover", call)
        transport.request(spec())
        transport.close()
        t.safe(case / "private/request_results.jsonl").write_text("", encoding="utf-8")
        resumed = t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=call)
        self.addCleanup(resumed.close)
        self.assertTrue(resumed.request(spec())["cache_hit"])
        self.assertEqual(len(calls), 1)

    def test_orphan_reservation_is_counted_and_failed_request_can_resume(self):
        transport, case = self.make("orphan")
        request = spec()
        _, fingerprint, _, _, _ = transport._prepare(request)
        transport._bind("r1", fingerprint)
        transport._reserve(request, fingerprint, 0, 0, request["prompt"])
        transport.close()
        resumed = t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=good)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.n_calls, 1)
        result = resumed.request(request)
        self.assertEqual(result["last_call_index"], 2)

    def test_format_only_repair_once_and_finite_schema(self):
        call, calls = sequence([{"response_text": '{"value":NaN}'}, {}])
        transport, _ = self.make("format", call)
        result = transport.request(spec())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], calls[0][0] + t.FORMAT_REPAIR)
        self.assertEqual(result["cli_attempt_count"], 2)
        self.assertEqual(transport.failed_streak, 0)

    def test_invalid_schema_then_failed_at_most_two_attempts(self):
        call, calls = sequence([{"response_text": '{"value":"bad"}'}])
        transport, _ = self.make("invalid_schema", call)
        result = transport.request(spec())
        self.assertEqual(result["last_status"], "format_invalid")
        self.assertIsNotNone(result["validation_error"])
        self.assertEqual(len(calls), 2)

    def test_query_semantic_invalid_is_admissible_and_never_cached_success(self):
        call, calls = sequence([{}])
        transport, _ = self.make("admissible", call)

        def invalid(value):
            raise ValueError("Invalid element does not occur in POOL")

        for index in range(4):
            result = transport.request(spec(str(index), kind="query", max_format_repairs=0,
                                            admissible_format_failure=True), validator=invalid)
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["admissible_format_failure"])
        self.assertFalse(transport.stop.is_set())
        self.assertEqual(transport.failed_streak, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(transport.responses, {})

    def test_admissible_query_does_not_exempt_cli_failures(self):
        call, _ = sequence([{"status": "cli_error"}])
        transport, _ = self.make("query_cli_failure", call)
        for index in range(3):
            transport.request(spec(str(index), kind="query", max_format_repairs=0, admissible_format_failure=True))
        self.assertTrue(transport.stop.is_set())
        self.assertEqual(transport.failed_streak, 3)

    def test_admissible_format_does_not_reset_previous_failure_streak(self):
        call, _ = sequence([{"status": "cli_error"}, {"response_text": "bad"}, {"status": "cli_error"}])
        transport, _ = self.make("preserve_streak", call)
        transport.request(spec("a"))
        transport.request(spec("q", max_format_repairs=0, admissible_format_failure=True))
        self.assertEqual(transport.failed_streak, 1)
        transport.request(spec("b"))
        self.assertEqual(transport.failed_streak, 2)

    def test_final429_same_request_backoff_bounded_and_recorded(self):
        call, calls = sequence([{"status": "rate_limited_final"}])
        delays = []
        transport, _ = self.make("rate_limit", call, wait_fn=lambda delay: delays.append(delay) or False)
        result = transport.request(spec())
        self.assertEqual(delays, [60, 120, 240])
        self.assertEqual(len(calls), 4)
        self.assertEqual(len({prompt for prompt, _ in calls}), 1)
        self.assertEqual(result["rate_limit_transport_retries"], 3)
        self.assertEqual(transport.failed_streak, 1)

    def test_rate_retry_budget_is_shared_across_format_repair(self):
        call, calls = sequence([{"status": "rate_limited_final"}, {"response_text": "bad"},
                                {"status": "rate_limited_final"}, {"status": "rate_limited_final"},
                                {"status": "rate_limited_final"}])
        transport, _ = self.make("shared_rate", call, wait_fn=lambda delay: False)
        result = transport.request(spec())
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["rate_limit_transport_retries"], 3)

    def test_usage_limit_and_integrity_stop_immediately(self):
        for status in ("terminal_usage_limit", "tool_activity_rejected", "model_mismatch_rejected"):
            call, calls = sequence([{"status": status}])
            transport, _ = self.make("terminal_" + status, call)
            transport.request(spec("first", admissible_format_failure=True))
            result = transport.request(spec("next"))
            self.assertTrue(transport.stop.is_set())
            self.assertEqual(result["last_status"], "undispatched")
            self.assertEqual(len(calls), 1)

    def test_global_finite_cli_and_logical_caps(self):
        transport, _ = self.make("cap", max_calls=1)
        transport.request(spec("one"))
        result = transport.request(spec("two"))
        self.assertEqual(transport.n_calls, 1)
        self.assertEqual(result["last_status"], "undispatched")
        self.assertEqual(transport.stop_reason, "finite_global_cli_call_cap")
        second, _ = self.make("logical_cap", max_logical_requests=1)
        second.request(spec("one"))
        second.request(spec("two"))
        self.assertEqual(second.n_calls, 1)
        self.assertEqual(second.stop_reason, "finite_global_logical_request_cap")

    def test_parallel_duplicate_ids_coalesce_success_and_failure(self):
        for final_status in ("ok", "cli_error"):
            entered = threading.Event()
            release = threading.Event()
            count = []

            def blocked(prompt, **kwargs):
                count.append(1)
                entered.set()
                release.wait(3)
                return {**good(prompt, **kwargs), "status": final_status}

            transport, _ = self.make("coalesce_" + final_status, blocked)
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(transport.request, spec())
                self.assertTrue(entered.wait(1))
                second = pool.submit(transport.request, spec())
                time.sleep(0.04)
                release.set()
                first_result, second_result = first.result(), second.result()
            self.assertEqual(len(count), 1)
            self.assertEqual(first_result["status"], second_result["status"])
            self.assertEqual(transport.n_calls, 1)

    def test_four_worker_max_and_every_reservation_has_a_result(self):
        active = maximum = 0
        lock = threading.Lock()

        def parallel(prompt, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.03)
            with lock:
                active -= 1
            return good(prompt, **kwargs)

        transport, case = self.make("four_workers", parallel)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(transport.request, [spec(str(i)) for i in range(12)]))
        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertLessEqual(maximum, 4)
        self.assertGreater(maximum, 1)
        records = t._read_lines(case / "out/llm_calls.jsonl")
        self.assertEqual(len(records), 24)
        self.assertEqual(len({r["call_index"] for r in records}), 12)
        self.assertEqual(transport.summary()["per_kind"]["eval"]["input_tokens"], 240)

    def test_cache_validated_against_semantics_and_schema_on_reuse(self):
        transport, _ = self.make("cached_semantics")
        transport.request(spec())
        with self.assertRaisesRegex(ValueError, "Semantic validator"):
            transport.request(spec(), validator=lambda value: False)
        self.assertTrue(transport.stop.is_set())

    def test_owner_lock_prevents_second_coordinator(self):
        transport, case = self.make("owner_lock")
        with self.assertRaisesRegex(RuntimeError, "Another H4 transport"):
            t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=good)
        transport.close()
        replacement = t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=good)
        replacement.close()

    def test_mutated_or_truncated_ledger_fails_closed(self):
        transport, case = self.make("ledger_tamper")
        transport.request(spec())
        transport.close()
        path = t.safe(case / "out/llm_calls.jsonl")
        records = t._read_lines(path)
        records[-1]["validated_response"]["value"] = 7
        path.write_text("".join(t.canonical(x) + "\n" for x in records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            t.FiniteTransport(RUNTIME, case / "private", case / "out", call_once_fn=good)

    def test_no_deadline_is_passed_and_protocol_files_private_immutable(self):
        call, calls = sequence([{}])
        transport, case = self.make("protocol", call)
        transport.request(spec())
        kwargs = calls[0][1]
        self.assertIsNone(kwargs["deadline_seconds"])
        for key in ("schema_path", "instructions_path", "catalog_path"):
            self.assertTrue(kwargs[key].is_relative_to(case / "private"))
        schema_path = t.safe(kwargs["schema_path"])
        schema_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Immutable protocol file changed"):
            transport.request(spec("other"))

    def test_simple_json_schema_grammar_and_finite_validator(self):
        schema = {"type": "object", "properties": {"queries": {"type": "array", "maxItems": 3,
                    "items": {"type": "object", "properties": {"n": {"type": "integer", "minimum": 0},
                    "include_all": {"type": "array", "items": {"type": "string", "enum": ["O", "Fe"]}}},
                    "required": ["n"], "additionalProperties": False}}, "rationale": {"type": "string", "maxLength": 200}},
                  "required": ["queries", "rationale"], "additionalProperties": False}
        value = t.validate_response('{"queries":[{"n":60,"include_all":["O"]}],"rationale":"CPU"}', schema)
        self.assertEqual(value["queries"][0]["n"], 60)
        for text in ('{"queries":[{"n":true}],"rationale":"x"}', '{"queries":[{"n":1,"include_all":["Bad"]}],"rationale":"x"}'):
            with self.assertRaises(ValueError):
                t.validate_response(text, schema)
        with self.assertRaises(ValueError):
            t.validate_response('{"value":1e999}', SCHEMA)

    def test_production_adapter_isolated_globals_without_calling_cli(self):
        case = private_case("binding")
        old_h3 = (t.ROOT / "h3_resume_transport.py").read_bytes()
        call, e2 = t._production_call_once(RUNTIME, case)
        self.assertEqual(call.__globals__["c"].DATA, case)
        self.assertEqual(call.__globals__["_base"]._e2, e2)
        self.assertEqual(e2.INPUT_LIMIT, 20_000)
        self.assertEqual(e2.MODEL, "gpt-6.1-sol")
        self.assertEqual(call.__kwdefaults__["deadline_seconds"], None)
        self.assertEqual((t.ROOT / "h3_resume_transport.py").read_bytes(), old_h3)

    def test_non_onedrive_and_closed_transport_guards(self):
        with self.assertRaisesRegex(ValueError, "OneDrive"):
            t.safe(Path(r'C:\Users\28908\OneDrive\would_not_write.txt'))
        transport, _ = self.make("closed")
        transport.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            transport.request(spec())


if __name__ == "__main__":
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FiniteTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    audit = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
             "failures": len(result.failures), "errors": len(result.errors), "elapsed_seconds": time.monotonic() - started,
             "real_llm_calls": 0, "real_model_fit_predict_calls": 0, "source_sha256": t.file_sha(SOURCE),
             "test_source_sha256": t.file_sha(Path(__file__)), "completed_utc": t.now()}
    destination = t.safe(TEST_ROOT / ("audit_" + uuid.uuid4().hex + ".json"))
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "audit_path": str(destination)}, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
