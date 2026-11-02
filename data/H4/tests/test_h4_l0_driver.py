"""Operational driver tests with every scientific operation replaced by a CPU mock.

All fixtures are invented IDs/targets under H4/software_l0_driver. No real H2
records, generated functions, model fitting, CLI calls or TEST labels are read.
"""
from contextlib import nullcontext
from copy import deepcopy
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
import numpy as np
import run_h4_l0 as driver

LOADED_SOURCE_HASHES = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
    Path(driver.__file__), Path(driver.c.__file__), Path(driver.wire.__file__))}

H4 = (ROOT / "work/h-data/H4").resolve()
FEATURES = [{"name": "mock_feature", "description": "CPU fixture only",
             "source": "def feature(s):\n    return 1.0\n"}]
SCHEMA = {"CPU_mock_schema": True}


class DriverHarness:
    def __init__(self, directory, *, prior_calls=60, bad_query=False):
        self.directory = Path(directory).resolve()
        self.phase = self.directory / "data/phase4_2"
        self.data = self.directory / "data"
        self.out = self.directory / "out"
        self.archive = self.phase / "stage4_1_public_snapshot"
        self.prior_calls = prior_calls
        self.bad_query = bad_query
        self.requests = []
        self.model_events = []
        self.fill_events = []
        self.final_events = []
        self.score_events = []
        self.construct_events = []
        self.selection_events = []
        self.lock = threading.Lock()
        self.pool = [f"MOCK-P-{i:04d}" for i in range(2500)]
        self.test = [f"MOCK-T-{i:04d}" for i in range(1000)]
        self.targets = {rid: float(i % 13) for i, rid in enumerate(self.pool)}
        self.bundle = {"pool_ids": self.pool, "test_ids": self.test,
                       "perseed": {seed: {"S0_ids": self.pool[:500]} for seed in (0, 1, 2)}}
        self.cfg = {"prompts": {"round1_schema_example": SCHEMA}, "protocol_files": {},
                    "data": {"test_ids": self.test}}
        self.public_original = self._old_predictions()
        self.write(self.archive / "predictions.csv", self.public_original.encode())
        self.write(self.out / "predictions.csv", self.public_original.encode())
        self.write(self.data / "phase4_1/evaluation_started.json", {"stage": "4.1", "CPU_mock": True})
        self.write(self.phase / "config.json", self.cfg)
        self.write(self.out / "metrics.json", {"status": "prepared"})
        for seed in (0, 1, 2):
            labels = self.initial_labels(seed)
            initial = self._model_record(seed, None, [], labels, self.phase / "initial" / str(seed), initial=True)
            state = {"seed": seed, "condition": "L0", "round_completed": 0, "labels": labels,
                     "carry": 0, "round_model_paths": [], "query_history": [],
                     "feedback_model_path": initial["metadata_path"], "final_fill": None,
                     "status": "initialized"}
            self.write(self.phase / "states" / f"seed{seed}.json", state)
            spec = self.build_request(seed, 1, [], initial["validation_mae"], [], labels, bundle=self.bundle)
            self.write(self.phase / "initial_requests" / (spec["request_id"] + ".json"), spec)

    def safe(self, path):
        path = Path(path).resolve()
        if "onedrive" in str(path).casefold() or not path.is_relative_to(self.directory):
            raise ValueError("CPU fixture write escaped its private H4 directory")
        return path

    def write(self, path, value, *, exclusive=False):
        path = self.safe(path)
        self.safe(path.parent).mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode()
        with path.open("xb" if exclusive else "wb") as stream:
            stream.write(raw)

    def text(self, path, value, *, exclusive=False):
        self.write(path, value.encode("utf-8"), exclusive=exclusive)

    def read(self, path):
        return json.loads(self.safe(path).read_text(encoding="utf-8"))

    def _old_predictions(self):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=["arm_id", "condition", "seed", "split", "id", "prediction_eV_per_atom"], lineterminator="\n")
        writer.writeheader()
        for seed in (0, 1, 2):
            for rid in self.test:
                writer.writerow({"arm_id": f"O_seed{seed}", "condition": "O", "seed": seed,
                                 "split": "TEST", "id": rid, "prediction_eV_per_atom": 0.})
        return stream.getvalue()

    def initial_labels(self, seed, **kwargs):
        return {rid: self.targets[rid] for rid in self.pool[:500]}

    def validate_labels(self, labels, seed, *, require_n=None, **kwargs):
        assert set(self.pool[:500]) <= set(labels) <= set(self.pool)
        assert not set(labels) & set(self.test)
        assert len(labels) <= 2000
        if require_n is not None:
            assert len(labels) == require_n
        return True

    def reveal(self, ids, seed, **kwargs):
        assert set(ids) <= set(self.pool)
        return {rid: self.targets[rid] for rid in ids}

    def build_request(self, seed, round_index, features, mae, worst20, labels, *, carry=0, **kwargs):
        return {"request_id": f"L0_seed{seed}_round{round_index:02d}", "seed": seed,
                "round_index": round_index, "kind": "feature_query", "schema": deepcopy(SCHEMA),
                "capacity": 75 + carry, "prompt": "CPU mock; not sent anywhere",
                "canonical_prompt_sha256": f"mock-prompt-{seed}-{round_index}",
                "feedback_features": deepcopy(features), "feedback_mae": mae}

    def _model_record(self, seed, round_index, features, labels, directory, *, initial=False):
        path = directory / ("initial_state.json" if initial else "round.json")
        record = {"status": "complete", "phase": "initial" if initial else "round", "seed": seed,
                  "round_index": round_index, "winner_eligible": not initial,
                  "features": deepcopy(features), "active_features": deepcopy(features), "worst20": [],
                  "labelled_count": len(labels), "fit_count": len(labels) * 4 // 5,
                  "validation_count": len(labels) - len(labels) * 4 // 5,
                  "validation_mae": 100. if initial else float(round_index), "validation_group_mae": 1.,
                  "elapsed_seconds": .001, "metadata_path": str(path),
                  "identity": {"fixed_template_assignment_sha256": f"mock-fixed-seed{seed}",
                               "train_ids": sorted(labels)}, "artifacts": {}}
        sealed = driver.model_base._seal(record)
        self.write(path, sealed)
        return sealed

    def evaluate_round(self, bundle, seed, round_index, features, labels, output_dir, **kwargs):
        self.model_events.append({"seed": seed, "round_index": round_index,
                                  "count": len(labels), "features": deepcopy(features)})
        return self._model_record(seed, round_index, features, labels, output_dir)

    def select_best(self, rounds):
        self.selection_events.append([(r["round_index"], r["labelled_count"]) for r in rounds])
        return self.original_select_best(rounds)

    def select_query(self, payload, *, seed, round_index, labeled_ids, carry, **kwargs):
        capacity = 75 + carry
        n = 70 if payload["queries"] else 0
        ids = sorted(set(self.pool) - set(labeled_ids))[:n]
        return {"selected_ids": ids, "selected_n": len(ids), "carry_out": capacity - len(ids),
                "capacity": capacity, "CPU_mock": True}

    def fill(self, *, seed, labeled_ids, total_new_revealed, **kwargs):
        assert len(labeled_ids) - 500 == total_new_revealed
        ids = sorted(set(self.pool) - set(labeled_ids))[:2000 - len(labeled_ids)]
        self.fill_events.append({"seed": seed, "before": len(labeled_ids), "fill": len(ids),
                                 "models_complete": len(self.model_events)})
        return {"selected_ids": ids, "selected_n": len(ids), "CPU_mock": True}

    def finalize(self, bundle, seed, selected, labels, test_ids, output_dir, **kwargs):
        assert len(labels) == 2000 and test_ids == self.test
        self.final_events.append({"seed": seed, "winner_round": selected["round_index"],
                                  "labelled_count": len(labels), "models_complete": len(self.model_events)})
        path = self.safe(output_dir / "predictions.npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, ids=np.asarray(test_ids), predictions=np.full(1000, seed / 10.))
        return {"seed": seed, "selected_round_index": selected["round_index"],
                "selected_validation_mae": selected["validation_mae"],
                "full_train_features": {"kept_names": ["mock_feature"]},
                "artifacts": {"predictions": {"path": str(path), "sha256": driver.c.sha(path)}}}

    def score(self, arms, bundle, *, evaluation_marker):
        marker = self.read(evaluation_marker)
        assert marker["stage"] == "4.2" and marker["current_L0_scoring_pass"] == 1
        assert marker["TEST_targets_previously_read_for_this_condition"] is False
        assert len(self.final_events) == 3 and len(self.model_events) == 60
        self.score_events.append({"CPU_mock_only": True, "marker": str(evaluation_marker)})
        return {"CPU_mock_only": True, "TEST_target_accessor_calls": 0}

    def report(self, *args):
        path = self.phase / "report/CPU_mock_report.txt"
        self.text(path, "CPU mock only; no scientific results\n")
        return {"report": str(path)}

    def transport_class(self):
        harness = self
        class TransportMock:
            def __init__(self, runtime, data_dir, out_dir, *args, **kwargs):
                harness.construct_events.append({"data_dir": Path(data_dir), "out_dir": Path(out_dir), **kwargs})
                self.n_calls = harness.prior_calls
                self.stop = threading.Event()
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def request(self, spec, validator):
                with harness.lock:
                    self.n_calls += 1
                    harness.requests.append(deepcopy(spec))
                payload = {"features": deepcopy(FEATURES), "queries": [{"n": 75}], "rationale": "CPU mock"}
                common = {"request_id": spec["request_id"], "logical_elapsed_seconds": .001,
                          "cli_attempt_count": 1, "cache_hit": False, "response_sha256": "CPU-mock-response"}
                if harness.bad_query:
                    payload["queries"] = [{"n": "invalid-after-repair"}]
                    return {**common, "status": "failed", "last_status": "format_invalid",
                            "admissible_format_failure": True, "raw_response_text": json.dumps(payload),
                            "validation_error": "CPU mock bad query after one repair"}
                return {**common, "status": "ok", "response": payload}
            def summary(self):
                return {"cli_attempt_count": self.n_calls, "stop_reason": None}
        return TransportMock

    def patches(self):
        self.original_select_best = driver.models.select_best
        return [
            patch.multiple(driver.c, PHASE=self.phase, DATA=self.data, OUT=self.out, ARCHIVE=self.archive,
                write=self.write, text=self.text, read=self.read, log=lambda *a: None,
                verify=lambda *a, **kw: {"status": "PASS", "CPU_mock": True},
                process_identity=lambda: {"pid": -1, "CPU_mock": True},
                RunLock=lambda: nullcontext(), Awake=lambda: nullcontext()),
            patch.object(driver.c.old, "bytes_write", self.write),
            patch.multiple(driver.data, load_bundle=lambda: self.bundle, initial_labels=self.initial_labels,
                           reveal=self.reveal, validate_labels=self.validate_labels),
            patch.multiple(driver.prompts, build_request=self.build_request,
                           validator=lambda *a, **kw: (lambda payload: True), validate_response=lambda *a, **kw: True),
            patch.multiple(driver.queries, select_only_no_reveal=self.select_query, final_fill_only_no_reveal=self.fill),
            patch.multiple(driver.models, evaluate_round=self.evaluate_round, select_best=self.select_best, finalize=self.finalize),
            patch.object(driver.wire, "StrictSchemaTransport", self.transport_class()),
            patch.multiple(driver.stats, _normalise=lambda *a, **kw: None, evaluate=self.score, private_pairing_cache=lambda: {}),
            patch.object(driver, "curve_and_report", self.report),
            patch.object(driver, "verify_terminal", lambda: {"status": "PASS", "CPU_mock": True})]


class Tests(unittest.TestCase):
    def setUp(self):
        top = (H4 / "software_l0_driver").resolve()
        assert "onedrive" not in str(top).casefold() and top.is_relative_to(H4)
        top.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=top)
        self.addCleanup(self.temporary.cleanup)

    def harness(self, **kwargs):
        harness = DriverHarness(self.temporary.name, **kwargs)
        for context in harness.patches():
            context.start()
            self.addCleanup(context.stop)
        return harness

    def test_two_round_pilot_over_ten_hours_stops_before_round3_fill_or_TEST(self):
        harness = self.harness()
        with patch.object(driver, "forecast", return_value={"estimated_total_seconds": 36001., "proceed": False}):
            result = driver.run()
        self.assertEqual(result["status"], "stopped_forecast_over_10h")
        self.assertEqual(len(harness.requests), 6)
        self.assertEqual({r["round_index"] for r in harness.requests}, {1, 2})
        self.assertEqual(len(harness.model_events), 6)
        self.assertEqual(harness.fill_events, [])
        self.assertEqual(harness.final_events, [])
        self.assertEqual(harness.score_events, [])
        self.assertFalse((harness.phase / "evaluation_started.json").exists())
        self.assertTrue((harness.phase / "process_finished.json").exists())

    def test_shared_prior60_cap300_and_full20x3_then_fill_without_reselection(self):
        harness = self.harness()
        result = driver.run()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(harness.requests), 60)
        self.assertEqual(len({r["request_id"] for r in harness.requests}), 60)
        self.assertEqual(len(harness.model_events), 60)
        self.assertEqual(len(harness.selection_events), 3)
        self.assertTrue(all([r for r, count in rounds] == list(range(1, 21)) for rounds in harness.selection_events))
        self.assertTrue(all(event["models_complete"] == 60 for event in harness.fill_events))
        self.assertEqual([event["fill"] for event in harness.fill_events], [100, 100, 100])
        self.assertTrue(all(event["winner_round"] == 1 for event in harness.final_events))
        self.assertTrue(all(event["labelled_count"] == 2000 for event in harness.final_events))
        self.assertEqual(len(harness.score_events), 1)
        constructed = harness.construct_events[0]
        self.assertEqual(constructed["data_dir"], harness.data / "transport")
        self.assertEqual((constructed["max_calls"], constructed["max_logical_requests"], constructed["max_workers"]), (300, 180, 4))
        metrics = harness.read(harness.out / "metrics.json")
        self.assertEqual((metrics["LLM_calls_global"], metrics["LLM_calls_stage4_2"]), (120, 60))
        self.assertEqual(metrics["current_L0_scoring_pass"], 1)
        self.assertFalse(metrics["next_stage_started"])
        self.assertTrue((harness.out / "predictions.csv").read_bytes().startswith(harness.public_original.encode()))
        with (harness.out / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6000)
        self.assertEqual(len([row for row in rows if row["condition"] == "L0"]), 3000)

    def test_global_ledger_missing_prior60_stops_without_dispatch(self):
        harness = self.harness(prior_calls=0)
        result = driver.run()
        self.assertEqual(result["status"], "failed")
        self.assertIn("exactly60 old calls", result["error"])
        self.assertEqual(harness.requests, [])
        self.assertEqual(harness.model_events, [])
        self.assertEqual(harness.score_events, [])

    def test_valid_features_bad_query_after_repair_becomes_empty_query_and_carry(self):
        harness = self.harness(bad_query=True)
        with patch.object(driver, "forecast", return_value={"estimated_total_seconds": 36001., "proceed": False}):
            result = driver.run()
        self.assertEqual(result["status"], "stopped_forecast_over_10h")
        for seed in (0, 1, 2):
            state = harness.read(harness.phase / "states" / f"seed{seed}.json")
            self.assertEqual((len(state["labels"]), state["carry"], state["round_completed"]), (500, 150, 2))
            self.assertTrue(all(record["query_response_audit"]["empty_query_fallback"] for record in state["query_history"]))
            self.assertTrue(all(record["query_payload"]["queries"] == [] for record in state["query_history"]))
        self.assertTrue(all(event["features"] == FEATURES for event in harness.model_events))

    def test_invalid_features_after_repair_never_reuse_feedback(self):
        harness = self.harness()
        bad = {"status": "failed", "last_status": "format_invalid", "admissible_format_failure": True,
               "raw_response_text": json.dumps({"features": [], "queries": [], "rationale": ""})}
        with self.assertRaises((ValueError, driver.ScientificStop)):
            driver.resolve_response(bad, bundle=harness.bundle, capacity=75)
        self.assertEqual(harness.model_events, [])
        self.assertEqual(harness.final_events, [])

    def test_existing_terminal_gate_refuses_reentry_without_calls_or_scoring(self):
        harness = self.harness()
        harness.write(harness.phase / "evaluation_started.json", {"stage": "4.2", "scoring_pass": 1})
        with self.assertRaisesRegex(driver.ScientificStop, "manual recovery required"):
            driver.run()
        self.assertEqual(harness.construct_events, [])
        self.assertEqual(harness.requests, [])
        self.assertEqual(harness.score_events, [])

    def test_preservation_prefix_accepts_only_append_and_rejects_old_byte_mutation(self):
        harness = self.harness()
        for name in ("llm_calls.jsonl", "run.log", "predictions.csv"):
            path = harness.out / name
            initial = b"CPU-mock-original-prefix\n"
            harness.write(path, initial)
            record = {"size": len(initial), "sha256": hashlib.sha256(initial).hexdigest()}
            self.assertTrue(driver.c.prefix_matches(path, record))
            harness.write(path, initial + b"new-L0-tail\n")
            self.assertTrue(driver.c.prefix_matches(path, record))
            harness.write(path, b"X" + initial[1:] + b"new-L0-tail\n")
            self.assertFalse(driver.c.prefix_matches(path, record))
            harness.write(path, initial[:-1])
            self.assertFalse(driver.c.prefix_matches(path, record))

    def test_actual_shared_transport_cpu_seam_restores60_and_enforces300_total(self):
        directory = (Path(self.temporary.name) / "actual_transport_CPU_only").resolve()
        assert "onedrive" not in str(directory).casefold() and directory.is_relative_to(H4)
        directory.mkdir(parents=True)
        calls = []
        def backend(prompt, **kwargs):
            metadata = kwargs["metadata"]
            calls.append((metadata["request_id"], metadata["call_index"]))
            invalid_first = metadata["request_id"].startswith("new_") and metadata["format_round"] == 0
            return {**metadata, "status": "ok", "elapsed_seconds": .001,
                    "response_text": '{"value":"bad-format"}' if invalid_first else '{"value":1}',
                    "input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0}
        runtime = {"executable": "CPU-MOCK-NO-CLI", "sha256": "mock", "cli_version": "mock",
                   "model": "gpt-6.1-sol", "reasoning_effort": "high"}
        schema = {"type": "object", "properties": {"value": {"type": "number"}},
                  "required": ["value"], "additionalProperties": False}
        def spec(rid):
            return {"request_id": rid, "kind": "CPU-transport-test", "prompt": "CPU mock only", "schema": schema}
        arguments = (runtime, directory / "shared", directory / "out")
        with driver.wire.StrictSchemaTransport(*arguments, call_once_fn=backend, max_calls=300,
                                                max_logical_requests=180) as previous:
            for index in range(60):
                self.assertEqual(previous.request(spec("old_" + str(index)))["status"], "ok")
            self.assertEqual(previous.n_calls, 60)
        ledger = directory / "out/llm_calls.jsonl"
        prefix = ledger.read_bytes()
        with driver.wire.StrictSchemaTransport(*arguments, call_once_fn=backend, max_calls=300,
                                                max_logical_requests=180) as current:
            self.assertEqual(current.n_calls, 60)
            self.assertTrue(current.request(spec("old_0"))["cache_hit"])
            for index in range(120):
                result = current.request(spec("new_" + str(index)))
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["cli_attempt_count"], 2)
            self.assertEqual(current.n_calls, 300)
            result = current.request(spec("must_not_dispatch"))
            self.assertEqual(result["last_status"], "undispatched")
            self.assertEqual(current.stop_reason, "finite_global_cli_call_cap")
        self.assertEqual(len(calls), 300)
        self.assertEqual(calls[-1][1], 300)
        self.assertTrue(ledger.read_bytes().startswith(prefix))


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                    for path, expected in LOADED_SOURCE_HASHES.items())
    audit = {"status": "PASS" if result.wasSuccessful() and unchanged else "FAIL",
             "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
             "elapsed_seconds": time.monotonic() - started, "real_LLM_calls": 0,
             "real_generated_feature_executions": 0, "real_model_fit_predict_calls": 0,
             "real_TEST_target_fields_read": 0,
             "CPU_mock_backend_calls_in_shared_cap_test": 300,
             "loaded_source_hashes": LOADED_SOURCE_HASHES, "loaded_source_files_unchanged": unchanged,
             "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination = (H4 / "software_l0_driver/last_audit.json").resolve()
    assert "onedrive" not in str(destination).casefold() and destination.is_relative_to(H4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({**audit, "audit_path": str(destination)}, ensure_ascii=False))
    raise SystemExit(0 if audit["status"] == "PASS" else 1)
