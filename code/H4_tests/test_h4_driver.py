"""Finite driver CPU mocks: no scientific transport, model fit or TEST read.

Only disposable dummy files below H4/tests/mock_outputs are created. The real
input loader, target accessor, scientific CLI, sandbox and model fitting are
all guarded or replaced before entering prepare/run.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from contextlib import ExitStack, contextmanager, nullcontext
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

ROOT = Path(__file__).resolve().parents[4]
H4 = (ROOT / "work/h-data/H4").resolve()
SRC = H4 / "src"
sys.path.insert(0, str(SRC))
import run_h4 as driver


def checked_test_path(path):
    resolved = Path(path).resolve()
    if "onedrive" in str(resolved).casefold() or not resolved.is_relative_to(H4 / "tests"):
        raise AssertionError("Test writes must stay below non-OneDrive H4/tests")
    return resolved


@contextmanager
def scratch():
    directory = checked_test_path(H4 / "tests/mock_outputs" / ("driver_" + uuid.uuid4().hex))
    checked_test_path(directory.parent).mkdir(parents=True, exist_ok=True)
    checked_test_path(directory).mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(checked_test_path(directory))


def specs60():
    schema_sha = driver.transport.digest((driver.transport.canonical(driver.prompts.FEATURE_SCHEMA) + "\n").encode())
    return [{"request_id": f"O_seed{seed}_candidate{index:02d}", "kind": "features",
        "seed": seed, "candidate_index": index, "prompt": "dummy feature request",
        "schema_sha256": schema_sha} for index in range(20) for seed in (0, 1, 2)]


def success(spec):
    return {"request_id": spec["request_id"], "status": "ok", "last_status": "ok",
        "logical_elapsed_seconds": 1., "cli_attempt_count": 1, "cache_hit": False,
        "response": {"features": [{"name": "dummy", "description": "mock only",
            "source": "def feature(s):\n    return 0.0\n"}]}}


class DummyClient:
    def __init__(self, *args, **kwargs):
        self.stop = SimpleNamespace(is_set=lambda: False)
        self.responses = {r["request_id"]: {} for r in specs60()}
        self._results = []
        self.n_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def request(self, *args, **kwargs):
        raise AssertionError("No scientific request may be executed by a driver mock")

    def summary(self):
        return {"stop_reason": None, "first_valid_cached": len(self.responses),
            "cli_attempt_count": self.n_calls}


@contextmanager
def harness(*, proceed=True, missing_result=False, duplicate_candidate=False,
            fail_pilot=False, marker=False, previous_run=False):
    with scratch() as directory, ExitStack() as stack:
        public = checked_test_path(directory / "public")
        private = checked_test_path(directory / "private")
        phase = checked_test_path(private / "phase4_1")
        for path in (public, private, phase):
            checked_test_path(path).mkdir(parents=True, exist_ok=True)
        state = SimpleNamespace(root=directory, public=public, private=private, phase=phase,
            batches=[], candidate_batches=[], selected_batches=[], refits=[], scorings=[],
            updates=[], readmes=[], client=None)
        stack.enter_context(patch.multiple(driver.c, OUT=public, DATA=private, PHASE=phase))
        cfg = {"protocol_files": {}}
        driver.c.write(public / "config.json", cfg, exclusive=True)
        if marker:
            driver.c.write(phase / "evaluation_started.json", {"stage": "4.1"}, exclusive=True)
        if previous_run:
            driver.c.write(phase / "run_started.json", {"stage": "4.1"}, exclusive=True)
        bundle = {"perseed": {seed: {"O_ids": [f"pool{seed}"]} for seed in (0, 1, 2)},
            "test_ids": [f"dummy-test{i}" for i in range(1000)]}

        def make_client(*args, **kwargs):
            state.client = DummyClient()
            return state.client

        def batch(client, specs):
            state.batches.append(list(specs))
            results = [success(spec) for spec in specs]
            if missing_result and len(state.batches) == 2:
                results.pop()
            if fail_pilot and len(state.batches) == 1:
                results[0].update(status="failed", last_status="mock_capacity_error")
            client._results.extend(results)
            client.n_calls += len(results)
            return results

        def candidates(bundle, specs, results):
            state.candidate_batches.append(list(specs))
            records = [{"seed": spec["seed"], "candidate_index": spec["candidate_index"],
                "validation_mae": .1 + spec["candidate_index"] / 1000.,
                "elapsed_seconds": 1., "sandbox_wall_seconds": .5,
                "hgb_fit_seconds": .5} for spec in specs]
            if duplicate_candidate and len(state.candidate_batches) == 2:
                record = next(item for item in records if item["seed"] == 2 and item["candidate_index"] == 19)
                record["candidate_index"] = 18
            return records

        def select(items):
            state.selected_batches.append(items)
            return min(items, key=lambda item: (item["validation_mae"], item["candidate_index"]))

        def refit(bundle, seed, selected, labels, test_ids, out, **kwargs):
            state.refits.append({"seed": seed, "selected": selected, "test_ids": list(test_ids)})
            return {"seed": seed, "candidate_index": selected["candidate_index"]}

        def score(arms, bundle, *, evaluation_marker):
            self_marker = driver.c.read(evaluation_marker)
            if (self_marker["stage"] != "4.1" or self_marker["scoring_pass"] != 1
                    or self_marker["TEST_targets_previously_read"] is not False):
                raise AssertionError("Driver did not commit the one-pass terminal gate")
            state.scorings.append({"arms": arms, "marker": self_marker})
            return {"mock_only": True, "TEST_targets_read": 0}

        def update(*args, **kwargs):
            state.updates.append((args, kwargs))
            return {"status": args[0], **kwargs}

        arms = [{"arm_id": f"O_seed{seed}", "condition": "O", "seed": seed,
            "prediction_rows": [{"id": rid, "prediction_eV_per_atom": 0.} for rid in bundle["test_ids"]]}
            for seed in (0, 1, 2)]
        state.cfg = cfg
        stack.enter_context(patch.object(driver.c, "verify", return_value={"status": "PASS"}))
        stack.enter_context(patch.object(driver.c, "RunLock", side_effect=lambda: nullcontext()))
        stack.enter_context(patch.object(driver.c, "Awake", side_effect=lambda: nullcontext()))
        stack.enter_context(patch.object(driver.c, "log"))
        stack.enter_context(patch.object(driver.c, "update", side_effect=update))
        stack.enter_context(patch.object(driver.c, "readme", side_effect=lambda status, **kwargs: state.readmes.append(status)))
        stack.enter_context(patch.object(driver, "process_identity", return_value={"pid": -1, "mock_only": True}))
        stack.enter_context(patch.object(driver.data, "load_bundle", return_value=bundle))
        stack.enter_context(patch.object(driver.data, "reveal_training", side_effect=lambda seed, **kwargs: {f"pool{seed}": 0.}))
        stack.enter_context(patch.object(driver.data, "score_truth", side_effect=AssertionError("No real TEST read")))
        stack.enter_context(patch.object(driver, "load_requests", return_value=specs60()))
        stack.enter_context(patch.object(driver.transport, "FiniteTransport", side_effect=make_client))
        stack.enter_context(patch.object(driver, "_request_batch", side_effect=batch))
        stack.enter_context(patch.object(driver, "evaluate_specs", side_effect=candidates))
        stack.enter_context(patch.object(driver.models, "evaluate_candidate", side_effect=AssertionError("No real model fit")))
        stack.enter_context(patch.object(driver.models, "select_best", side_effect=select))
        stack.enter_context(patch.object(driver.models, "refit_predict", side_effect=refit))
        stack.enter_context(patch.object(driver, "forecast", return_value={"proceed": proceed,
            "estimated_total_seconds": 1000. if proceed else 36001., "estimated_remaining_seconds": 500.}))
        stack.enter_context(patch.object(driver, "prediction_arms", return_value=arms))
        stack.enter_context(patch.object(driver, "export_predictions",
            side_effect=lambda arms: driver.c.text(public / "predictions.csv", "mock-predictions\n")))
        stack.enter_context(patch.object(driver.stats, "_normalise", return_value=arms))
        stack.enter_context(patch.object(driver.stats, "evaluate", side_effect=score))
        stack.enter_context(patch.object(driver.stats, "private_pairing_cache", return_value={}))
        stack.enter_context(patch.object(driver.models, "_write_npz"))
        stack.enter_context(patch.object(driver, "verify_terminal", return_value={"status": "PASS", "mock_only": True}))
        yield state


class RunBoundaryTests(unittest.TestCase):
    def test_complete_requires60_and20_per_seed_before_one_terminal_gate(self):
        with harness() as state:
            result = driver.run()
            self.assertEqual("complete", result["status"])
            self.assertEqual([8, 52], [len(items) for items in state.batches])
            self.assertEqual([8, 52], [len(items) for items in state.candidate_batches])
            self.assertEqual([20, 20, 20], [len(items) for items in state.selected_batches])
            self.assertTrue(all({item["candidate_index"] for item in group} == set(range(20))
                for group in state.selected_batches))
            self.assertEqual([0, 1, 2], [item["seed"] for item in state.refits])
            self.assertEqual(1, len(state.scorings))
            self.assertEqual(1000, len(state.scorings[0]["arms"][0]["prediction_rows"]))
            stop = driver.c.read(state.phase / "run_stop.json")
            self.assertEqual("complete", stop["status"])
            self.assertIs(False, stop["next_stage_started"])
            self.assertTrue((state.phase / "process_finished.json").exists())

    def test_forecast_over10h_stops_after8_without_selection_or_test(self):
        with harness(proceed=False) as state:
            result = driver.run()
            self.assertEqual("stopped_forecast_over_10h", result["status"])
            self.assertEqual([8], [len(items) for items in state.batches])
            self.assertEqual([8], [len(items) for items in state.candidate_batches])
            self.assertFalse(state.selected_batches or state.refits or state.scorings)
            self.assertFalse((state.phase / "evaluation_started.json").exists())
            self.assertEqual("stopped_forecast_over_10h", driver.c.read(state.phase / "run_stop.json")["status"])

    def test59_results_deny_selection_refit_and_terminal_scoring(self):
        with harness(missing_result=True) as state:
            result = driver.run()
            self.assertEqual("failed", result["status"])
            self.assertIn("Require60", result["error"])
            self.assertEqual([8], [len(items) for items in state.candidate_batches])
            self.assertFalse(state.selected_batches or state.refits or state.scorings)
            self.assertFalse((state.phase / "evaluation_started.json").exists())

    def test_duplicate_candidate_index_deny_terminal_even_with60_responses(self):
        with harness(duplicate_candidate=True) as state:
            result = driver.run()
            self.assertEqual("failed", result["status"])
            self.assertIn("exactly20 complete candidates", result["error"])
            self.assertFalse(state.refits or state.scorings)
            self.assertFalse((state.phase / "evaluation_started.json").exists())

    def test_existing_terminal_marker_denies_any_transport_or_test_reentry(self):
        with harness(marker=True) as state, self.assertRaises(driver.ScientificStop):
            driver.run()
        self.assertIsNone(state.client)
        self.assertFalse(state.batches or state.candidate_batches or state.scorings)

    def test_previous_started_run_requires_explicit_manual_recovery(self):
        with harness(previous_run=True) as state, self.assertRaises(driver.ScientificStop):
            driver.run()
        self.assertIsNone(state.client)
        self.assertFalse(state.batches or state.scorings)

    def test_failed_pilot_stops_before_any_candidate_fit(self):
        with harness(fail_pilot=True) as state:
            result = driver.run()
            self.assertEqual("stopped_requires_user", result["status"])
            self.assertFalse(state.candidate_batches or state.refits or state.scorings)
            self.assertEqual([8], [len(items) for items in state.batches])


class ForecastTests(unittest.TestCase):
    def measured_inputs(self):
        results = [{"logical_elapsed_seconds": 0., "cli_attempt_count": 1,
            "cache_hit": False} for _ in range(8)]
        candidates = [{"elapsed_seconds": 0., "sandbox_wall_seconds": 0.,
            "hgb_fit_seconds": 0.} for _ in range(8)]
        return results, candidates

    def test_exact10h_proceeds_and_any_excess_stops(self):
        results, candidates = self.measured_inputs()
        boundary = driver.forecast(results, candidates, 35400.)
        self.assertEqual(36000., boundary["estimated_total_seconds"])
        self.assertIs(True, boundary["proceed"])
        over = driver.forecast(results, candidates, 35400.1)
        self.assertGreater(over["estimated_total_seconds"], 36000.)
        self.assertIs(False, over["proceed"])

    def test_cache_hit_cannot_be_presented_as_measured_pilot_latency(self):
        results, candidates = self.measured_inputs()
        results[0]["cache_hit"] = True
        with self.assertRaises(driver.ScientificStop):
            driver.forecast(results, candidates, 10.)

    def test_forecast_requires_eight_complete_requests_and_candidates(self):
        results, candidates = self.measured_inputs()
        for r, c in ((results[:7], candidates), (results, candidates[:7])):
            with self.assertRaises(ValueError):
                driver.forecast(r, c, 10.)


class PrepareInvariantTests(unittest.TestCase):
    @contextmanager
    def prepare_harness(self, *, bad_schema=False):
        with scratch() as directory, ExitStack() as stack:
            public = directory / "public"
            private = directory / "private"
            phase = private / "phase4_1"
            stack.enter_context(patch.multiple(driver.c, OUT=public, DATA=private, PHASE=phase))
            requests = specs60()
            if bad_schema:
                requests[0]["schema_sha256"] = "wrong-schema"
            bundle = {"input_hashes": {}}
            state = SimpleNamespace(public=public, private=private, phase=phase, requests=requests)
            stack.enter_context(patch.object(driver.c, "verify", return_value={"status": "PASS"}))
            stack.enter_context(patch.object(driver.c, "snapshot_old", return_value={}))
            stack.enter_context(patch.object(driver.c, "log"))
            stack.enter_context(patch.object(driver.c, "readme"))
            stack.enter_context(patch.object(driver.c, "update"))
            stack.enter_context(patch.object(driver.data, "load_bundle", return_value=bundle))
            stack.enter_context(patch.object(driver.data, "configuration", return_value={"dummy": True}))
            stack.enter_context(patch.object(driver.data, "score_truth", side_effect=AssertionError("No TEST truth in prepare")))
            stack.enter_context(patch.object(driver.prompts, "build_requests", return_value=requests))
            stack.enter_context(patch.object(driver.prompts, "configuration", return_value={"dummy": True}))
            state.audit = stack.enter_context(patch.object(driver.prompts, "audit_input_bound",
                return_value={"within_budget": True, "input_token_upper_bound": 1000}))
            stack.enter_context(patch.object(driver.stats, "configuration", return_value={"dummy": True}))
            stack.enter_context(patch.object(driver.transport, "FiniteTransport",
                side_effect=AssertionError("Preparing must never construct scientific transport")))
            yield state

    def test_prepare_freezes_exact_schema_and_checks_all60_repair_suffixes(self):
        with self.prepare_harness() as state:
            self.assertEqual("PASS", driver.prepare()["status"])
            config = driver.c.read(state.public / "config.json")
            self.assertEqual(60, len(config["prompt_request_files"]))
            self.assertEqual(driver.prompts.FORMAT_REPAIR, driver.transport.FORMAT_REPAIR)
            schema = Path(config["protocol_files"]["features_schema_path"])
            self.assertEqual(driver.prompts.schema_bytes(), schema.read_bytes())
            self.assertEqual(driver.transport.canonical(driver.prompts.FEATURE_SCHEMA) + "\n", schema.read_text())
            self.assertEqual(60, state.audit.call_count)
            for call, item in zip(state.audit.call_args_list, config["prompt_request_files"]):
                request = driver.c.read(item["path"])
                self.assertEqual(request["prompt"] + driver.transport.FORMAT_REPAIR, call.args[0])
                self.assertEqual(driver.prompts.FEATURE_SCHEMA, request["schema"])
                self.assertEqual(driver.prompts.INSTRUCTIONS, request["instructions"])
                self.assertEqual(1, request["max_format_repairs"])
                self.assertIs(False, request["admissible_format_failure"])
            self.assertEqual(driver.transport.FORMAT_REPAIR, config["operational"]["format_repair_suffix"])
            self.assertEqual(driver.transport.digest(driver.transport.FORMAT_REPAIR.encode()),
                config["operational"]["format_repair_suffix_sha256"])

    def test_prepare_rejects_request_schema_hash_mismatch_before_sealing(self):
        with self.prepare_harness(bad_schema=True) as state:
            with self.assertRaisesRegex(RuntimeError, "schema bytes differ"):
                driver.prepare()
            self.assertFalse((state.private / "frozen_config.json").exists())
            self.assertFalse((state.public / "config.json").exists())


if __name__ == "__main__":
    unittest.main()
