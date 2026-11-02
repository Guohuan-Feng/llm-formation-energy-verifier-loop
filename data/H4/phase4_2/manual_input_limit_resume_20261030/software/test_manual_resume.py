"""Manual continuation software mocks; no scientific backend/target access."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/h-data/H4/tests"))
from test_h4_l0_driver import DriverHarness, FEATURES

SOURCE = HERE / "resume_l0_input_limit.py"
definition = importlib.util.spec_from_file_location("manual_l0_under_test", SOURCE)
m = importlib.util.module_from_spec(definition)
definition.loader.exec_module(m)
LOADED_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    def setUp(self):
        top = (HERE / "software/fixtures").resolve()
        assert "onedrive" not in str(top).casefold() and top.is_relative_to(HERE)
        top.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=top)
        self.addCleanup(self.temporary.cleanup)

    def harness(self):
        harness = DriverHarness(self.temporary.name, prior_calls=78)
        baseline = {}
        for seed in (0, 1, 2):
            state = harness.read(harness.phase / "states" / f"seed{seed}.json")
            for rnd in range(1, 7):
                previous = len(state["labels"])
                chosen = harness.pool[previous:previous + 75]
                state["labels"].update({rid: harness.targets[rid] for rid in chosen})
                model = harness.evaluate_round(harness.bundle, seed, rnd, FEATURES, state["labels"],
                          harness.phase / "rounds" / f"L0_seed{seed}" / f"round{rnd:02d}")
                state["round_model_paths"].append(model["metadata_path"])
                state["query_history"].append({"round_index": rnd, "query_selection": {
                    "selected_ids": chosen, "selected_n": 75, "carry_in": 0, "carry_out": 0, "capacity": 75}})
                state.update(round_completed=rnd, feedback_model_path=model["metadata_path"])
            harness.write(harness.phase / "states" / f"seed{seed}.json", state)
            baseline[seed] = deepcopy(state)
        for name in ("run_started.json", "run_stop.json", "process_finished.json"):
            harness.write(harness.phase / name, {"CPU_mock_old_record": name})
        harness.write(harness.phase / "pilot_forecast.json", {"estimated_total_seconds": 9000., "proceed": True})
        old_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in harness.phase.rglob("*") if path.is_file()}
        for context in harness.patches():
            context.start(); self.addCleanup(context.stop)
        old_mock = m.original.wire.StrictSchemaTransport
        class ContinuationTransportMock(old_mock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.responses = {"old_CPU_success_" + str(i): {} for i in range(78)}
        context = patch.object(m.original.wire, "StrictSchemaTransport", ContinuationTransportMock)
        context.start(); self.addCleanup(context.stop)
        manual_dir = harness.phase / "manual_test"
        manifest_path = manual_dir / "manifest.json"
        manifest = {"historical_run_elapsed_seconds": 1475.047}
        harness.write(manifest_path, manifest)
        def fake_verify(*args, **kwargs):
            live = {seed: harness.read(harness.phase / "states" / f"seed{seed}.json") for seed in (0, 1, 2)}
            return manifest, harness.cfg, deepcopy(baseline), live
        contexts = [patch.object(m, "HERE", manual_dir), patch.object(m, "verify_manifest", fake_verify),
            patch.object(m, "assert_owned_limit", lambda client: None),
            patch.object(m.data, "_REVEALED", {}),
            patch.object(m.data, "initial_labels", side_effect=AssertionError("Old initial labels must not be replayed")),
            patch.object(m.models, "initial_state", side_effect=AssertionError("Old initialization must not be refit")),
            patch.object(m.original, "run", side_effect=AssertionError("Old guarded run must not be called")),
            patch.object(m.prompts, "INPUT_LIMIT", 20000), patch.object(m.transport, "INPUT_LIMIT", 20000)]
        for context in contexts:
            context.start(); self.addCleanup(context.stop)
        def reveal(ids, seed, **kwargs):
            observed = m.data._REVEALED[seed]
            assert set(baseline[seed]["labels"]) <= set(observed)
            fresh = harness.reveal(ids, seed, **kwargs)
            observed.update(fresh)
            assert len(observed) <= 2000
            return fresh
        context = patch.object(m.data, "reveal", reveal)
        context.start(); self.addCleanup(context.stop)
        return harness, baseline, manifest_path, old_hashes

    def test_only_round7_to20_new78_ledger_and_original_stop_records_unchanged(self):
        harness, baseline, manifest, old_hashes = self.harness()
        result = m.run(manifest)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(len(harness.requests), 42)
        self.assertEqual({spec["round_index"] for spec in harness.requests}, set(range(7, 21)))
        self.assertTrue(all(m.NEW_REQUEST.fullmatch(spec["request_id"]) for spec in harness.requests))
        self.assertEqual(len(harness.model_events), 60)
        self.assertTrue(all(event["round_index"] >= 7 for event in harness.model_events[18:]))
        self.assertEqual(len(harness.selection_events), 3)
        self.assertEqual(len(harness.score_events), 1)
        self.assertTrue(all(event["winner_round"] == 1 for event in harness.final_events))
        self.assertEqual(m.prompts.INPUT_LIMIT, 34000)
        self.assertEqual(m.transport.INPUT_LIMIT, 34000)
        self.assertEqual(result["LLM_calls_global"], 120)
        for seed in (0, 1, 2):
            self.assertEqual(len(m.data._REVEALED[seed]), 2000)
            current = harness.read(harness.phase / "states" / f"seed{seed}.json")
            self.assertEqual(current["query_history"][:6], baseline[seed]["query_history"])
        for name in ("run_started.json", "run_stop.json", "process_finished.json"):
            path = harness.phase / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), old_hashes[str(path)])
            self.assertTrue((manifest.parent / name).exists())
        for path, expected in old_hashes.items():
            if "/rounds/" in Path(path).as_posix():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), expected)
        metrics = harness.read(harness.out / "metrics.json")
        self.assertIsNone(metrics["elapsed_seconds"])
        self.assertIsNone(metrics["historical_total_active_seconds"])
        self.assertEqual(metrics["historical_recorded_run_elapsed_seconds"], 1475.047)
        self.assertIn("manual_elapsed_seconds", metrics)
        self.assertLessEqual(len((harness.out / "README.md").read_text().splitlines()), 20)

    def test_manual_run_guard_refuses_replay_without_requests(self):
        harness, baseline, manifest, old_hashes = self.harness()
        harness.write(manifest.parent / "run_started.json", {"old_manual_attempt": True})
        with self.assertRaisesRegex(m.original.ScientificStop, "no automatic replay"):
            m.run(manifest)
        self.assertEqual(harness.requests, [])
        self.assertEqual(harness.construct_events, [])

    def test_budget_failure_stops_new_manual_only_and_clears_no_cached_work(self):
        harness, baseline, manifest, old_hashes = self.harness()
        with patch.object(m.original, "request_round", side_effect=m.prompts.PromptBudgetExceeded("CPU mock bound>34000")):
            result = m.run(manifest)
        self.assertEqual(result["status"], "stopped_requires_user", result)
        self.assertEqual(harness.requests, [])
        self.assertEqual(len(harness.model_events), 18)
        self.assertEqual(harness.score_events, [])
        self.assertTrue((manifest.parent / "run_stop.json").exists())
        self.assertFalse((harness.phase / "evaluation_started.json").exists())
        for name in ("run_started.json", "run_stop.json", "process_finished.json"):
            path = harness.phase / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), old_hashes[str(path)])

    def test_limit_changes_no_prompt_strings_schema_or_feature_source_maxlength(self):
        before = [m.prompts.PREFIX, m.prompts.DICTIONARY_CONTRACT, m.prompts.ACTIVE_FEATURE_POLICY,
                  m.prompts.INSTRUCTIONS, deepcopy(m.prompts.feature_contract.FEATURE_SCHEMA)]
        with patch.object(m.prompts, "INPUT_LIMIT", 20000), patch.object(m.transport, "INPUT_LIMIT", 20000):
            m.install_limit()
            self.assertEqual((m.prompts.INPUT_LIMIT, m.transport.INPUT_LIMIT), (34000, 34000))
        after = [m.prompts.PREFIX, m.prompts.DICTIONARY_CONTRACT, m.prompts.ACTIVE_FEATURE_POLICY,
                 m.prompts.INSTRUCTIONS, m.prompts.feature_contract.FEATURE_SCHEMA]
        self.assertEqual(before, after)
        self.assertEqual(after[-1]["properties"]["features"]["items"]["properties"]["source"]["maxLength"], 20000)

    def test_actual_frozen_owned_binding_gets34000_without_calling_CLI(self):
        directory = (Path(self.temporary.name) / "owned_binding").resolve()
        assert "onedrive" not in str(directory).casefold() and directory.is_relative_to(HERE)
        directory.mkdir()
        with patch.object(m.transport, "INPUT_LIMIT", 34000):
            call, e2 = m.transport._production_call_once(m.c.RUNTIME, directory)
        client = types.SimpleNamespace(_canonical_owned_call_once=call, _e2=e2,
                                       max_calls=300, max_logical_requests=180)
        m.assert_owned_limit(client)
        self.assertEqual(e2.INPUT_LIMIT, 34000)
        e2.INPUT_LIMIT = 20000
        with self.assertRaisesRegex(RuntimeError, "actual pinned owned CLI"):
            m.assert_owned_limit(client)

    def test_exact_config_diff_rejects_schema_or_text_change(self):
        before = {"prompts": {"input_limit": 20000, "text": "keep20000", "schema": {"maxLength": 20000}}}
        after = deepcopy(before);after["prompts"]["input_limit"] = 34000
        self.assertEqual(m.differences(before, after), [("prompts", "input_limit")])
        after["prompts"]["text"] = "wrong34000"
        self.assertNotEqual(m.differences(before, after), [("prompts", "input_limit")])
        after = deepcopy(before);after["prompts"]["schema"]["maxLength"] = 34000
        self.assertNotEqual(m.differences(before, after), [("prompts", "input_limit")])

    def test_real_manifest_inputs_readonly_validation_has_no_driver_requirement(self):
        manifest, cfg, baseline, states = m.verify_manifest(HERE / "manifest_inputs.json", require_driver=False)
        self.assertEqual(cfg["prompts"]["input_limit"], 34000)
        self.assertEqual([states[seed]["round_completed"] for seed in (0, 1, 2)], [6, 6, 6])
        self.assertEqual(len(manifest["preserved_trees"]), 19)
        self.assertEqual(len(manifest["preserved_files"]), 140)


if __name__ == "__main__":
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    unchanged = hashlib.sha256(SOURCE.read_bytes()).hexdigest() == LOADED_SHA
    audit = {"status": "PASS" if result.wasSuccessful() and unchanged else "FAIL",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "elapsed_seconds": time.monotonic() - started, "source_sha256": LOADED_SHA,
        "source_unchanged_during_tests": unchanged, "real_LLM_calls": 0,
        "real_generated_feature_executions": 0, "real_model_fit_predict_calls": 0, "TEST_target_fields_read": 0}
    path = (HERE / "software/software_audit.json").resolve()
    assert "onedrive" not in str(path).casefold() and path.is_relative_to(HERE)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({**audit, "audit_path": str(path)}, ensure_ascii=False))
    raise SystemExit(0 if audit["status"] == "PASS" else 1)
