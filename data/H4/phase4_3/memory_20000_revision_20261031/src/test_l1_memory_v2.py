"""Synthetic CPU tests only; no real histories, CLI, fit, feature or TEST access."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parent
PHASE = SRC.parents[1]
ROOT = PHASE.parents[3]
sys.path[:0] = [str(SRC), str(PHASE / "src"), str(PHASE.parent / "src")]
import l1_memory_v2 as m
import l1_prompts_v2 as p
import l1_prompts as prior
import l1_revision_measurement as measurement
import test_l1_memory_prompts as fixtures


def history(count):
    return [fixtures.round_record(rnd, [fixtures.feature(rnd)], [.001 * rnd], notes="note" + str(rnd))
        for rnd in range(1, count + 1)]


class MemoryRevisionTests(unittest.TestCase):
    def test_metadata_and_archive_never_retain_name_description_code(self):
        h = history(5)
        result = m.build_memory(h)
        for key in ("payload", "archive"):
            text = m.counter.canonical(result[key])
            self.assertNotIn("def feature", text)
            self.assertNotIn("descriptor", text)
            self.assertNotIn('"name"', text)
            self.assertNotIn('"description"', text)
            self.assertNotIn('"source"', text)
        self.assertEqual([3, 4, 5], [r["round"] for r in result["payload"]["rounds"]])
        self.assertEqual(5, len(result["payload"]["features"]))
        self.assertFalse(result["trimmed"])
        self.assertEqual(result["token_count"], result["component_breakdown"]["sum_token_increments"])

    def test_external20000_policy_does_not_mutate_original_counter_record6000(self):
        original_measure = m.counter.measure
        def force_7000(value):
            result = original_measure(value)
            if isinstance(value, dict) and value.get("features"): result["token_count"] = 7000
            return result
        with patch.object(m.counter, "measure", side_effect=force_7000):
            result = m.build_memory(history(1))
        self.assertFalse(result["trimmed"])
        self.assertFalse(result["overflow"])
        self.assertEqual(20000, result["limit_tokens"])
        self.assertEqual(6000, m.counter.MEMORY_LIMIT)
        cfg = m.configuration()
        self.assertEqual("seed", cfg["stop_scope"])
        self.assertEqual(20000, cfg["external_policy_limit_tokens"])
        self.assertEqual("TRAIN_validation_only", cfg["scope"])

    def test_unique_hash_latest_delta_and_recent_exact_instance_predicate(self):
        f = fixtures.feature
        h = [fixtures.round_record(1, [f(i) for i in range(20)], [100. if i == 0 else float(i) for i in range(20)]),
            fixtures.round_record(2, [f(i) for i in range(20, 40)], [float(i - 19) for i in range(20, 40)]),
            fixtures.round_record(3, [f(0)], [None]), fixtures.round_record(4, [f(40)], [1.]),
            fixtures.round_record(5, [f(41)], [1.])]
        original_measure = m.counter.measure
        full_n = len(m.metadata_history(h)["features"])
        def force_trim(value):
            result = original_measure(value)
            if isinstance(value, dict) and len(value.get("features", {})) == full_n: result["token_count"] = 20001
            return result
        with patch.object(m.counter, "measure", side_effect=force_trim): result = m.build_memory(h)
        a_sha = hashlib.sha256(f(0)["source"].encode()).hexdigest()
        self.assertTrue(result["trimmed"])
        self.assertEqual(30, len(result["audit"]["top_source_hashes"]))
        self.assertNotIn(a_sha, result["audit"]["top_source_hashes"])
        self.assertNotIn("1:0", result["payload"]["features"])
        self.assertIn("3:0", result["payload"]["features"])
        self.assertEqual([3, 4, 5], [r["round"] for r in result["payload"]["rounds"]])

    def test_over20000_after_one_trim_reports_or_stops_only_seed(self):
        real = m.counter.measure
        def force_over(value):
            result = real(value)
            if isinstance(value, dict) and value.get("features"): result["token_count"] = 20001
            return result
        with patch.object(m.counter, "measure", side_effect=force_over):
            result = m.build_memory(history(1), measurement=True)
            self.assertTrue(result["overflow"])
            with self.assertRaises(m.MemoryBudgetExceeded) as raised: m.build_memory(history(1))
        self.assertEqual("seed", raised.exception.stop_scope)
        self.assertEqual("seed", raised.exception.result["stop_scope"])
        self.assertEqual(result["text"], raised.exception.result["text"])

    def test_rejects_ids_test_fields_and_overlong_notes(self):
        for kind in ("id", "notes", "feature_field"):
            h = history(1)
            if kind == "id": h[0]["notes"] = "H2-T-000123"
            elif kind == "notes": h[0]["notes"] = "字" * 301
            else:
                archive = m.metadata_history(h)
                archive["features"]["1:0"]["TEST_label"] = -1.
                with self.assertRaises(ValueError): m.build_memory(archive)
                continue
            with self.assertRaises(ValueError): m.build_memory(h)


class PromptRevisionTests(unittest.TestCase):
    def setUp(self):
        self.b = fixtures.bundle()
        self.labels = {rid: -1.5 for rid in self.b["pool_ids"][:800]}
        val = [rid for rid in self.labels if self.b["perseed"][0]["fit_group_mapping"][self.b["groups"][rid]] == "val"][:20]
        self.worst = [{"id": rid, "label_eV_per_atom": -1.5, "prediction_eV_per_atom": -1., "absolute_error_eV_per_atom": .5} for rid in val]
        self.patch = patch.object(p.feature_contract, "_global_instruction_bytes", return_value=1169)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def request(self, *, h=None, features=None, rnd=5, input_limit=189000, measurement_only=False):
        return p.build_request(0, rnd, features or [], .3, self.worst, self.labels, history(4) if h is None else h,
            input_limit=input_limit, bundle=self.b, measurement=measurement_only)

    def test_only_memory_wording_changes_prompt_scientific_skeleton(self):
        h = history(4)
        f = fixtures.feature(0, source="def feature(s):\n    # exact original full active 中文 code\n    return 1.0\n")
        old = prior.build_request(0, 5, [f], .3, self.worst, self.labels, h, input_limit=189000, bundle=self.b)
        new = self.request(h=h, features=[f])
        self.assertEqual(old["prompt"].split(prior.MEMORY_EXPLANATION, 1)[0], new["prompt"].split(p.MEMORY_EXPLANATION, 1)[0])
        self.assertEqual(old["prompt"].rsplit("Return only final JSON", 1)[1], new["prompt"].rsplit("Return only final JSON", 1)[1])
        self.assertIn(f["source"], new["prompt"])
        self.assertEqual(prior.PREFIX, p.PREFIX)
        self.assertEqual(prior.DICTIONARY_CONTRACT, p.DICTIONARY_CONTRACT)
        self.assertEqual(prior.INSTRUCTIONS, p.INSTRUCTIONS)
        self.assertEqual(prior.NUMERIC_FORMAT, p.NUMERIC_FORMAT)

    def test_schema_wire_and_repair_unchanged_new_namespace_unchanged_arm(self):
        for capacity in (75, 1500):
            old, new = prior.response_schema(self.b, capacity), p.response_schema(self.b, capacity)
            self.assertEqual(old, new)
            self.assertEqual(prior.canonical_schema_bytes(prior.wire_schema(old)), p.canonical_schema_bytes(p.wire_schema(new)))
        self.assertEqual(prior.FORMAT_REPAIR, p.FORMAT_REPAIR)
        req = self.request()
        self.assertEqual("L1R2_seed0_round05", req["request_id"])
        self.assertEqual("L1", req["arm"])
        self.assertEqual(4, req["memory_history_rounds"])
        self.assertEqual(3, req["memory_detailed_round_records"])
        self.assertEqual([2, 3, 4], [x["round"] for x in req["memory_notes_chars_by_round"]])

    def test_actual_whole_guard_counts_full_source_wire_context_and_repair(self):
        req = self.request()
        a, z = req["input_budget"], req["format_repair_input_budget"]
        self.assertEqual(len(req["prompt"].encode()), a["canonical_prompt_bytes"])
        self.assertEqual(len(p.queries.WIRE_NOTE.encode()), a["wire_prompt_bytes"] - a["canonical_prompt_bytes"])
        self.assertEqual(155, z["wire_prompt_bytes"] - a["wire_prompt_bytes"])
        self.assertEqual(a["wire_prompt_bytes"] + a["known_fixed_context_bytes"] + 1500, a["input_token_upper_bound"])
        with self.assertRaises(p.PromptBudgetExceeded): self.request(input_limit=a["input_token_upper_bound"])
        oversized = [{"name": "x" + str(i), "description": "d", "source": "def feature(s):\n    return " + str(i) + ".0\n#" + "x" * 19500} for i in range(20)]
        with self.assertRaises(p.PromptBudgetExceeded): self.request(features=oversized)

    def test_total_history_audit_and_archive_tamper_rejected(self):
        mem = m.build_memory(history(4))
        mem["audit"]["history_rounds"] = 3
        with self.assertRaises(ValueError): self.request(h=mem)
        with self.assertRaises(ValueError): self.request(h=history(3))

    def test_query_fallback_preserves_complete_features_and_notes(self):
        value = {"features": [fixtures.feature(0)], "queries": [{"n": 76}], "rationale": "", "notes": "保留"}
        result = {"status": "failed", "last_status": "format_invalid", "admissible_format_failure": True,
            "raw_response_text": json.dumps(value, ensure_ascii=False)}
        resolved, audit = p.resolve_response(result, bundle=self.b, capacity=75)
        self.assertEqual(value["features"], resolved["features"])
        self.assertEqual(value["notes"], resolved["notes"])
        self.assertEqual([], resolved["queries"])
        self.assertTrue(audit["query_fallback"])

    def test_measurement_gate_rejects_unamended_policy_before_any_dataset_access(self):
        cfg = {"memory": {"limit_tokens": 6000}, "prompts": {"input_limit": 189000}}
        with patch.object(measurement, "_checked", side_effect=lambda path: Path(path).resolve()), \
                patch.object(measurement, "_sha", return_value="same"), \
                patch.object(measurement, "_read", side_effect=lambda path: {"sha256": "same"} if Path(path).name == "frozen_config.json" else cfg), \
                patch.object(measurement.label_free_data, "load_bundle", side_effect=AssertionError("No dataset access before gate")) as loader:
            with self.assertRaises(ValueError): measurement.measure_all_l0()
        loader.assert_not_called()


if __name__ == "__main__": unittest.main()
