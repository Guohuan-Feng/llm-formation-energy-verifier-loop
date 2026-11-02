"""Local CPU-only memory/prompt tests; no scientific model, feature, fit or truth."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1_counter as c
import l1_memory as m
import l1_prompts as p


def feature(index=0, *, source=None):
    return {"name": "f" + str(index), "description": "descriptor" + str(index),
        "source": source or "def feature(s):\n    return " + str(index + 1) + ".0\n"}


def round_record(rnd, features=None, deltas=None, notes="", *, with_query=True):
    fs = features or [feature(rnd)]
    ds = deltas or [float(i) for i in range(len(fs))]
    qs = [{"query": {"n": 75, "include_all": ["O"]}, "requested_count": 75,
        "matched_count": 100, "revealed_count": 75, "carry_after": 0}] if with_query else []
    return {"round_index": rnd, "features": deepcopy(fs),
        "contributions": [{"feature_index": i, "delta_without_minus_full": ds[i]} for i in range(len(fs))],
        "failures": [], "queries": qs, "notes": notes,
        "query_summary": {"capacity": 75, "carry_in": 0, "carry_out": 0 if qs else 75,
            "requested_count": 75 if qs else 0, "matched_count": 100 if qs else 0,
            "revealed_count": 75 if qs else 0}}


def bundle():
    pool = [f"P{i:05d}" for i in range(2100)]
    test = [f"T{i:05d}" for i in range(1000)]
    ids = pool + test
    by_id = {rid: {"formula": "Fe1O1", "composition": {"Fe": .5, "O": .5},
        "query_elements": ["Fe", "O"], "n_elements": 2, "n_sites": 2,
        "template_spacegroup": i % 230 + 1} for i, rid in enumerate(ids)}
    groups = {rid: "g" + str(i % 50) for i, rid in enumerate(ids)}
    mapping = {"g" + str(i): "val" if i < 10 else "fit" for i in range(50)}
    return {"ids": ids, "pool_ids": pool, "test_ids": test, "by_id": by_id, "groups": groups,
        "perseed": {s: {"S0_ids": pool[:500], "fit_group_mapping": mapping} for s in (0, 1, 2)}}


class MemoryTests(unittest.TestCase):
    def test_counter_canonical_exact_no_newline_and_special_plaintext(self):
        value = {"b": "<|endoftext|>中文", "a": 2}
        measure = c.measure(value)
        self.assertEqual('{"a":2,"b":"<|endoftext|>中文"}', measure["text"])
        self.assertEqual(c.count_text(measure["text"]), measure["token_count"])
        self.assertEqual(len(measure["text"].encode("utf-8")), measure["UTF8_bytes"])

    def test_empty_history_and_source_dedup_preserve_complete_archive(self):
        self.assertEqual([], m.build_memory([])["payload"]["rounds"])
        f = feature(0, source="def feature(s):\n    # 原文完整\n    return 1.0\n")
        h = [round_record(1, [f], [1.0]), round_record(2, [f], [-.1])]
        result = m.build_memory(h)
        self.assertEqual(1, len(result["payload"]["definitions"]))
        self.assertEqual(h, result["archive"]["rounds"])
        self.assertEqual(f["source"], next(iter(result["payload"]["definitions"].values()))["source"])
        self.assertFalse(result["trimmed"])

    def test_prescribed_trim_top30_latest_ties_recent3_queries_and_notes(self):
        h = []
        for rnd in range(1, 8):
            fs = [feature((rnd - 1) * 6 + i) for i in range(6)]
            h.append(round_record(rnd, fs, [float(i % 2) for i in range(6)], notes=f"note{rnd}"))
        original_measure = c.measure
        def measure(payload):
            result = original_measure(payload)
            result["token_count"] = 7000 if len(payload["definitions"]) == 42 else 5000
            return result
        with patch.object(c, "measure", side_effect=measure):
            result = m.build_memory(h)
        self.assertTrue(result["trimmed"])
        self.assertFalse(result["overflow"])
        self.assertEqual([5, 6, 7], result["audit"]["recent_rounds_preserved"])
        for before, after in zip(h, result["payload"]["rounds"]):
            self.assertEqual(before["notes"], after["notes"])
            self.assertEqual(before["queries"], after["queries"])
            if before["round_index"] >= 5: self.assertEqual(6, len(after["features"]))
        ranking = result["audit"]["top_feature_hashes"]
        expected_top_first = m.source_hash(h[0]["features"][1]["source"])
        self.assertEqual(expected_top_first, ranking[0])
        self.assertEqual(30, len(ranking))
        self.assertEqual(42, len({m.source_hash(f["source"]) for r in result["archive"]["rounds"] for f in r["features"]}))

    def test_latest_contribution_replaces_old_and_none_is_last(self):
        f = feature(0)
        h = [round_record(1, [f, feature(1)], [100., 1.]), round_record(2, [f, feature(2)], [None, 2.])]
        view, ranking = m._view(m.normalize_history(h))
        self.assertEqual(m.source_hash(feature(2)["source"]), ranking[0])
        self.assertEqual(m.source_hash(f["source"]), ranking[-1])

    def test_true_overflow_measurement_reports_and_runtime_stops_without_source_cut(self):
        rng = random.Random(20261031)
        fs = [feature(i, source="def feature(s):\n    # " + "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(1200))
            + "\n    return " + str(i) + ".0\n") for i in range(20)]
        h = [round_record(1, fs)]
        result = m.build_memory(h, measurement=True)
        self.assertGreater(result["token_count"], 6000)
        self.assertTrue(result["overflow"])
        self.assertTrue(result["trimmed"])
        with self.assertRaises(m.MemoryBudgetExceeded) as raised:
            m.build_memory(h)
        self.assertEqual(result["text"], raised.exception.result["text"])
        self.assertEqual({f["source"] for f in fs}, {v["source"] for v in result["payload"]["definitions"].values()})

    def test_notes_unicode_cap_and_no_truncation(self):
        h = [round_record(1, notes="😀" * 300)]
        self.assertEqual("😀" * 300, m.build_memory(h)["payload"]["rounds"][0]["notes"])
        h[0]["notes"] += "字"
        with self.assertRaises(ValueError): m.build_memory(h)

    def test_private_ids_and_test_diagnostics_rejected(self):
        for mutation in ("notes", "extra", "phase"):
            h = [round_record(1)]
            if mutation == "notes": h[0]["notes"] = "H2-T-000123"
            elif mutation == "extra": h[0]["TEST_mae"] = .1
            else: h[0]["failures"] = [{"feature_index": 0, "phase": "TEST", "failed_count": 1, "total_count": 1000, "reasons": {"timeout": 1}}]
            with self.assertRaises(ValueError): m.build_memory(h)
        h = [round_record(1, notes="custom-private-identity")]
        with self.assertRaises(ValueError): m.build_memory(h, private_ids=["custom-private-identity"])

    def test_failure_reason_aggregation_and_empty_query_carry_are_retained(self):
        h = [round_record(1, with_query=False)]
        h[0]["failures"] = [{"feature_index": 0, "phase": "inner_fit", "failed_count": 4, "total_count": 100,
            "reasons": {"timeout": 3, "runtime_exception": 1}}]
        result = m.build_memory(h)
        self.assertEqual(75, result["payload"]["rounds"][0]["query_summary"]["carry_out"])
        self.assertEqual({"timeout": 3, "runtime_exception": 1}, result["payload"]["rounds"][0]["failures"][0]["reasons"])
        h[0]["failures"][0]["reasons"]["timeout"] = 2
        with self.assertRaises(ValueError): m.build_memory(h)

    def test_runtime_requires_summary_and_rejects_nonfinite_contribution(self):
        h = [round_record(1)]
        del h[0]["query_summary"]
        self.assertEqual(1, m.build_memory(h, measurement=True)["audit"]["history_rounds"])
        with self.assertRaises(ValueError): m.build_memory(h)
        h = [round_record(1)]
        h[0]["contributions"][0]["delta_without_minus_full"] = float("nan")
        with self.assertRaises(ValueError): m.build_memory(h)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.b = bundle()
        self.labels = {rid: -1.5 for rid in self.b["pool_ids"][:500]}
        val = [rid for rid in self.labels if self.b["perseed"][0]["fit_group_mapping"][self.b["groups"][rid]] == "val"][:20]
        self.worst = [{"id": rid, "label_eV_per_atom": -1.5, "prediction_eV_per_atom": -1., "absolute_error_eV_per_atom": .5} for rid in val]
        self.context = patch.object(p.feature_contract, "_global_instruction_bytes", return_value=1169)
        self.context.start(); self.addCleanup(self.context.stop)

    def request(self, *, memory=None, round_index=1, features=None, input_limit=60000):
        return p.build_request(0, round_index, features or [], .3, self.worst, self.labels,
            [] if memory is None else memory, input_limit=input_limit, bundle=self.b)

    def test_schema_preserves_feature_and_query_contract_including_null_wire(self):
        a, b = p.response_schema(self.b, 75), p.response_schema(self.b, 1500)
        self.assertEqual(a, b)
        self.assertEqual(p.feature_contract.FEATURE_SCHEMA["properties"]["features"], a["properties"]["features"])
        self.assertEqual(20000, a["properties"]["features"]["items"]["properties"]["source"]["maxLength"])
        self.assertEqual(300, a["properties"]["notes"]["maxLength"])
        self.assertEqual(["features", "queries", "rationale", "notes"], a["required"])
        self.assertEqual(["array", "null"], [node["type"] for node in
            p.wire_schema(a)["properties"]["queries"]["items"]["properties"]["n_elements"]["anyOf"]])

    def test_full_original_source_and_all_l0_feedback_are_present(self):
        f = feature(0, source="def feature(s):\n    # preserved exact 中文 comment\n    return 1.0\n")
        req = self.request(features=[f], memory=[round_record(1, [f], [.1])], round_index=2)
        self.assertIn(f["source"], req["prompt"])
        self.assertIn(p.l0._worst_text(p.l0._worst_rows(self.worst, self.labels, 0, self.b)), req["prompt"])
        self.assertIn(p.MEMORY_LABEL, req["prompt"])
        self.assertEqual("L1_seed0_round02", req["request_id"])
        self.assertEqual(hashlib.sha256(req["prompt"].encode()).hexdigest(), req["canonical_prompt_sha256"])
        self.assertEqual(1, req["memory_history_rounds"])

    def test_ordinary_and_repair_bounds_exactly_include155bytes(self):
        req = self.request()
        self.assertEqual(155, len(p.FORMAT_REPAIR.encode("utf-8")))
        self.assertEqual(155, req["format_repair_input_budget"]["input_token_upper_bound"] - req["input_budget"]["input_token_upper_bound"])
        with self.assertRaises(p.PromptBudgetExceeded): self.request(input_limit=100)
        limit = req["input_budget"]["input_token_upper_bound"]
        with self.assertRaises(p.PromptBudgetExceeded): self.request(input_limit=limit)

    def test_memory_text_count_tamper_and_wrong_round_rejected(self):
        mem = m.build_memory([])
        mem["token_count"] += 1
        with self.assertRaises(ValueError): self.request(memory=mem)
        with self.assertRaises(ValueError): self.request(memory=[round_record(1)])

    def test_untrusted_memory_nested_extra_fields_or_missing_carry_rejected(self):
        mem = m.build_memory([round_record(1)])
        mem["payload"]["rounds"][0]["queries"][0]["TEST_label"] = -1.0
        with self.assertRaises(ValueError): self.request(memory=mem, round_index=2)
        history = [round_record(1)]
        del history[0]["query_summary"]
        mem = m.build_memory(history, measurement=True)
        with self.assertRaises(ValueError): self.request(memory=mem, round_index=2)

    def test_query_only_final_invalid_fallback_preserves_features_and_notes(self):
        value = {"features": [feature(0)], "queries": [{"n": 76}], "rationale": "valid", "notes": "保留笔记"}
        result = {"status": "failed", "last_status": "format_invalid", "admissible_format_failure": True,
            "raw_response_text": json.dumps(value, ensure_ascii=False)}
        resolved, audit = p.resolve_response(result, bundle=self.b, capacity=75)
        self.assertEqual(value["features"], resolved["features"])
        self.assertEqual(value["notes"], resolved["notes"])
        self.assertEqual([], resolved["queries"])
        self.assertTrue(audit["query_fallback"])
        value["notes"] = "x" * 301
        result["raw_response_text"] = json.dumps(value)
        with self.assertRaises(ValueError): p.resolve_response(result, bundle=self.b, capacity=75)

    def test_invalid_feature_or_capacity_failure_cannot_become_empty_set(self):
        for status in ("cli_error", "format_invalid"):
            result = {"status": "failed", "last_status": status, "admissible_format_failure": status == "format_invalid",
                "raw_response_text": json.dumps({"features": [], "queries": [], "rationale": "", "notes": ""})}
            with self.assertRaises(ValueError): p.resolve_response(result, bundle=self.b, capacity=75)

    def test_success_and_optional_null_normalization(self):
        value = {"features": [feature(0)], "queries": [{"n": 75, "include_all": None}], "rationale": "", "notes": ""}
        resolved, audit = p.resolve_response({"status": "ok", "response": value}, bundle=self.b, capacity=75)
        self.assertEqual([{"n": 75}], resolved["queries"])
        self.assertFalse(audit["query_fallback"])


if __name__ == "__main__": unittest.main()
