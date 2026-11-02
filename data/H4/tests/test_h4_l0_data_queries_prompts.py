"""Stage4.2 CPU mocks: no real labels, feature execution, fit or model calls."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import random
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
import h4_l0_data as d
import h4_l0_prompts as p
import h4_l0_queries as q
import h4_transport as transport


def bundle():
    pool = [f"P{i:05d}" for i in range(2100)]
    test = [f"T{i:05d}" for i in range(1000)]
    by_id = {rid: {"formula": "Fe1O1", "composition": {"Fe": .5, "O": .5},
        "query_elements": ["Fe", "O"], "n_elements": 2, "n_sites": 2,
        "template_spacegroup": (i % 230) + 1} for i, rid in enumerate(pool + test)}
    groups = {rid: "g" + str(i % 50) for i, rid in enumerate(pool + test)}
    mapping = {"g" + str(i): "val" if i < 10 else "fit" for i in range(50)}
    return {"ids": pool + test, "pool_ids": pool, "test_ids": test, "by_id": by_id,
        "groups": groups, "perseed": {seed: {"S0_ids": pool[:500], "fit_group_mapping": mapping}
            for seed in (0, 1, 2)},
        "source_hashes": {str(ROOT / "work/h-data/H2/synthetic_records_private.json.gz"): "frozen"},
        "evaluation_truth_loaded": False}


class GuardedRow(dict):
    def __init__(self, *args, forbid=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbid = forbid
        self.reads = 0

    def __getitem__(self, key):
        if key == "formation_energy_eV_per_atom":
            if self.forbid:
                raise AssertionError("Unselected target field accessed")
            self.reads += 1
        return super().__getitem__(key)


@contextmanager
def private_mock(rows, marker=None):
    with patch.object(d, "checked", side_effect=lambda value, **kw: Path(value)), \
            patch.object(d, "sha", return_value="frozen"), \
            patch.object(d.gzip, "open", side_effect=lambda *a, **kw: io.StringIO("unused")), \
            patch.object(d.json, "load", return_value=[{"row": row} for row in rows]) as loader, \
            patch.object(d, "_read", return_value=marker or {}):
        yield loader


class OracleTests(unittest.TestCase):
    def test_initial_reveal_accesses_only_S0_not_old_O_or_TEST(self):
        b = bundle()
        rows = [GuardedRow(id=rid, split="POOL", formation_energy_eV_per_atom=-1., forbid=i >= 500)
            for i, rid in enumerate(b["pool_ids"])]
        rows += [GuardedRow(id=rid, split="TEST", formation_energy_eV_per_atom=9., forbid=True)
            for rid in b["test_ids"]]
        with patch.object(d, "_REVEALED", {}), private_mock(rows), \
                patch.object(d.old, "reveal_training", side_effect=AssertionError("Old O oracle forbidden")):
            labels = d.initial_labels(0, bundle=b)
        self.assertEqual(500, len(labels))
        self.assertEqual(500, sum(row.reads for row in rows))

    def test_incremental_reveal_only_reads_selected75(self):
        b = bundle()
        observed = {0: {rid: -1. for rid in b["pool_ids"][:500]}}
        chosen = b["pool_ids"][500:575]
        rows = [GuardedRow(id=rid, split="POOL", formation_energy_eV_per_atom=-2., forbid=rid not in chosen)
            for rid in b["pool_ids"]]
        with patch.object(d, "_REVEALED", observed), private_mock(rows):
            labels = d.reveal(chosen, 0, bundle=b)
        self.assertEqual(set(chosen), set(labels))
        self.assertEqual(75, sum(row.reads for row in rows))
        self.assertEqual(575, len(observed[0]))

    def test_reveal_denies_TEST_and_over2000_before_loading(self):
        b = bundle()
        with patch.object(d, "_REVEALED", {}), private_mock([]) as loader:
            with self.assertRaises(ValueError):
                d.reveal([b["test_ids"][0]], 0, bundle=b)
            with self.assertRaises(ValueError):
                d.reveal(b["pool_ids"][:2001], 0, bundle=b)
            with self.assertRaises(ValueError):
                d.reveal(b["pool_ids"][500:575], 0, bundle=b)
        loader.assert_not_called()

    def test_L0_truth_discloses_O_pass_and_reads_once_only(self):
        b = bundle()
        rows = [GuardedRow(id=rid, split="TEST", formation_energy_eV_per_atom=-1.) for rid in b["test_ids"]]
        rows.append(GuardedRow(id=b["pool_ids"][0], split="POOL", formation_energy_eV_per_atom=9., forbid=True))
        marker = {"stage": "4.2", "condition": "L0", "scoring_pass": 1, "current_L0_scoring_pass": 1,
            "prior_H4_TEST_scoring_passes": 1, "TEST_targets_previously_read_for_this_condition": False,
            "prior_stage4_1_marker_sha256": "frozen"}
        with patch.object(d, "_TRUTH_READ", False), private_mock(rows, marker):
            truth, provenance = d.score_truth(b["test_ids"], evaluation_marker=d.PHASE / "evaluation_started.json", bundle=b)
            with self.assertRaises(RuntimeError):
                d.score_truth(b["test_ids"], evaluation_marker=d.PHASE / "evaluation_started.json", bundle=b)
        self.assertEqual(1000, len(truth))
        self.assertEqual(1000, sum(row.reads for row in rows))
        self.assertEqual(1, provenance["prior_H4_TEST_scoring_passes"])

    def test_L0_truth_rejects_bool_pass_count_and_prior_hash_drift(self):
        b = bundle()
        marker = {"stage": "4.2", "condition": "L0", "scoring_pass": True, "current_L0_scoring_pass": 1,
            "prior_H4_TEST_scoring_passes": 1, "TEST_targets_previously_read_for_this_condition": False,
            "prior_stage4_1_marker_sha256": "frozen"}
        with private_mock([], marker) as loader, self.assertRaises(ValueError):
            d.score_truth(b["test_ids"], evaluation_marker=d.PHASE / "evaluation_started.json", bundle=b)
        loader.assert_not_called()


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.b = bundle()

    def test_spacegroup_closed_range_matching(self):
        row = self.b["by_id"][self.b["pool_ids"][0]]
        self.assertTrue(q.matches({"template_spacegroup": [1, 1]}, row))
        self.assertFalse(q.matches({"template_spacegroup": [2, 230]}, row))

    def test_invalid_spacegroup_ranges_and_bool_are_rejected(self):
        for interval in ([0, 20], [1, 231], [20, 1], [True, 4], [1]):
            with self.assertRaises(ValueError):
                q.validate({"queries": [{"n": 1, "template_spacegroup": interval}], "rationale": ""}, bundle=self.b, capacity=75)

    def test_query_quota75_plus_carry_and_three_query_limit(self):
        value = {"queries": [{"n": 76}], "rationale": ""}
        with self.assertRaises(ValueError):
            q.validate(value, bundle=self.b, capacity=75)
        self.assertEqual(value, q.validate(value, bundle=self.b, capacity=76))
        with self.assertRaises(ValueError):
            q.validate({"queries": [{"n": 1}] * 4, "rationale": ""}, bundle=self.b)

    def test_sampling_reuses_exact_H4A_rng_and_never_duplicates(self):
        labeled = self.b["pool_ids"][:500]
        result = q.select_only_no_reveal({"queries": [{"n": 50}, {"n": 25}], "rationale": ""},
            arm="L0", seed=0, round_index=20, labeled_ids=labeled, bundle=self.b)
        token = "H4A-query-selection-v1|query|L0|0|20|0"
        expected = random.Random(int(hashlib.sha256(token.encode()).hexdigest(), 16)).sample(sorted(set(self.b["pool_ids"]) - set(labeled)), 50)
        self.assertEqual(expected, result["selected_ids"][:50])
        self.assertEqual(75, len(set(result["selected_ids"])))
        self.assertEqual(0, result["carry_out"])

    def test_shortfall_is_carried_without_filling(self):
        result = q.select_only_no_reveal({"queries": [{"n": 75, "include_all": ["Fe"], "exclude": ["Fe"]}], "rationale": ""},
            arm="L0", seed=1, round_index=1, labeled_ids=self.b["pool_ids"][:500], bundle=self.b)
        self.assertEqual([], result["selected_ids"])
        self.assertEqual(75, result["carry_out"])
        self.assertIs(False, result["automatic_topup"])

    def test_final_fill_only_after20_uses_round21_rng(self):
        labeled = self.b["pool_ids"][:1974]
        result = q.final_fill_only_no_reveal(arm="L0", seed=1, labeled_ids=labeled, bundle=self.b, total_new_revealed=1474)
        self.assertEqual(26, len(result["selected_ids"]))
        self.assertEqual(20, result["after_round"])
        self.assertFalse(set(labeled) & set(result["selected_ids"]))
        self.assertEqual(q._rng("L0", 1, 21, 0, "final_uniform_fill").sample(self.b["pool_ids"][1974:], 26), result["selected_ids"])

    def test_wire_null_is_omission_only_for_optional_conditions(self):
        schema = q.schema(self.b, 75)
        wire = q.strict_schema(schema)
        props = wire["properties"]["queries"]["items"]["properties"]
        self.assertIn({"type": "null"}, props["template_spacegroup"]["anyOf"])
        value = {"queries": [{"n": 1, "template_spacegroup": None, "include_all": None}], "rationale": ""}
        normalized = q.normalize_optional_nulls(value, schema)
        self.assertEqual({"queries": [{"n": 1}], "rationale": ""}, normalized)
        q.validate(normalized, bundle=self.b, capacity=75)
        with self.assertRaises(ValueError):
            q.validate(value, bundle=self.b, capacity=75)

    def test_catalogue_updates_remaining_SG_counts_without_targets(self):
        first = q.catalogue([], bundle=self.b)
        second = q.catalogue(self.b["pool_ids"][:500], bundle=self.b)
        self.assertEqual(2100, sum(first["template_spacegroup_distribution"].values()))
        self.assertEqual(1600, sum(second["template_spacegroup_distribution"].values()))
        self.assertIs(False, second["label_values_accessed"])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.b = bundle()
        self.labels = {rid: -1. for rid in self.b["pool_ids"][:500]}
        val = [rid for rid in self.labels if self.b["perseed"][0]["fit_group_mapping"][self.b["groups"][rid]] == "val"]
        self.worst = [{"id": rid, "label_eV_per_atom": -1., "prediction_eV_per_atom": 0., "absolute_error_eV_per_atom": 1.}
            for rid in val[:20]]

    def request(self, features):
        with patch.object(p.feature_contract, "_global_instruction_bytes", return_value=500):
            return p.build_request(0, 1, features, 1., self.worst, self.labels, bundle=self.b)

    def test_schema_hash_stays_identical_for_every_round_capacity(self):
        first = p.canonical_schema_bytes(p.response_schema(self.b, 75))
        for capacity in (75, 100, 225, 1500):
            self.assertEqual(first, p.canonical_schema_bytes(p.response_schema(self.b, capacity)))
        self.assertEqual(1500, p.response_schema(self.b, 75)["properties"]["queries"]["items"]["properties"]["n"]["maximum"])

    def test_local_refs_resolve_with_same_element_semantics_and_unchanged_features(self):
        from copy import deepcopy
        schema = p.response_schema(self.b, 75)
        inline = deepcopy(schema)
        common = inline.pop("$defs")["element_filter"]
        props = inline["properties"]["queries"]["items"]["properties"]
        for key in ("include_all", "include_any", "exclude"):
            props[key] = deepcopy(common)
        self.assertEqual(p.feature_contract.FEATURE_SCHEMA["properties"]["features"], schema["properties"]["features"])
        for elements in (["Fe"], ["O"], [], ["Xx"], ["Fe", "Fe"]):
            value = {"features": [{"name": "f", "description": "d", "source": "def feature(s): return 0"}],
                "queries": [{"n": 1, "include_all": elements}], "rationale": ""}
            accepted = []
            for candidate_schema in (schema, inline):
                try:
                    transport._schema_validate(value, candidate_schema, root=candidate_schema)
                    accepted.append(True)
                except ValueError:
                    accepted.append(False)
            self.assertEqual(accepted[0], accepted[1])
            self.assertEqual(elements not in (["Xx"], ["Fe", "Fe"]), accepted[0])
        wire = p.wire_schema(schema)
        self.assertEqual(common["items"]["enum"], wire["$defs"]["element_filter"]["items"]["enum"])
        self.assertEqual(p.feature_contract.FEATURE_SCHEMA["properties"]["features"], wire["properties"]["features"])

    def test_admissible_query_failure_still_has_only_one_format_repair(self):
        request = self.request([])
        self.assertIs(True, request["admissible_format_failure"])
        self.assertEqual(1, request["max_format_repairs"])

    def test_initial_magpie_only_has_empty_current_features(self):
        request = self.request([])
        self.assertEqual(0, request["current_feature_count"])
        self.assertTrue(request["format_repair_input_budget"]["within_budget"])
        self.assertFalse(any(rid in request["prompt"] for rid in self.b["ids"]))

    def test_current_source_string_is_preserved_completely(self):
        features = [{"name": "f", "description": "original", "source": "def feature(s):\n    return float(s['n_sites'])\n"}]
        request = self.request(features)
        self.assertIn(features[0]["source"], request["prompt"])
        self.assertIn(p._json(features[0]["description"]), request["prompt"])
        self.assertEqual(len(features[0]["source"].encode()), request["current_complete_source_bytes"])

    def test_large_source_stops_unchanged_instead_of_truncating(self):
        features = [{"name": "f", "description": "original", "source": "#" * 19500}]
        before = features[0]["source"]
        with self.assertRaises(p.PromptBudgetExceeded):
            self.request(features)
        self.assertEqual(before, features[0]["source"])

    def test_feedback_cannot_use_an_unrevealed_or_fit_row(self):
        self.worst[0]["id"] = self.b["pool_ids"][500]
        with self.assertRaises(ValueError):
            self.request([])

    def test_feedback_label_must_match_current_oracle(self):
        self.worst[0]["label_eV_per_atom"] = -2.
        with self.assertRaises(ValueError):
            self.request([])

    def test_response_preserves_old_feature_shape_and_checks_query_quota(self):
        spec = {"features": [{"name": "f", "description": "d", "source": "invalid AST is a scientific failure"}],
            "queries": [{"n": 75, "template_spacegroup": [1, 230]}], "rationale": ""}
        self.assertTrue(p.validate_response(spec, bundle=self.b, capacity=75))
        spec["queries"][0]["n"] = 76
        with self.assertRaises(ValueError):
            p.validate_response(spec, bundle=self.b, capacity=75)


if __name__ == "__main__":
    unittest.main()
