"""CPU-only gates, deterministic membership and payload tests; no real truth read."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from contextlib import contextmanager
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
import h4_data as d
import h4_prompts as p


class GuardedRow(dict):
    def __init__(self, *args, forbid_target=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbid_target = forbid_target
        self.target_reads = 0

    def __getitem__(self, key):
        if key == "formation_energy_eV_per_atom":
            if self.forbid_target:
                raise AssertionError("Forbidden held-out/unselected target access")
            self.target_reads += 1
        return super().__getitem__(key)


def fake_bundle():
    pool = [f"P{i:05d}" for i in range(2000)]
    test = [f"T{i:05d}" for i in range(1000)]
    by_id = {rid: {"formula": "Fe1O1", "composition": {"Fe": .5, "O": .5},
        "n_sites": 2, "n_elements": 2, "structure_statistics": {
            "volume_per_atom_A3": 10., "mean_neighbor_distance_6A_A": 3.}}
        for rid in pool + test}
    return {"ids": pool + test, "pool_ids": pool, "test_ids": test, "by_id": by_id,
        "perseed": {seed: {"O_ids": pool, "S0_ids": pool[:500]} for seed in d.SEEDS},
        "source_hashes": {str(d.H2 / "synthetic_records_private.json.gz"): "frozen"},
        "field_spec_sha256": "field-spec", "evaluation_truth_loaded": False}


@contextmanager
def fake_private_records(rows, marker=None):
    with patch.object(d, "checked", side_effect=lambda path, **kwargs: Path(path)), \
            patch.object(d, "sha", return_value="frozen"), \
            patch.object(d.gzip, "open", return_value=io.StringIO("unused")), \
            patch.object(d.json, "load", return_value=rows) as loader, \
            patch.object(d, "_read", return_value=marker or {}):
        yield loader


class DataTests(unittest.TestCase):
    def test_template_split_is_deterministic_and_disjoint(self):
        ids = [f"id{i}" for i in range(40)]
        groups = {rid: f"g{i // 2}" for i, rid in enumerate(ids)}
        fit, val = d._inner_split(ids, groups, 1)
        self.assertEqual((fit, val), d._inner_split(list(reversed(ids)), groups, 1))
        self.assertEqual(set(ids), set(fit + val))
        self.assertEqual(4, len({groups[rid] for rid in val}))
        self.assertFalse({groups[rid] for rid in fit} & {groups[rid] for rid in val})

    def test_reveal_reads_only_authorized_pool2000(self):
        b = fake_bundle()
        allowed = [GuardedRow(id=rid, split="POOL", formation_energy_eV_per_atom=-1.) for rid in b["pool_ids"]]
        forbidden = [GuardedRow(id=rid, split="TEST", formation_energy_eV_per_atom=99., forbid_target=True)
            for rid in b["test_ids"]]
        forbidden.append(GuardedRow(id="outside-pool", split="POOL", formation_energy_eV_per_atom=99., forbid_target=True))
        with patch.object(d, "_TRAIN_LABELS", {}), fake_private_records([{"row": row} for row in allowed + forbidden]):
            labels = d.reveal_training(0, bundle=b)
        self.assertEqual(2000, len(labels))
        self.assertEqual(2000, sum(row.target_reads for row in allowed))
        self.assertEqual(0, sum(row.target_reads for row in forbidden))

    def test_training_whitelist_rejects_even_one_test_label(self):
        b = fake_bundle()
        labels = {rid: -1. for rid in b["pool_ids"]}
        labels.pop(b["pool_ids"][0]); labels[b["test_ids"][0]] = -1.
        with self.assertRaises(ValueError):
            d.validate_training_labels(labels, 0, bundle=b)

    def test_training_whitelist_rejects_bool_and_nonfinite(self):
        b = fake_bundle()
        labels = {rid: -1. for rid in b["pool_ids"]}
        for invalid in (True, float("nan"), float("inf")):
            labels[b["pool_ids"][0]] = invalid
            with self.assertRaises(ValueError):
                d.validate_training_labels(labels, 0, bundle=b)

    def test_truth_requires_all_frozen_ids_in_order(self):
        b = fake_bundle()
        with self.assertRaises(ValueError):
            d.score_truth(list(reversed(b["test_ids"])), evaluation_marker="missing", bundle=b)

    def test_truth_gate_rejects_previously_read_before_private_load(self):
        b = fake_bundle()
        marker = {"stage": "4.1", "scoring_pass": 1, "TEST_targets_previously_read": True}
        with fake_private_records([], marker) as loader, self.assertRaises(ValueError):
            d.score_truth(b["test_ids"], evaluation_marker=d.CACHE / "phase4_1/evaluation_started.json", bundle=b)
        loader.assert_not_called()

    def test_truth_gate_rejects_wrong_stage(self):
        b = fake_bundle()
        with fake_private_records([], {"stage": "4.2", "scoring_pass": 1, "TEST_targets_previously_read": False}), self.assertRaises(ValueError):
            d.score_truth(b["test_ids"], evaluation_marker=d.CACHE / "phase4_1/evaluation_started.json", bundle=b)

    def test_truth_reads_one_pass_no_pool_or_holdout_target(self):
        b = fake_bundle()
        allowed = [GuardedRow(id=rid, split="TEST", formation_energy_eV_per_atom=-2.) for rid in b["test_ids"]]
        forbidden = [GuardedRow(id="P00000", split="POOL", formation_energy_eV_per_atom=99., forbid_target=True),
            GuardedRow(id="R00000", split="HOLDOUT", formation_energy_eV_per_atom=99., forbid_target=True)]
        marker = {"stage": "4.1", "scoring_pass": 1, "TEST_targets_previously_read": False}
        with patch.object(d, "_TRUTH_ALREADY_READ", False), fake_private_records([{"row": row} for row in allowed + forbidden], marker):
            truth, provenance = d.score_truth(b["test_ids"], evaluation_marker=d.CACHE / "phase4_1/evaluation_started.json", bundle=b)
            with self.assertRaises(RuntimeError):
                d.score_truth(b["test_ids"], evaluation_marker=d.CACHE / "phase4_1/evaluation_started.json", bundle=b)
        self.assertEqual(1000, len(truth))
        self.assertEqual(1000, sum(row.target_reads for row in allowed))
        self.assertEqual(0, sum(row.target_reads for row in forbidden))
        self.assertEqual(1, provenance["scoring_pass"])


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.b = fake_bundle()
        self.labels = {rid: -1.25 for rid in self.b["pool_ids"]}

    def test_sampling_is_300_unique_deterministic_training_only(self):
        sample = p.sample_ids(0, 0, bundle=self.b)
        self.assertEqual(300, len(set(sample)))
        self.assertTrue(set(sample) <= set(self.b["pool_ids"]))
        self.assertEqual(sample, p.sample_ids(0, 0, bundle=self.b))
        self.assertNotEqual(sample, p.sample_ids(0, 1, bundle=self.b))

    def test_prompt_has_300_rows_and_no_private_id(self):
        with patch.object(p, "_global_instruction_bytes", return_value=1000):
            request = p.feature_request(0, 0, self.labels, bundle=self.b)
        table = request["prompt"].split("Examples (tab-separated):\n", 1)[1].split("Return only the final JSON", 1)[0]
        self.assertEqual(301, len(table.strip().splitlines()))
        self.assertFalse(any(rid in request["prompt"] for rid in self.b["ids"]))
        self.assertEqual(300, request["example_count"])
        self.assertTrue(request["input_budget"]["within_budget"])

    def test_all_sixty_requests_are_prebudgeted(self):
        with patch.object(d, "reveal_training", return_value=self.labels), patch.object(p, "_global_instruction_bytes", return_value=1000):
            requests = p.build_requests(self.b)
            config = p.configuration(requests, bundle=self.b)
        self.assertEqual(60, len(requests))
        self.assertEqual(60, len({r["request_id"] for r in requests}))
        self.assertTrue(all(r["input_budget"]["within_budget"] for r in requests))
        self.assertEqual(300, config["example_count_per_request"])

    def test_budget_overrun_stops_instead_of_dropping_rows(self):
        with patch.object(p, "_global_instruction_bytes", return_value=20000), self.assertRaises(ValueError):
            p.feature_request(0, 0, self.labels, bundle=self.b)

    def test_validator_allows_one_or_twenty_format_only(self):
        one = {"name": "x", "description": "short", "source": "invalid python is a scientific AST failure"}
        self.assertTrue(p.validator({"features": [one]}))
        twenty = [{"name": str(i), "description": "desc", "source": "source" + str(i)} for i in range(20)]
        self.assertTrue(p.validator({"features": twenty}))

    def test_validator_rejects_missing_extra_or_duplicate_fields(self):
        item = {"name": "x", "description": "desc", "source": "def feature(s): return 1"}
        bad = ({"features": []}, {"features": [item], "query": []},
            {"features": [dict(item, query=[])]}, {"features": [item, item]})
        for spec in bad:
            with self.assertRaises(ValueError):
                p.validator(spec)


if __name__ == "__main__":
    unittest.main()
