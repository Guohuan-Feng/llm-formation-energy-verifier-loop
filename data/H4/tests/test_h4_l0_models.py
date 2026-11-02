"""CPU software mocks only: no generated function, real model, CLI or TEST truth."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
import numpy as np
import h4_l0_models as m


class SandboxMock:
    calls = []

    def __init__(self, sources, **configuration):
        self.names = list(sources)
        self.configuration = configuration

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def evaluate_many(self, records, **options):
        ids = [row["id"] for row in records]
        SandboxMock.calls.append({"ids": ids, "configuration": self.configuration, "options": options})
        values = np.tile(np.array([10., 20., 30.]), (len(ids), 1))
        first_fit = [i for i in range(575) if i % 5 != 0]
        fail92, fail93 = set(first_fit[:92]), set(first_fit[:93])
        for row, rid in enumerate(ids):
            if rid.startswith("P"):
                i = int(rid[1:])
                if i in fail92:
                    values[row, 0] = np.nan
                if i >= 1500:
                    values[row, 1] = np.nan
                if i in fail93:
                    values[row, 2] = np.nan
            else:
                values[row, 0] = np.nan
        status = np.where(np.isfinite(values), "ok", "timeout")
        return {"ids": np.asarray(ids), "feature_names": np.asarray(self.names),
                "values": values, "status": status, "errors": [], "validation": [],
                "performance": {"mock_only": True}}


class FitterMock:
    def __init__(self):
        self.calls = []

    def __call__(self, train, y, predict):
        self.calls.append({"train": train.copy(), "y": y.copy(), "predict": predict.copy()})
        return np.full(len(predict), float(np.mean(y))), {"mock_only": True}


class Tests(unittest.TestCase):
    def setUp(self):
        top = m.safe(m.DATA / "software_model_tests")
        top.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=top)
        self.out = m.safe(self.temporary.name)
        pool = [f"P{i:04d}" for i in range(2500)]
        test = [f"T{i:04d}" for i in range(1000)]
        ids = pool + test
        groups = {rid: f"g{i:04d}" for i, rid in enumerate(ids)}
        mapping = {groups[rid]: "val" if i % 5 == 0 else "fit" for i, rid in enumerate(pool)}
        self.bundle = {"ids": ids, "index": {rid: i for i, rid in enumerate(ids)},
            "magpie132": np.zeros((len(ids), 132)), "structures_by_id": {rid: {} for rid in ids},
            "groups": groups, "pool_ids": pool, "test_ids": test, "source_hashes": {"mock": "fixed"},
            "perseed": {0: {"S0_ids": pool[:500], "fit_group_mapping": mapping,
                             "cutoff": {"sha256": "mock", "group": "mock"}}}}
        self.all_labels = {rid: float(i % 17) for i, rid in enumerate(pool)}
        self.features = [{"name": f"f{i}", "description": "CPU mock", "source": "def feature(s):\n    return 1.0\n"}
                         for i in range(3)]
        self.fitter = FitterMock()
        SandboxMock.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def labels(self, count):
        return {rid: self.all_labels[rid] for rid in self.bundle["pool_ids"][:count]}

    def evaluate(self, round_index=1, count=575):
        return m.evaluate_round(self.bundle, 0, round_index, self.features,
            self.labels(count), self.out / ("round" + str(round_index)),
            sandbox_factory=SandboxMock, model_fitter=self.fitter)

    def test_initial_is_magpie_only_s0_not_a_competing_round(self):
        initial = m.initial_state(self.bundle, 0, self.labels(500), self.out / "initial", model_fitter=self.fitter)
        self.assertEqual(initial["features"], [])
        self.assertEqual(initial["labelled_count"], 500)
        self.assertEqual((initial["fit_count"], initial["validation_count"]), (400, 100))
        self.assertFalse(initial["winner_eligible"])
        self.assertIsNone(initial["round_index"])
        self.assertEqual(self.fitter.calls[0]["train"].shape, (400, 132))
        self.assertEqual(SandboxMock.calls, [])
        self.assertEqual(len(initial["worst20_ids"]), 20)
        self.assertTrue(set(initial["worst20_ids"]) <= set(initial["identity"]["val_ids"]))
        for row in initial["worst20"]:
            self.assertEqual(row["label_eV_per_atom"], self.all_labels[row["id"]])
        again = m.initial_state(self.bundle, 0, self.labels(500), self.out / "initial", model_fitter=self.fitter)
        self.assertEqual(initial["record_sha256"], again["record_sha256"])
        self.assertEqual(len(self.fitter.calls), 1)

    def test_round_dynamic_membership_and_inner_only_twenty_percent_filter(self):
        round_ = self.evaluate()
        self.assertEqual(round_["labelled_count"], 575)
        self.assertEqual(round_["feature_selection"]["kept_names"], ["f0", "f1"])
        self.assertEqual(round_["feature_selection"]["dropped_names"], ["f2"])
        self.assertEqual([item["name"] for item in round_["active_features"]], ["f0", "f1"])
        self.assertEqual(round_["active_features"], self.features[:2])
        self.assertEqual(round_["inner_fit_features"]["failure_rates"][0], .2)
        self.assertEqual(round_["identity"]["train_ids"], self.bundle["pool_ids"][:575])
        self.assertFalse(round_["TEST_target_values_read"])
        self.assertTrue(all(rid.startswith("P") for rid in SandboxMock.calls[0]["ids"]))

    def test_template_membership_does_not_move_across_growing_rounds(self):
        first = self.evaluate()
        second = self.evaluate(2, 650)
        for group in ("fit_ids", "val_ids"):
            self.assertTrue(set(first["identity"][group]) <= set(second["identity"][group]))
        self.assertEqual(first["identity"]["fixed_template_assignment_sha256"],
                         second["identity"]["fixed_template_assignment_sha256"])

    def test_label_quota_and_test_target_input_fail_before_execution(self):
        with self.assertRaisesRegex(ValueError, "cumulative75"):
            self.evaluate(1, 576)
        bad = self.labels(500)
        bad["T0000"] = -999.
        with self.assertRaisesRegex(ValueError, "POOL"):
            m.evaluate_round(self.bundle, 0, 1, self.features, bad, self.out / "bad",
                             sandbox_factory=SandboxMock, model_fitter=self.fitter)
        self.assertEqual(SandboxMock.calls, [])
        self.assertEqual(self.fitter.calls, [])

    def test_initial_requires_exact_s0_and_whole_pool_fixed_map(self):
        with self.assertRaisesRegex(ValueError, "initial/current"):
            m.initial_state(self.bundle, 0, self.labels(501), self.out / "bad", model_fitter=self.fitter)
        del self.bundle["perseed"][0]["fit_group_mapping"]["g2499"]
        with self.assertRaisesRegex(ValueError, "every POOL"):
            self.evaluate()

    def test_round_cache_reuse_skips_fit_and_generated_execution(self):
        first, again = self.evaluate(), self.evaluate()
        self.assertEqual(first["record_sha256"], again["record_sha256"])
        self.assertEqual(len(SandboxMock.calls), 1)
        self.assertEqual(len(self.fitter.calls), 1)
        self.features[0]["source"] += "\n"
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            self.evaluate()

    def test_select_best_requires_twenty_rounds_and_tie_earliest(self):
        first = self.evaluate()
        with self.assertRaisesRegex(ValueError, "exactly20"):
            m.select_best([first])
        rounds = [{**first, "round_index": index, "validation_mae": 1.} for index in range(1, 21)]
        self.assertEqual(m.select_best(list(reversed(rounds)))["round_index"], 1)
        rounds[4] = {**rounds[4], "winner_eligible": False, "phase": "initial"}
        with self.assertRaisesRegex(ValueError, "round-validation"):
            m.select_best(rounds)

    def test_selected_round_cache_is_extended_only_for_new_final_train_members(self):
        selected = self.evaluate()
        old_hash = selected["artifacts"]["train_features"]["sha256"]
        final = m.finalize(self.bundle, 0, selected, self.labels(2000),
            self.bundle["test_ids"], self.out / "final", sandbox_factory=SandboxMock,
            model_fitter=self.fitter)
        self.assertEqual(final["selected_round_index"], 1)
        self.assertEqual(final["selected_validation_mae"], selected["validation_mae"])
        self.assertEqual(final["additional_train_structures_executed"], 1425)
        self.assertEqual(final["cached_train_structures_reexecuted"], 0)
        self.assertEqual(SandboxMock.calls[1]["ids"], self.bundle["pool_ids"][575:2000])
        self.assertEqual(SandboxMock.calls[2]["ids"], self.bundle["test_ids"])
        self.assertEqual(final["full_train_features"]["kept_names"], ["f0", "f2"])
        self.assertEqual(final["full_train_features"]["dropped_names"], ["f1"])
        self.assertTrue(np.isnan(self.fitter.calls[-1]["predict"][:, 132]).all())
        self.assertEqual(m.base._file_hash(selected["artifacts"]["train_features"]["path"]), old_hash)
        self.assertFalse(final["fill_changes_saved_validation_or_winner"])
        self.assertFalse(final["TEST_scoring_performed"])
        self.assertEqual(self.fitter.calls[-1]["train"].shape, (2000, 134))
        self.assertEqual(self.fitter.calls[-1]["predict"].shape, (1000, 134))
        with np.load(final["artifacts"]["full_train_features"]["path"], allow_pickle=False) as archive:
            with np.load(selected["artifacts"]["train_features"]["path"], allow_pickle=False) as old:
                np.testing.assert_equal(archive["values"][:575], old["values"])
                np.testing.assert_equal(archive["status"][:575], old["status"])
        cached = m.finalize(self.bundle, 0, selected, self.labels(2000),
            self.bundle["test_ids"], self.out / "final", sandbox_factory=SandboxMock,
            model_fitter=self.fitter)
        self.assertEqual(cached["record_sha256"], final["record_sha256"])
        self.assertEqual(len(SandboxMock.calls), 3)
        self.assertEqual(len(self.fitter.calls), 2)

    def test_unchanged2000_winner_has_no_additional_train_execution(self):
        selected = self.evaluate(20, 2000)
        final = m.finalize(self.bundle, 0, selected, self.labels(2000),
            self.bundle["test_ids"], self.out / "final", sandbox_factory=SandboxMock,
            model_fitter=self.fitter)
        self.assertEqual(final["additional_train_structures_executed"], 0)
        self.assertEqual(len(SandboxMock.calls), 2)

    def test_changed_old_labels_or_feature_artifact_refuses_final_fit(self):
        selected = self.evaluate()
        bad = self.labels(2000)
        bad["P0000"] += 1.
        with self.assertRaisesRegex(RuntimeError, "labels changed"):
            m.finalize(self.bundle, 0, selected, bad, self.bundle["test_ids"], self.out / "bad",
                       sandbox_factory=SandboxMock, model_fitter=self.fitter)
        path = m.safe(selected["artifacts"]["train_features"]["path"])
        path.write_bytes(b"tampered CPU mock only")
        with self.assertRaisesRegex(RuntimeError, "artifact hash"):
            m.finalize(self.bundle, 0, selected, self.labels(2000), self.bundle["test_ids"],
                       self.out / "bad", sandbox_factory=SandboxMock, model_fitter=self.fitter)
        self.assertEqual(len(self.fitter.calls), 1)

    def test_hard_paths_and_fixed_sandbox_limits(self):
        with self.assertRaisesRegex(ValueError, "phase4_2"):
            m.safe(ROOT / "work/h-data/H4/phase4_1/would_not_write")
        self.evaluate()
        call = SandboxMock.calls[0]
        self.assertEqual(call["configuration"], {"n_workers": 2, "timeout_ms": 50, "memory_limit_mb": 512})
        self.assertEqual(call["options"]["checkpoint_every"], 100)


if __name__ == "__main__":
    unittest.main()
