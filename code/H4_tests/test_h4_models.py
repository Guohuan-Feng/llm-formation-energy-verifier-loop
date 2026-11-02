"""CPU-only software mocks: never execute generated code or fit official models."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
import numpy as np
import h4_models as models


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
        records = list(records)
        ids = [record["id"] for record in records]
        SandboxMock.calls.append((ids, self.configuration, options))
        values = np.tile(np.array([10., 20., 30.])[:len(self.names)], (len(ids), 1))
        if ids[0].startswith("P"):
            values[:321, 0] = np.nan    # inner >20%; full <=20%.
            values[1599:, 1] = np.nan  # inner <=20%; full >20%.
            values[:320, 2] = np.nan   # exactly20% of inner-fit: keep.
        else:
            values[:, 0] = np.nan      # TEST failure must not change TRAIN mask.
        status = np.where(np.isfinite(values), "ok", "timeout")
        return {"ids": np.asarray(ids), "feature_names": np.asarray(self.names),
                "values": values, "status": status, "validation": [],
                "errors": [], "performance": {"mock_only": True}}


class ModelMock:
    def __init__(self):
        self.calls = []

    def __call__(self, train, y, predict):
        self.calls.append((train.copy(), y.copy(), predict.copy()))
        return np.full(len(predict), float(np.mean(y))), {"mock_only": True}


class TestModels(unittest.TestCase):
    def setUp(self):
        target = models.safe(ROOT / "work/h-data/H4/tests/mock_outputs")
        target.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=target)
        self.out = models.safe(self.temporary.name)
        pool = [f"P{i:04d}" for i in range(2000)]
        test = [f"T{i:04d}" for i in range(1000)]
        ids = pool + test
        self.bundle = {"ids": ids, "index": {rid: i for i, rid in enumerate(ids)},
                       "magpie132": np.zeros((3000, 132)),
                       "structures_by_id": {rid: {} for rid in ids},
                       "groups": {rid: "fit" if i < 1600 else "val" if i < 2000 else "TEST"
                                  for i, rid in enumerate(ids)},
                       "pool_ids": pool, "test_ids": test}
        self.split = {"O_ids": pool, "fit_ids": pool[:1600], "val_ids": pool[1600:]}
        self.labels = {rid: float(i % 11) for i, rid in enumerate(pool)}
        self.features = [{"name": f"f{i}", "description": "mock", "source": "def feature(s):\n    return 1.0\n"}
                         for i in range(3)]
        self.fitter = ModelMock()
        SandboxMock.calls = []

    def tearDown(self):
        self.temporary.cleanup()

    def candidate(self, index=0):
        return models.evaluate_candidate(self.bundle, 0, index, self.features,
                   self.labels, self.split, self.out / f"candidate{index}",
                   sandbox_factory=SandboxMock, model_fitter=self.fitter)

    def test_inner_mask_only_and_exact_twenty_percent_kept(self):
        result = self.candidate()
        self.assertEqual(result["feature_selection"]["kept_names"], ["f1", "f2"])
        self.assertEqual(result["feature_selection"]["dropped_names"], ["f0"])
        self.assertEqual(result["inner_fit_features"]["failure_rates"][2], .2)
        self.assertEqual(self.fitter.calls[0][0].shape, (1600, 134))
        self.assertEqual(self.fitter.calls[0][2].shape, (400, 134))
        self.assertFalse(result["test_target_values_read"])
        self.assertEqual(result["test_structures_executed"], 0)
        self.assertTrue(all(rid.startswith("P") for rid in SandboxMock.calls[0][0]))

    def test_completed_candidate_skips_sandbox_and_fit(self):
        first = self.candidate()
        second = self.candidate()
        self.assertEqual(first["record_sha256"], second["record_sha256"])
        self.assertEqual(len(SandboxMock.calls), 1)
        self.assertEqual(len(self.fitter.calls), 1)

    def test_candidate_refuses_changed_binding(self):
        self.candidate()
        self.features[0]["source"] += "\n"
        with self.assertRaisesRegex(RuntimeError, "binding mismatch"):
            self.candidate()
        self.assertEqual(len(SandboxMock.calls), 1)

    def test_candidate_refuses_modified_completed_artifact(self):
        first = self.candidate()
        path = models.safe(first["artifacts"]["train_features"]["path"])
        path.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "artifact hash"):
            self.candidate()

    def test_winner_only_test_fulltrain_mask_and_cache_reuse(self):
        candidate = self.candidate()
        final = models.refit_predict(self.bundle, 0, candidate, self.labels,
                 self.bundle["test_ids"], self.out / "final", sandbox_factory=SandboxMock,
                 model_fitter=self.fitter)
        self.assertEqual(final["full_train_features"]["kept_names"], ["f0", "f2"])
        self.assertEqual(final["full_train_features"]["dropped_names"], ["f1"])
        self.assertEqual(self.fitter.calls[1][0].shape, (2000, 134))
        self.assertEqual(self.fitter.calls[1][2].shape, (1000, 134))
        self.assertTrue(np.isnan(self.fitter.calls[1][2][:, 132]).all())
        self.assertFalse(final["test_failure_rates_used_for_selection"])
        self.assertFalse(final["test_scoring_performed"])
        self.assertFalse(final["training_feature_cache_reexecuted"])
        again = models.refit_predict(self.bundle, 0, candidate, self.labels,
                 self.bundle["test_ids"], self.out / "final", sandbox_factory=SandboxMock,
                 model_fitter=self.fitter)
        self.assertEqual(final["record_sha256"], again["record_sha256"])
        self.assertEqual(len(self.fitter.calls), 2)
        self.assertEqual(len(SandboxMock.calls), 2)
        self.assertTrue(all(rid.startswith("T") for rid in SandboxMock.calls[1][0]))

    def test_test_target_mapping_cannot_be_supplied(self):
        self.labels["T0000"] = -123.
        with self.assertRaisesRegex(ValueError, "exact labelled TRAIN"):
            self.candidate()
        self.assertEqual(SandboxMock.calls, [])

    def test_template_overlap_stops_before_execution(self):
        self.bundle["groups"]["P1999"] = "fit"
        with self.assertRaisesRegex(ValueError, "template groups overlap"):
            self.candidate()
        self.assertEqual(SandboxMock.calls, [])

    def test_predeclared_tie_break_is_earliest(self):
        candidate = self.candidate()
        other = {**candidate, "candidate_index": 1}
        self.assertEqual(models.select_best([other, candidate])["candidate_index"], 0)

    def test_workers_deadline_checkpoint_settings(self):
        self.candidate()
        configuration = SandboxMock.calls[0][1]
        options = SandboxMock.calls[0][2]
        self.assertEqual(configuration, {"n_workers": 2, "timeout_ms": 50, "memory_limit_mb": 512})
        self.assertEqual(options["checkpoint_every"], 100)

    def test_namespace_copy_retains_original_scientific_sandbox(self):
        original = (ROOT / "h3b_sandbox.py").read_text(encoding="utf-8")
        expected = original.replace("H3b", "H4").replace("work/h-data/H3b", "work/h-data/H4")
        expected = expected.replace("ROOT=Path(__file__).resolve().parent", "ROOT=Path(__file__).resolve().parents[4]")
        expected = expected.replace("import h3_sandbox as old", "ROOT=Path(__file__).resolve().parents[4]\nif str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))\nimport h3_sandbox as old")
        self.assertEqual(expected, (ROOT / "work/h-data/H4/src/h4_sandbox.py").read_text(encoding="utf-8"))

    def test_real_fit_wrapper_uses_only_train_imputer_unweighted_fit(self):
        actions = []
        class Imputer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                actions.append(("imputer", kwargs))
            def fit_transform(self, values):
                actions.append(("fit_transform", values.copy()))
                return np.nan_to_num(values)
            def transform(self, values):
                actions.append(("transform", values.copy()))
                return np.nan_to_num(values)
        class Estimator:
            def __init__(self, **kwargs):
                actions.append(("estimator", kwargs))
            def fit(self, values, y):
                actions.append(("fit", values.copy(), y.copy()))
            def predict(self, values):
                actions.append(("predict", values.copy()))
                return np.ones(len(values))
        ensemble = types.ModuleType("sklearn.ensemble")
        ensemble.HistGradientBoostingRegressor = Estimator
        impute = types.ModuleType("sklearn.impute")
        impute.SimpleImputer = Imputer
        with patch.dict(sys.modules, {"sklearn.ensemble": ensemble, "sklearn.impute": impute}):
            predictions, metadata = models._fit_hgb(np.array([[np.nan], [2.]]), np.array([1., 2.]), np.array([[9.]]))
        self.assertEqual(actions[0], ("imputer", {"strategy": "median", "keep_empty_features": True}))
        self.assertEqual(actions[3], ("estimator", models.HGB_PARAMETERS))
        self.assertEqual([action[0] for action in actions].count("fit"), 1)
        self.assertEqual(metadata["training_weights"], "unweighted")
        self.assertEqual(predictions.tolist(), [1.])


if __name__ == "__main__":
    unittest.main()
