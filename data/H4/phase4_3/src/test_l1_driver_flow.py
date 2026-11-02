"""Finite L1 driver flow mocks: zero CLI, fits, features, real labels or output writes."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[4]
sys.path[:0] = [str(SRC), str(ROOT / "work/h-data/H4/src")]
import run_h4_l1 as d
import l1_report


class VirtualStorage:
    """All mutations are dictionary entries; no filesystem write is allowed."""
    def __init__(self):
        self.phase = SRC.parent / "software" / "virtual_driver"
        self.out = self.phase / "public"
        self.events = []
        self.files = {}
        self.cfg = {"prompts": {"input_limit": 60000, "schema": {}}, "runtime": {}, "protocol_files": {},
            "statistics": {"marker_contract": {"stage": "4.3", "condition": "L1", "scoring_pass": 1}}}
        self.states = {s: {"seed": s, "round_completed": 0, "labels": {f"P{s}": -1.0},
            "carry": 0, "history": [], "round_model_paths": [], "query_history": [],
            "memory_audits": [], "feedback_model_path": "initial"} for s in (0, 1, 2)}
        self.files[self.phase / "config.json"] = self.cfg
        for seed, state in self.states.items(): self.files[self.phase / "states" / f"seed{seed}.json"] = state
        self.files[self.out / "metrics.json"] = {"status": "prepared"}

    def read(self, path):
        return deepcopy(self.files[Path(path)])

    def write(self, path, value, *, exclusive=False):
        path = Path(path)
        if exclusive and path in self.files: raise FileExistsError(path.name)
        self.events.append(("write", path.name))
        self.files[path] = deepcopy(value)

    def update(self, status, phase, **values):
        self.events.append(("update", status, phase))
        self.files[self.out / "metrics.json"].update(status=status, phase=phase, **deepcopy(values))

    def patches(self, stack):
        original_exists = Path.exists
        stack.enter_context(patch.object(d, "PHASE", self.phase))
        stack.enter_context(patch.object(d.c, "PHASE", self.phase))
        stack.enter_context(patch.object(d.c, "OUT", self.out))
        stack.enter_context(patch.object(d.c, "read", side_effect=self.read))
        stack.enter_context(patch.object(d.c, "write", side_effect=self.write))
        stack.enter_context(patch.object(d.c, "text", side_effect=lambda p, v: self.write(p, v)))
        stack.enter_context(patch.object(d.c, "bytes_write", side_effect=lambda p, v, **kw: self.write(p, v, **kw)))
        stack.enter_context(patch.object(d.c, "update", side_effect=self.update))
        stack.enter_context(patch.object(d.c, "verify", return_value={"status": "PASS"}))
        stack.enter_context(patch.object(d.c, "readme", side_effect=lambda status, **kw: self.events.append(("readme", status))))
        stack.enter_context(patch.object(d.c, "log", side_effect=lambda text: self.events.append(("log", text))))
        stack.enter_context(patch.object(d.c, "sha", return_value="fixed-hash"))
        stack.enter_context(patch.object(d.c, "RunLock", side_effect=lambda: nullcontext()))
        stack.enter_context(patch.object(d.c, "Awake", side_effect=lambda: nullcontext()))
        stack.enter_context(patch.object(d.c, "process_identity", return_value={"pid": 1}))
        stack.enter_context(patch.object(Path, "exists", lambda path: path in self.files if path.is_relative_to(self.phase) else original_exists(path)))


class FakeClient:
    def __init__(self, limit, events):
        self.n_calls = 120
        self.responses = [None] * 120
        self._e2 = SimpleNamespace(INPUT_LIMIT=limit)
        self._canonical_owned_call_once = SimpleNamespace(__globals__={"_e2": self._e2})
        self.stop = SimpleNamespace(is_set=lambda: False)
        self.events = events

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def summary(self): return {"cli_attempt_count": self.n_calls, "stop_reason": None}
    def request(self, spec, validator):
        self.events.append(("dispatch", spec["request_id"]))
        self.n_calls += 1
        return {"status": "ok", "response": {}, "logical_elapsed_seconds": 1.}


class DriverFlowTests(unittest.TestCase):
    def test_whole_batch_memory_or_input_guard_stops_before_any_dispatch(self):
        for failure_type in ("memory", "input"):
            v = VirtualStorage()
            client = FakeClient(60000, v.events)
            budget = {"token_count": 7000, "input_token_upper_bound": 70000, "input_limit": 60000,
                "payload": {}, "text": "private", "archive": {}}
            error = d.memory.MemoryBudgetExceeded(budget) if failure_type == "memory" else d.prompts.PromptBudgetExceeded(budget)
            mem = {"audit": {}, "payload_sha256": "memory"}
            spec = {"request_id": "L1_seed0_round02", "schema": {}, "capacity": 75,
                "memory_token_count": 1, "memory_UTF8_bytes": 2, "memory_trimmed": False,
                "round_index": 2, "input_budget": {}, "format_repair_input_budget": {}}
            states = deepcopy(v.states)
            for state in states.values(): state["round_completed"] = 1
            with ExitStack() as stack:
                v.patches(stack)
                stack.enter_context(patch.object(d.base, "_read_sealed", return_value={"active_features": [], "validation_mae": .2, "worst20": []}))
                stack.enter_context(patch.object(d.memory, "build_memory", side_effect=[mem, error] if failure_type == "memory" else [mem, mem]))
                stack.enter_context(patch.object(d.prompts, "build_request", side_effect=[spec, error] if failure_type == "input" else [spec]))
                with self.assertRaises(type(error)):
                    d.request_batch(client, {"ids": []}, states, 2, v.cfg)
            self.assertFalse(any(e[0] == "dispatch" for e in v.events))
            stop = v.files[v.phase / "budget_stop.json"]
            self.assertTrue(stop["request_not_sent"])
            self.assertNotIn("payload", stop["details"])
            self.assertNotIn("text", stop["details"])
            self.assertNotIn("archive", stop["details"])

    def _run_harness(self, *, proceed):
        v = VirtualStorage()
        client = FakeClient(60000, v.events)
        ctor = []
        def transport_factory(*args, **kwargs):
            ctor.append(kwargs)
            return client
        def request_batch(mock_client, bundle, states, rnd, cfg, seeds=(0, 1, 2)):
            records = []
            for seed in seeds:
                self.assertEqual(rnd - 1, states[seed]["round_completed"])
                states[seed]["round_completed"] = rnd
                mock_client.n_calls += 1
                v.events.append(("logical_request", rnd, seed))
                records.append({"seed": seed, "round_index": rnd, "request_elapsed_seconds": 1.,
                    "baseline_model_seconds": 1., "lofo_seconds": 1., "labelled_count": 500 + 75 * rnd})
            return {"records": records, "batch_elapsed_seconds": 1.}
        def forecast(records, states, elapsed, cfg):
            v.events.append(("forecast", len(records)))
            self.assertEqual(8, client.n_calls - 120)
            return {"proceed": proceed, "estimated_total_seconds": 36001 if not proceed else 100,
                "estimated_remaining_seconds": 50, "pilot_requests": 8}
        def finish(states, bundle, cfg, transport_client, started):
            self.assertEqual([20, 20, 20], [states[s]["round_completed"] for s in (0, 1, 2)])
            self.assertEqual(180, transport_client.n_calls)
            v.events.append(("finish",))
            return {"status": "complete"}
        with ExitStack() as stack:
            v.patches(stack)
            stack.enter_context(patch.object(d.data, "load_bundle", return_value={"ids": []}))
            stack.enter_context(patch.object(d.data, "initial_labels", side_effect=lambda seed, **kw: deepcopy(v.states[seed]["labels"])))
            stack.enter_context(patch.object(d.wire, "StrictSchemaTransport", side_effect=transport_factory))
            stack.enter_context(patch.object(d, "request_batch", side_effect=request_batch))
            stack.enter_context(patch.object(d, "forecast", side_effect=forecast))
            stack.enter_context(patch.object(d, "finish", side_effect=finish))
            stack.enter_context(patch.object(d, "verify", return_value={"status": "complete"}))
            stack.enter_context(patch.object(d.transport, "INPUT_LIMIT", 20000))
            result = d.run()
        return v, client, ctor, result

    def test_eighth_request_forecast_over10h_prevents_ninth_and_terminal(self):
        v, client, ctor, result = self._run_harness(proceed=False)
        logical = [x for x in v.events if x[0] == "logical_request"]
        self.assertEqual(8, len(logical))
        self.assertEqual([(3, 0), (3, 1)], [(x[1], x[2]) for x in logical[-2:]])
        self.assertEqual(128, client.n_calls)
        self.assertEqual("stopped_requires_user", result["status"])
        self.assertFalse(any(x[0] == "finish" for x in v.events))
        self.assertNotIn(v.phase / "evaluation_started.json", v.files)

    def test_finite_run_exact20_per_seed_60new_calls_preserves_global_caps(self):
        v, client, ctor, result = self._run_harness(proceed=True)
        logical = [x for x in v.events if x[0] == "logical_request"]
        self.assertEqual(60, len(logical))
        self.assertEqual(20, len({x[1] for x in logical}))
        for seed in (0, 1, 2): self.assertEqual(list(range(1, 21)), [x[1] for x in logical if x[2] == seed])
        eighth = v.events.index(logical[7]); ninth = v.events.index(logical[8])
        forecast = next(i for i, x in enumerate(v.events) if x[0] == "forecast")
        self.assertLess(eighth, forecast); self.assertLess(forecast, ninth)
        self.assertEqual(180, client.n_calls)
        self.assertEqual(300, ctor[0]["max_calls"])
        self.assertEqual(180, ctor[0]["max_logical_requests"])
        self.assertEqual(4, ctor[0]["max_workers"])
        self.assertEqual("complete", result["status"])
        self.assertFalse(v.files[v.phase / "run_stop.json"]["next_stage_started"])

    def test_winner_saved_before_fill_and_single_terminal_gate_after_all_predictions(self):
        v = VirtualStorage()
        states = deepcopy(v.states)
        for seed, state in states.items():
            state.update(round_completed=20, round_model_paths=[f"round{n}" for n in range(1, 21)],
                labels={f"P{seed}-{n}": -1. for n in range(1999)})
        v.files[v.phase / "pilot_forecast.json"] = {"estimated_total_seconds": 100}
        v.files[d.c.ARCHIVE / "predictions.csv"] = b"oldOandL0\n"
        client = FakeClient(60000, v.events); client.n_calls = 180
        truth_reads = []
        def fill(**kwargs):
            self.assertIn(v.phase / "selected_rounds.json", v.files)
            self.assertIn(v.phase / "selection_pre_fill.json", v.files)
            v.events.append(("fill", kwargs["seed"]))
            return {"selected_ids": [f"new{kwargs['seed']}"], "selected_n": 1}
        def finalize(bundle, seed, selected, labels, test_ids, output_dir, **kwargs):
            self.assertEqual(2000, len(labels))
            v.events.append(("predict", seed))
            return {"seed": seed}
        def evaluate(arms, bundle, *, evaluation_marker):
            self.assertIn(evaluation_marker, v.files)
            self.assertEqual(3, len([x for x in v.events if x[0] == "predict"]))
            truth_reads.append(1)
            return {"TEST_target_accessor_calls": 1}
        with ExitStack() as stack:
            v.patches(stack)
            # The CSV archive read is intercepted; no actual public file is read or written.
            original_read_bytes = Path.read_bytes
            stack.enter_context(patch.object(Path, "read_bytes", lambda path: v.files[path] if path in v.files else original_read_bytes(path)))
            stack.enter_context(patch.object(d.base, "_read_sealed", side_effect=lambda path: {"round_index": int(str(path)[5:]), "validation_mae": .2}))
            stack.enter_context(patch.object(d.models, "select_best", side_effect=lambda rounds: rounds[0]))
            stack.enter_context(patch.object(d.queries, "final_fill_only_no_reveal", side_effect=fill))
            stack.enter_context(patch.object(d.data, "reveal", side_effect=lambda ids, seed, **kw: {rid: -1. for rid in ids}))
            stack.enter_context(patch.object(d.data, "validate_labels", return_value=True))
            stack.enter_context(patch.object(d.models, "finalize", side_effect=finalize))
            arms = [{"arm_id": f"L1_seed{s}", "seed": s, "prediction_rows": [{"id": "mockTEST", "prediction_eV_per_atom": -.5}]} for s in (0, 1, 2)]
            stack.enter_context(patch.object(d, "prediction_arms", return_value=arms))
            stack.enter_context(patch.object(d.stats, "_normalise", return_value=True))
            stack.enter_context(patch.object(d.stats, "evaluate", side_effect=evaluate))
            stack.enter_context(patch.object(d.stats, "private_pairing_cache", return_value={}))
            stack.enter_context(patch.object(d.base, "_write_npz", side_effect=lambda path, **kw: v.write(path, {})))
            stack.enter_context(patch.object(d.stats, "paired_comparisons", return_value={"mock": True}))
            stack.enter_context(patch.object(l1_report, "make_report", return_value={"mock": True}))
            result = d.finish(states, {"test_ids": ["mockTEST"]}, v.cfg, client, time.monotonic())
        self.assertEqual(1, len(truth_reads))
        self.assertEqual(1, len([x for x in v.events if x == ("write", "evaluation_started.json")]))
        self.assertEqual("complete", result["status"])
        self.assertFalse(v.files[v.out / "metrics.json"]["next_stage_started"])


if __name__ == "__main__": unittest.main()
