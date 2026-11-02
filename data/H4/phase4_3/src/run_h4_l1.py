"""Finite authorized stage4.3; one L1 TEST gate, no stage4.4 or auto-resume."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
SRC = Path(__file__).resolve().parent
PHASE = SRC.parent
ROOT = PHASE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "work/h-data/H4/src"))
sys.path.insert(0, str(SRC))
import os
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import io
import json
import statistics
import time
import traceback
import numpy as np
import l1_common as c
import l1_counter as counter
import l1_data as data
import l1_memory as memory
import l1_models as models
import l1_prompts as prompts
import l1_stats as stats
import l1_queries as queries
import h4_l0_wire as wire
import h4_transport as transport
import h4_models as base

FILES = ["l1_common.py", "l1_counter.py", "l1_data.py", "l1_memory.py", "l1_models.py", "l1_queries.py",
         "l1_prompts.py", "l1_stats.py", "l1_report.py", "run_h4_l1.py"]


class ScientificStop(RuntimeError):
    pass


def state_path(seed):
    return PHASE / "states" / ("seed" + str(seed) + ".json")


def progress(event):
    c.write(PHASE / "current_operation.json", {"utc": c.utc(), **event})
    if event.get("phase") in ("round_complete", "winner_prediction_complete", "lofo_complete"):
        c.log(event["phase"] + " seed=" + str(event.get("seed")) + " round=" + str(event.get("round_index")))


def prepare():
    if (PHASE / "frozen_config.json").exists():
        return verify()
    started = time.monotonic()
    preserved = c.preserve()
    counter.verify()
    projection = c.read(PHASE / "input_limit_projection_L1.json")
    input_limit = projection["selected_input_limit_bytes"]
    if type(input_limit) is not int or input_limit <= 0:
        raise ScientificStop("Completed measured L1 input-limit projection is required before freeze")
    if not (PHASE / "input_limit_amendment_L1.md").is_file():
        raise ScientificStop("L1 pre-freeze input-limit record is missing")
    old = c.read(c.ARCHIVE / "config.json")
    bundle = data.load_bundle()
    first = []
    for seed in (0, 1, 2):
        labels = data.initial_labels(seed, bundle=bundle)
        data.validate_labels(labels, seed, bundle=bundle, require_n=500)
        initial = models.initial_state(bundle, seed, labels, PHASE / "initial" / ("L1_seed" + str(seed)), progress=progress)
        state = {"seed": seed, "condition": "L1", "round_completed": 0, "labels": labels,
            "carry": 0, "round_model_paths": [], "query_history": [], "history": [],
            "memory_audits": [], "feedback_model_path": initial["metadata_path"],
            "final_fill": None, "status": "initialized"}
        if state_path(seed).exists():
            if c.read(state_path(seed)) != state:
                raise ScientificStop("Pre-freeze L1 initial state differs")
        else:
            c.write(state_path(seed), state, exclusive=True)
        spec = prompts.build_request(seed, 1, [], initial["validation_mae"], initial["worst20"],
            labels, [], input_limit=input_limit, bundle=bundle)
        c.write(PHASE / "initial_requests" / (spec["request_id"] + ".json"), spec, exclusive=True)
        first.append({k: spec[k] for k in ("request_id", "canonical_prompt_sha256", "input_budget",
            "format_repair_input_budget", "memory_token_count", "memory_UTF8_bytes")})
    schemas = prompts.configuration(input_limit=input_limit, bundle=bundle)
    if prompts.response_schema(bundle, 75) != prompts.response_schema(bundle, 1500):
        raise ScientificStop("L1 round schema is not fixed across quotas")
    inputs = {**old["input_hashes"], **bundle["source_hashes"]}
    for path in (PHASE / "token_counter.json", PHASE / "input_projection_L1.json",
            PHASE / "input_limit_projection_L1.json", PHASE / "input_limit_amendment_L1.md",
            PHASE / "prior_stages_preservation.json", c.ARCHIVE / "config.json"):
        inputs[str(path.resolve())] = c.sha(path)
    global_context = {x["path"]: x["sha256"] for x in c.read(PHASE / "input_projection_L1.json")["global_instructions"]}
    cfg = {"protocol": "H4-stage4.3-L1-new-freeze-v1", "stage": "4.3", "condition": "L1",
        "frozen_utc": c.utc(), "runtime": c.RUNTIME, "data": data.configuration(bundle=bundle),
        "prompts": schemas, "memory": memory.configuration(), "token_counter": counter.verify(),
        "models": {**old["models"], "parameters": models.HGB_PARAMETERS,
            "initialization": "Reuse exact original L0 S0500 Magpie-only baseline; never a candidate",
            "memory_contribution": "LOFO refit on identical current inner TRAIN/validation; delta=MAE_without-minus-full; no TEST"},
        "statistics": stats.configuration(), "first_round_requests": first,
        "budget": {**old["budget"], "prior_stage4_1_calls": 60, "prior_stage4_2_calls": 60,
            "prior_global_calls": 120, "remaining_before_L1": 180, "logical_round_requests": 60,
            "H4_global_CLI_cap": 300, "H4_global_logical_cap": 180,
            "pilot": "first8 L1 logical requests: rounds1/2 all3 plus round3 seed0/1; before ninth request",
            "stage4_3_started": True, "stage4_4_started": False},
        "protocol_files": old["protocol_files"], "input_limit_basis": projection,
        "source_hashes": {str((SRC / name).resolve()): c.sha(SRC / name) for name in FILES},
        "input_hashes": inputs, "global_context_hashes": global_context,
        "prior_stages": {"preservation_manifest": str(PHASE / "prior_stages_preservation.json"),
            "O_terminal_sha256": preserved["O_terminal_sha256"], "L0_terminal_sha256": preserved["L0_terminal_sha256"]},
        "prepare_elapsed_seconds": time.monotonic() - started}
    c.write(PHASE / "config.json", cfg, exclusive=True)
    c.write(PHASE / "frozen_config.json", {"sha256": c.sha(PHASE / "config.json"), "utc": c.utc(),
        "amendment_sha256": c.sha(PHASE / "input_limit_amendment_L1.md"),
        "token_counter_sha256": c.sha(PHASE / "token_counter.json")}, exclusive=True)
    c.bytes_write(c.OUT / "config.json", (PHASE / "config.json").read_bytes())
    c.write(c.OUT / "metrics.json", {"stage": "4.3", "condition": "L1", "status": "prepared",
        "phase": "frozen_before_first_L1_call", "updated_utc": c.utc(), "rounds_complete": 0,
        "rounds_total": 60, "LLM_calls_global": 120, "LLM_calls_stage4_3": 0,
        "prior_H4_TEST_scoring_passes": 2, "current_L1_scoring_pass": 0,
        "TEST_targets_read_this_condition": 0, "next_stage_started": False})
    c.readme("prepared", input_limit=input_limit)
    c.log("Stage4.3 frozen; exact memory counter6000; real-request guard=" + str(input_limit) + "; no inference yet")
    return c.verify(cfg)


def request_batch(client, bundle, states, round_index, cfg, seeds=(0, 1, 2)):
    started = time.monotonic()
    specs, results, audits = {}, {}, {}
    for seed in seeds:
        state = states[seed]
        if state["round_completed"] != round_index - 1:
            raise ScientificStop("Seed round ordering changed")
        feedback = base._read_sealed(state["feedback_model_path"])
        try:
            mem = memory.build_memory(state["history"], private_ids=bundle["ids"])
            spec = prompts.build_request(seed, round_index, feedback["active_features"], feedback["validation_mae"],
                feedback["worst20"], state["labels"], mem, input_limit=cfg["prompts"]["input_limit"],
                carry=state["carry"], bundle=bundle)
        except (memory.MemoryBudgetExceeded, prompts.PromptBudgetExceeded) as error:
            details = getattr(error, "result", getattr(error, "budget", getattr(error, "audit", {})))
            if isinstance(details, dict):
                details = {key: value for key, value in details.items() if key not in ("payload", "text", "archive")}
            c.write(PHASE / "budget_stop.json", {"seed": seed, "round_index": round_index,
                "utc": c.utc(), "error_type": type(error).__name__, "details": details,
                "request_not_sent": True, "source_truncation": False})
            raise
        if spec["schema"] != cfg["prompts"]["schema"]:
            raise ScientificStop("Fixed L1 schema changed")
        if round_index == 1 and spec != c.read(PHASE / "initial_requests" / (spec["request_id"] + ".json")):
            raise ScientificStop("Frozen first-round L1 prompt changed")
        specs[seed] = spec
        audits[seed] = {**{key: spec[key] for key in ("request_id", "round_index", "memory_token_count",
            "memory_UTF8_bytes", "memory_trimmed", "input_budget", "format_repair_input_budget")},
            "memory_policy_audit": mem["audit"], "memory_payload_sha256": mem["payload_sha256"]}
    # Every actual payload in this batch passes both guards before any model call.
    for seed in seeds:
        spec = specs[seed]
        c.write(PHASE / "requests" / (spec["request_id"] + ".json"), spec, exclusive=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(client.request, specs[seed], prompts.validator(bundle, specs[seed]["capacity"])): seed for seed in seeds}
        for future in as_completed(pending):
            seed = pending[future]
            results[seed] = future.result()
            c.write(PHASE / "logical_results" / (specs[seed]["request_id"] + ".json"), results[seed], exclusive=True)
    if client.stop.is_set():
        raise ScientificStop("Finite transport stop: " + str(client.summary()["stop_reason"]))
    records = []
    for seed in seeds:
        state, spec = states[seed], specs[seed]
        payload, response_audit = prompts.resolve_response(results[seed], bundle=bundle, capacity=spec["capacity"])
        query = queries.select_only_no_reveal({"queries": payload["queries"], "rationale": payload["rationale"]},
            arm="L1", seed=seed, round_index=round_index, labeled_ids=sorted(state["labels"]),
            bundle=bundle, carry=state["carry"])
        selected = query["selected_ids"]
        labels = {**state["labels"], **(data.reveal(selected, seed, bundle=bundle) if selected else {})}
        data.validate_labels(labels, seed, bundle=bundle)
        query_record = {"stage": "4.3", "condition": "L1", "seed": seed, "round_index": round_index,
            "request_id": spec["request_id"], "canonical_prompt_sha256": spec["canonical_prompt_sha256"],
            "query_payload": {"queries": payload["queries"], "rationale": payload["rationale"]},
            "query_selection": query, "response_audit": response_audit, "labels_before": len(state["labels"]),
            "labels_after": len(labels)}
        c.write(PHASE / "query_records" / (spec["request_id"] + ".json"), query_record, exclusive=True)
        model = models.evaluate_round(bundle, seed, round_index, payload["features"], labels,
            PHASE / "rounds" / ("L1_seed" + str(seed)) / ("round" + str(round_index).zfill(2)),
            progress=progress, query_record=query_record, notes=payload["notes"])
        state.update(labels=labels, carry=query["carry_out"], round_completed=round_index,
            feedback_model_path=model["metadata_path"], status="round_complete")
        state["round_model_paths"].append(model["metadata_path"])
        state["query_history"].append(query_record)
        state["history"].append(model["history_record"])
        state["memory_audits"].append(audits[seed])
        c.write(state_path(seed), state)
        memory.persist_archive(PHASE / "memory_archives" / (spec["request_id"] + ".json"), state["history"], private_ids=bundle["ids"])
        c.write(PHASE / "request_byte_ledger" / (spec["request_id"] + ".json"), audits[seed], exclusive=True)
        c.update("running", "loop_round_validation", rounds_complete=sum(x["round_completed"] for x in states.values()),
            round_index=round_index, seed=seed, labels_by_seed={str(k): len(v["labels"]) for k, v in states.items()},
            LLM_calls_global=client.n_calls, LLM_calls_stage4_3=client.n_calls - 120, transport=client.summary())
        records.append({"seed": seed, "round_index": round_index, "labelled_count": len(labels),
            "request_elapsed_seconds": results[seed]["logical_elapsed_seconds"],
            "baseline_model_seconds": model["elapsed_seconds"], "lofo_seconds": model["lofo"]["elapsed_seconds"]})
    return {"records": records, "batch_elapsed_seconds": time.monotonic() - started}


def forecast(pilot_records, states, elapsed, cfg):
    if len(pilot_records) != 8:
        raise ScientificStop("Pilot must contain exactly first8 completed logical L1 requests")
    mean_latency = max(statistics.mean(x["request_elapsed_seconds"] for x in pilot_records if x["seed"] == seed) for seed in (0, 1, 2))
    validation_per_row = max((x["baseline_model_seconds"] + x["lofo_seconds"]) / x["labelled_count"] for x in pilot_records)
    remaining_rows = sum(sum(500 + 75 * rnd for rnd in range(state["round_completed"] + 1, 21)) for state in states.values())
    generation = max(20 - state["round_completed"] for state in states.values()) * mean_latency
    model_seconds = remaining_rows * validation_per_row
    total = elapsed + generation + model_seconds + 3 * 3000 * validation_per_row + 600
    return {"utc": c.utc(), "pilot_requests": 8, "pilot_logical_requests": 8,
        "elapsed_seconds_at_pilot": elapsed, "estimated_total_seconds": total,
        "estimated_remaining_seconds": total - elapsed, "proceed": total <= 36000,
        "threshold_seconds": 36000, "observed_validation_and_LOFO_seconds_per_labelled_row": validation_per_row,
        "remaining_generation_seconds": generation, "remaining_validation_and_LOFO_seconds": model_seconds,
        "caveat": "Provider latency, function cost and future retained-feature widths can change; this is a forecast, not measured total runtime."}


def prediction_arms(finals, bundle):
    arms = []
    for item in finals:
        artifact = item["artifacts"]["predictions"]
        if c.sha(artifact["path"]) != artifact["sha256"]:
            raise ScientificStop("L1 final prediction hash changed")
        with np.load(artifact["path"], allow_pickle=False) as cache:
            ids, predictions = cache["ids"].tolist(), cache["predictions"].tolist()
        if ids != bundle["test_ids"]:
            raise ScientificStop("L1 final fixed1000TEST identity changed")
        arms.append({"arm_id": "L1_seed" + str(item["seed"]), "condition": "L1", "seed": item["seed"],
            "status": "complete", "method": "HGB_Magpie132_LLM_memory_features",
            "prediction_rows": [{"id": rid, "prediction_eV_per_atom": float(value)} for rid, value in zip(ids, predictions)]})
    return arms


def finish(states, bundle, cfg, client, started):
    selected = [models.select_best([base._read_sealed(path) for path in states[seed]["round_model_paths"]]) for seed in (0, 1, 2)]
    c.write(PHASE / "selected_rounds.json", selected, exclusive=True)
    c.write(PHASE / "selection_pre_fill.json", {"utc": c.utc(), "selected_rounds_sha256": c.sha(PHASE / "selected_rounds.json"),
        "states_before_fill": {str(seed): c.sha(state_path(seed)) for seed in (0, 1, 2)},
        "fill_labels_not_yet_revealed": True}, exclusive=True)
    for seed in (0, 1, 2):
        state = states[seed]
        fill = queries.final_fill_only_no_reveal(arm="L1", seed=seed, labeled_ids=sorted(state["labels"]),
            bundle=bundle, total_new_revealed=len(state["labels"]) - 500)
        state["labels"].update(data.reveal(fill["selected_ids"], seed, bundle=bundle) if fill["selected_ids"] else {})
        data.validate_labels(state["labels"], seed, bundle=bundle, require_n=2000)
        state.update(final_fill=fill, status="final_labels_ready")
        c.write(state_path(seed), state)
    finals = []
    for seed in (0, 1, 2):
        finals.append(models.finalize(bundle, seed, selected[seed], states[seed]["labels"], bundle["test_ids"],
            PHASE / "final" / ("L1_seed" + str(seed)), progress=progress))
        c.update("running", "winner_TEST_predictions", final_prediction_arms_complete=len(finals))
    arms = prediction_arms(finals, bundle)
    stats._normalise(arms, bundle["test_ids"])
    c.write(PHASE / "committed_prediction_arms.json", arms, exclusive=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["arm_id", "condition", "seed", "split", "id", "prediction_eV_per_atom"], lineterminator="\n")
    for arm in arms:
        for row in arm["prediction_rows"]:
            writer.writerow({"arm_id": arm["arm_id"], "condition": "L1", "seed": arm["seed"], "split": "TEST", **row})
    c.bytes_write(c.OUT / "predictions.csv", (c.ARCHIVE / "predictions.csv").read_bytes() + stream.getvalue().encode())
    c.verify(cfg)
    marker = PHASE / "evaluation_started.json"
    c.write(marker, {**cfg["statistics"]["marker_contract"], "utc": c.utc(),
        "config_sha256": c.sha(PHASE / "config.json"),
        "committed_prediction_arms_sha256": c.sha(PHASE / "committed_prediction_arms.json"),
        "public_predictions_sha256": c.sha(c.OUT / "predictions.csv")}, exclusive=True)
    c.update("running", "terminal_scoring")
    terminal = stats.evaluate(arms, bundle, evaluation_marker=marker)
    c.write(PHASE / "terminal_results.json", terminal, exclusive=True)
    base._write_npz(PHASE / "private_pairing_cache.npz", **stats.private_pairing_cache())
    c.update("running", "private_cached_paired_comparisons", results=terminal,
        current_L1_scoring_pass=1, total_H4_TEST_scoring_passes=3, TEST_targets_read_this_condition=1000)
    paired = stats.paired_comparisons({"O": c.DATA / "phase4_1/private_pairing_cache.npz",
        "L0": c.DATA / "phase4_2/private_pairing_cache.npz", "L1": PHASE / "private_pairing_cache.npz"}, l1_evaluation_marker=marker)
    c.write(PHASE / "paired_comparisons.json", paired, exclusive=True)
    from l1_report import make_report
    elapsed = time.monotonic() - started
    summary = client.summary()
    reports = make_report(states, terminal, finals, paired, summary, elapsed, cfg)
    c.readme("complete", input_limit=cfg["prompts"]["input_limit"], forecast=c.read(PHASE / "pilot_forecast.json"))
    c.update("complete", "stage4_3_complete_waiting_for_continue", results=terminal, paired_comparisons=paired,
        final_models=finals, reports=reports, rounds_complete=60, rounds_total=60,
        elapsed_seconds=elapsed, transport=summary, LLM_calls_global=client.n_calls,
        LLM_calls_stage4_3=client.n_calls - 120, TEST_targets_read_this_condition=1000,
        current_L1_scoring_pass=1, total_H4_TEST_scoring_passes=3, next_stage_started=False,
        query_summary={str(seed): {"final_labels": len(state["labels"]), "revealed_by_queries": 1500 - state["final_fill"]["selected_n"],
            "uniform_fill": state["final_fill"]["selected_n"]} for seed, state in states.items()})
    return {"status": "complete", "elapsed_seconds": elapsed, "LLM_calls_global": client.n_calls}


def run():
    cfg = c.read(PHASE / "config.json")
    c.verify(cfg)
    with c.RunLock(), c.Awake():
        if (PHASE / "run_started.json").exists() or (PHASE / "evaluation_started.json").exists():
            raise ScientificStop("A prior finite L1 run/gate exists; no automatic replay")
        started, client = time.monotonic(), None
        c.write(PHASE / "process_identity.json", c.process_identity(), exclusive=True)
        c.write(PHASE / "run_started.json", {"utc": c.utc(), "config_sha256": c.sha(PHASE / "config.json")}, exclusive=True)
        try:
            bundle = data.load_bundle()
            states = {seed: c.read(state_path(seed)) for seed in (0, 1, 2)}
            for seed in states:
                if states[seed]["round_completed"] != 0 or data.initial_labels(seed, bundle=bundle) != states[seed]["labels"]:
                    raise ScientificStop("L1 must start at the frozen originalS0500 boundary")
            transport.INPUT_LIMIT = cfg["prompts"]["input_limit"]
            c.update("running", "first_L1_requests", error=None)
            with wire.StrictSchemaTransport(c.RUNTIME, c.DATA / "transport", c.OUT, cfg["protocol_files"], c.log,
                    max_calls=300, max_logical_requests=180, max_workers=4) as client:
                owned = client._canonical_owned_call_once
                if client.n_calls != 120 or len(client.responses) != 120 or client._e2.INPUT_LIMIT != cfg["prompts"]["input_limit"] or owned.__globals__["_e2"] is not client._e2:
                    raise ScientificStop("Frozen successful-call prefix or actual owned input guard differs")
                pilot = []
                for rnd in range(1, 21):
                    if rnd == 3:
                        pilot.extend(request_batch(client, bundle, states, rnd, cfg, seeds=(0, 1))["records"])
                        measured = forecast(pilot, states, time.monotonic() - started, cfg)
                        c.write(PHASE / "pilot_forecast.json", measured, exclusive=True)
                        c.readme("running" if measured["proceed"] else "stopped_forecast_over_10h", input_limit=cfg["prompts"]["input_limit"], forecast=measured)
                        c.update("running", "pilot_forecast", pilot_forecast=measured)
                        if not measured["proceed"]:
                            raise ScientificStop("Pilot forecast exceeds10 hours; awaiting user")
                        request_batch(client, bundle, states, rnd, cfg, seeds=(2,))
                    else:
                        batch = request_batch(client, bundle, states, rnd, cfg)
                        if rnd <= 2:
                            pilot.extend(batch["records"])
                    c.verify(cfg, full=False)
                result = finish(states, bundle, cfg, client, started)
                audit = verify()
                c.write(PHASE / "terminal_audit.json", audit, exclusive=True)
                c.write(PHASE / "run_stop.json", {**result, "utc": c.utc(), "next_stage_started": False}, exclusive=True)
                c.log("Stage4.3 complete; waiting for explicit continue; stage4.4 not started")
                return result
        except BaseException as error:
            reason = type(error).__name__ + ": " + str(error)
            expected = (ScientificStop, stats.ScientificStop, memory.MemoryBudgetExceeded, prompts.PromptBudgetExceeded)
            status = "stopped_requires_user" if isinstance(error, expected) else "failed"
            c.write(PHASE / "run_stop.json", {"status": status, "utc": c.utc(), "error": reason,
                "elapsed_seconds": time.monotonic() - started, "transport": client.summary() if client else None,
                "input_limit": cfg["prompts"]["input_limit"], "memory_limit_tokens": 6000,
                "L1_TEST_scoring_started": (PHASE / "evaluation_started.json").exists(), "next_stage_started": False})
            c.update(status, "stopped", error=reason, elapsed_seconds=time.monotonic() - started,
                transport=client.summary() if client else None, next_stage_started=False)
            c.readme(status, input_limit=cfg["prompts"]["input_limit"], forecast=c.read(PHASE / "pilot_forecast.json") if (PHASE / "pilot_forecast.json").exists() else None)
            c.text(PHASE / "traceback.txt", traceback.format_exc())
            c.log("STOP " + reason)
            return {"status": status, "error": reason}
        finally:
            c.write(PHASE / "process_finished.json", {"pid": os.getpid(), "utc": c.utc(), "elapsed_seconds": time.monotonic() - started})


def verify():
    cfg = c.read(PHASE / "config.json")
    audit = c.verify(cfg)
    ledger = transport._read_lines(c.OUT / "llm_calls.jsonl")
    reservations = [x for x in ledger if x["record_type"] == "reservation"]
    results = [x for x in ledger if x["record_type"] == "result"]
    successes = [x for x in results if x["status"] == "ok"]
    if len(reservations) > 300 or len({x["request_id"] for x in reservations}) > 180:
        raise RuntimeError("Finite global call/logical budget violated")
    if len({x["request_id"] for x in successes}) != len(successes):
        raise RuntimeError("A completed successful request was replayed")
    for item in successes:
        if transport.digest(transport.canonical(item["validated_response"]).encode()) != item["validated_response_sha256"]:
            raise RuntimeError("An old/new valid response hash changed")
    m = c.read(c.OUT / "metrics.json")
    total_rounds = 0
    for seed in (0, 1, 2):
        state = c.read(state_path(seed))
        total_rounds += state["round_completed"]
        if any(len(state[key]) != state["round_completed"] for key in
                ("history", "round_model_paths", "query_history", "memory_audits")):
            raise RuntimeError("Memory/model/round state cardinalities differ")
        seen, carry = set(cfg["data"]["perseed"][str(seed)]["S0_ids"]), 0
        for rnd, (path, query_record, hist, mem_audit) in enumerate(zip(state["round_model_paths"], state["query_history"], state["history"], state["memory_audits"]), 1):
            model = base._read_sealed(path)
            base._check_artifacts(model)
            lofo = base._read_sealed(Path(path).parent / "lofo" / "lofo.json")
            base._check_artifacts(lofo)
            if (lofo["identity"]["source_round_record_sha256"] != model["record_sha256"]
                    or lofo["raw_features"] != model["features"]
                    or lofo["baseline_validation_mae"] != model["validation_mae"]):
                raise RuntimeError("LOFO record no longer binds the exact baseline model")
            query = query_record["query_selection"]
            chosen = query["selected_ids"]
            if query["carry_in"] != carry or query["capacity"] != 75 + carry or len(chosen) != len(set(chosen)) or set(chosen) & seen:
                raise RuntimeError("Query carry/membership ledger changed")
            if query["selected_n"] != len(chosen) or query["carry_out"] != 75 + carry - len(chosen):
                raise RuntimeError("Query revealed cardinality changed")
            seen.update(chosen)
            carry = query["carry_out"]
            if set(model["identity"]["train_ids"]) != seen or model["round_index"] != rnd:
                raise RuntimeError("Training members differ from actual reveals")
            if base._matrix_hash([state["labels"][rid] for rid in model["identity"]["train_ids"]]) != model["identity"]["train_labels_sha256"]:
                raise RuntimeError("Previously revealed training labels changed")
            if hist["round_index"] != rnd or hist["features"] != model["features"]:
                raise RuntimeError("Memory feature definitions differ from trained round")
            if hist["contributions"] != lofo["contributions"] or hist["failures"] != lofo["failures"]:
                raise RuntimeError("Memory contributions/failures differ from cached LOFO evidence")
            request = c.read(PHASE / "requests" / (mem_audit["request_id"] + ".json"))
            byte_ledger = c.read(PHASE / "request_byte_ledger" / (mem_audit["request_id"] + ".json"))
            if byte_ledger != mem_audit or request["canonical_prompt_sha256"] != prompts._sha(request["prompt"]):
                raise RuntimeError("Request/byte-ledger identity changed")
            if mem_audit["memory_token_count"] > 6000 or mem_audit["format_repair_input_budget"]["input_token_upper_bound"] > cfg["prompts"]["input_limit"]:
                raise RuntimeError("An emitted request violated a frozen memory/input guard")
        if m["status"] == "complete":
            if state["round_completed"] != 20 or len(state["labels"]) != 2000:
                raise RuntimeError("Final L1 label/round coverage incomplete")
            fill = state["final_fill"]["selected_ids"]
            if len(fill) != len(set(fill)) or seen & set(fill) or seen | set(fill) != set(state["labels"]):
                raise RuntimeError("Final uniform fill changed query-selected labels")
            selected = c.read(PHASE / "selected_rounds.json")[seed]
            best = models.select_best([base._read_sealed(path) for path in state["round_model_paths"]])
            if selected["record_sha256"] != best["record_sha256"]:
                raise RuntimeError("Winner changed after fill")
            prefill = c.read(PHASE / "selection_pre_fill.json")
            if prefill["selected_rounds_sha256"] != c.sha(PHASE / "selected_rounds.json"):
                raise RuntimeError("Persisted pre-fill winner identity changed")
            before_fill = {**state, "labels": {rid: value for rid, value in state["labels"].items() if rid not in set(fill)},
                "final_fill": None, "status": "round_complete"}
            payload = (json.dumps(before_fill, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            if transport.digest(payload) != prefill["states_before_fill"][str(seed)]:
                raise RuntimeError("Reconstructed pre-fill state differs from the winner commitment")
    if m["status"] == "complete":
        if c.read(PHASE / "terminal_results.json") != m["results"] or c.read(PHASE / "paired_comparisons.json") != m["paired_comparisons"]:
            raise RuntimeError("Public/private terminal results differ")
        marker = PHASE / "evaluation_started.json"
        stats._marker(marker)
        if m["results"]["TEST_target_accessor_calls"] != 1 or m["current_L1_scoring_pass"] != 1 or m["total_H4_TEST_scoring_passes"] != 3:
            raise RuntimeError("L1 terminal TEST-pass boundary changed")
        marker_record = c.read(marker)
        if m["results"]["evaluation_marker"]["sha256"] != c.sha(marker):
            raise RuntimeError("L1 terminal result marker hash changed")
        if marker_record["committed_prediction_arms_sha256"] != c.sha(PHASE / "committed_prediction_arms.json"):
            raise RuntimeError("L1 private scored predictions changed")
        for final in m["final_models"]:
            sealed = base._read_sealed(final["metadata_path"])
            if sealed["record_sha256"] != final["record_sha256"]:
                raise RuntimeError("Final model metadata changed")
            base._check_artifacts(sealed)
        with (c.OUT / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 9000 or len([x for x in rows if x["condition"] == "L1"]) != 3000:
            raise RuntimeError("L1 public predictions do not preserve all O/L0 rows plus3000L1 rows")
        arms = c.read(PHASE / "committed_prediction_arms.json")
        stats._normalise(arms, cfg["data"]["test_ids"])
        expected = {(arm["arm_id"], str(arm["seed"]), row["id"]): float(row["prediction_eV_per_atom"]).hex()
            for arm in arms for row in arm["prediction_rows"]}
        actual = {(row["arm_id"], row["seed"], row["id"]): float(row["prediction_eV_per_atom"]).hex()
            for row in rows if row["condition"] == "L1"}
        if expected != actual or any(row["split"] != "TEST" for row in rows if row["condition"] == "L1"):
            raise RuntimeError("Published L1 rows differ from the exact scored prediction rows")
        if c.read(marker)["public_predictions_sha256"] != c.sha(c.OUT / "predictions.csv"):
            raise RuntimeError("Scored public predictions changed")
    audit.update(status=m["status"], rounds_complete=total_rounds, CLI_calls_global=len(reservations),
        CLI_results=len(results), current_L1_scoring_pass=m.get("current_L1_scoring_pass", 0), no4_4=m.get("next_stage_started") is False)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "verify"])
    action = parser.parse_args().action
    result = prepare() if action == "prepare" else run() if action == "run" else verify()
    if sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
