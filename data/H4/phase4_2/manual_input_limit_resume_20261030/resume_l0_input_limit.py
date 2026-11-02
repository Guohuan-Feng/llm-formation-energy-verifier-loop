"""One authorized H4.2 continuation from round7 after input-limit amendment.

The old scientific sources, requests, successful models and run records are
immutable. The sole runtime protocol override is the separately authorized
34000 complete-input bound; prompts/schema/feature-source limits stay exact.
Import and verify never call an LLM, fit, execute a feature or score TEST.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SRC = HERE.parents[1] / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
import argparse
import hashlib
import json
import os
import re
import time
import traceback

import run_h4_l0 as original

c = original.c
data = original.data
models = original.models
prompts = original.prompts
transport = original.transport
MANIFEST = HERE / "manifest.json"
PROTOCOL = "H4-stage4.2-input-limit-manual-resume-v1"
INPUT_LIMIT = 34000
CONFIG_SHA256 = "bfff6e10ecf9b2d51c76fc20c59b7590e965a7c6988258c317ca5c0ef2fadb22"
AMENDMENT_SHA256 = "a2d67d707dbd7215e0fd3bfb14377061297c48d64fc2e72e6664a154fdb5f11d"
GLOBAL_CONTEXT = Path(r"C:\Users\28908\.codex\AGENTS.md").resolve()
NEW_REQUEST = re.compile(r"L0_seed[012]_round(?:0[789]|1[0-9]|20)\Z")


def read_path(path):
    value = Path(path).resolve(strict=True)
    if "onedrive" in str(value).casefold():
        raise ValueError("OneDrive input is forbidden")
    if not value.is_relative_to(ROOT.resolve()) and value != GLOBAL_CONTEXT:
        raise ValueError("Read outside research root and the explicit global storage instructions")
    return value


def sha(path):
    digest = hashlib.sha256()
    with read_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(read_path(path).read_text(encoding="utf-8-sig"))


def committed(path):
    raw = read_path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("Incomplete durable JSONL tail")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def differences(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        answer = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                answer.append(path + (key,))
            else:
                answer.extend(differences(left[key], right[key], path + (key,)))
        return answer
    return [] if type(left) is type(right) and left == right else [path]


def verify_prefix(record):
    path = read_path(record["path"])
    size = record["size"]
    if type(size) is not int or size < 0 or path.stat().st_size < size:
        raise ValueError("Preserved ledger/prediction prefix was truncated")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            block = stream.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Preserved prefix ended early")
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != record["sha256"]:
        raise ValueError("Preserved ledger/prediction prefix changed")


def snapshot_state(seed, ref, cfg):
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Original state snapshot changed")
    if Path(ref["live_path"]).resolve() != original.state_path(seed).resolve():
        raise ValueError("State snapshot points to a different live seed")
    state = read_json(ref["path"])
    if (state["seed"] != seed or state["round_completed"] != 6
            or len(state["labels"]) != 950 or state["carry"] != 0
            or len(state["round_model_paths"]) != 6 or len(state["query_history"]) != 6):
        raise ValueError("Original snapshot is not the authorized round6/950 boundary")
    initial = set(cfg["data"]["perseed"][str(seed)]["S0_ids"])
    seen, carry = set(initial), 0
    if len(initial) != 500:
        raise ValueError("Original S0 membership changed")
    for index, (record, path) in enumerate(zip(state["query_history"], state["round_model_paths"]), 1):
        selection = record["query_selection"]
        chosen = selection["selected_ids"]
        if (record["round_index"] != index or selection["carry_in"] != carry
                or selection["capacity"] != 75 + carry or selection["selected_n"] != len(chosen)
                or len(chosen) != len(set(chosen)) or set(chosen) & seen
                or selection["carry_out"] != 75 + carry - len(chosen)):
            raise ValueError("Original query/reveal/carry history changed")
        seen.update(chosen)
        carry = selection["carry_out"]
        model = original.model_base._read_sealed(path)
        original.model_base._check_artifacts(model)
        ids = model["identity"]["train_ids"]
        if (model["seed"] != seed or model["round_index"] != index or set(ids) != seen
                or original.model_base._matrix_hash([state["labels"][rid] for rid in ids])
                   != model["identity"]["train_labels_sha256"]):
            raise ValueError("Original round model/label binding differs from saved state")
    if seen != set(state["labels"]) or carry != 0:
        raise ValueError("Original state labels differ from six-round reveal history")
    return state


def verify_live_state(seed, baseline, cfg):
    state = c.read(original.state_path(seed))
    if state["seed"] != seed or not 6 <= state["round_completed"] <= 20:
        raise ValueError("Live state moved outside the authorized continuation")
    if state["round_model_paths"][:6] != baseline["round_model_paths"] or state["query_history"][:6] != baseline["query_history"]:
        raise ValueError("Old six-round model/history prefix changed")
    if any(state["labels"].get(rid) != value for rid, value in baseline["labels"].items()):
        raise ValueError("An old revealed member or target changed")
    if (not set(state["labels"]) <= set(cfg["data"]["pool_ids"])
            or not 950 <= len(state["labels"]) <= 2000):
        raise ValueError("Current state labels exceed the POOL-only budget")
    return state


def verify_manifest(manifest_path=MANIFEST, *, require_driver=True):
    manifest = read_json(manifest_path)
    if manifest["protocol"] != PROTOCOL or Path(manifest["manual_directory"]).resolve() != HERE:
        raise ValueError("Manual recovery manifest identity differs")
    if (manifest["authorization"]["input_limit"] != INPUT_LIMIT
            or not manifest["authorization"].get("user_text")
            or not manifest["authorization"].get("recorded_utc")
            or [manifest["start_round"], manifest["end_round"], manifest["max_calls"],
                manifest["max_logical_requests"], manifest["max_workers"]] != [7, 20, 300, 180, 4]
            or manifest["expected_boundary"] != {"round_completed": 6, "labels_per_seed": 950,
                                                   "carry": 0, "global_calls": 78}):
        raise ValueError("Manual authorization or finite boundary differs")
    if require_driver:
        ref = manifest["driver"]
        if Path(ref["path"]).resolve() != Path(__file__).resolve() or sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Manual driver hash changed")
    for key, expected in (("config", CONFIG_SHA256), ("amendment", AMENDMENT_SHA256)):
        ref = manifest[key]
        if ref["sha256"] != expected or sha(ref["path"]) != expected:
            raise ValueError("Authorized input-limit amendment/config hash differs")
    for key in ("prior_config", "projection"):
        if sha(manifest[key]["path"]) != manifest[key]["sha256"]:
            raise ValueError("Saved amendment baseline/projection changed")
    cfg = read_json(manifest["config"]["path"])
    previous_cfg = read_json(manifest["prior_config"]["path"])
    if (differences(previous_cfg, cfg) != [("prompts", "input_limit")]
            or previous_cfg["prompts"]["input_limit"] != 20000 or cfg["prompts"]["input_limit"] != INPUT_LIMIT):
        raise ValueError("The config amendment is not exactly input_limit20000-to34000")
    if manifest["frozen_sources"] != cfg["source_hashes"]:
        raise ValueError("Frozen scientific source list changed")
    for section in ("frozen_sources", "preserved_files", "global_context_hashes"):
        for path, expected in manifest[section].items():
            if sha(path) != expected:
                raise ValueError("Preserved source/cache/context bytes changed: " + Path(path).name)
    if set(Path(path).resolve() for path in manifest["global_context_hashes"]) != {GLOBAL_CONTEXT}:
        raise ValueError("Unexpected external global-context input")
    for directory, expected in manifest["preserved_trees"].items():
        if c.tree_hashes(read_path(directory)) != expected:
            raise ValueError("An immutable initial/old-round tree changed")
    for ref in manifest["preserved_prefixes"].values():
        verify_prefix(ref)
    snapshots = {seed: snapshot_state(seed, manifest["state_snapshots"][str(seed)], cfg)
                 for seed in (0, 1, 2)}
    live = {seed: verify_live_state(seed, snapshots[seed], cfg) for seed in (0, 1, 2)}
    ledger = committed(c.OUT / "llm_calls.jsonl")
    reservations = [row for row in ledger if row.get("record_type") == "reservation"]
    results = [row for row in ledger if row.get("record_type") == "result"]
    if len(reservations) > 300 or len({row["request_id"] for row in reservations}) > 180:
        raise ValueError("Shared finite H4 budget was exceeded")
    before = [row for row in reservations if row["call_index"] <= 78]
    old_results = [row for row in results if row["call_index"] <= 78]
    if (len(before) != 78 or len(old_results) != 78
            or {row["call_index"] for row in old_results} != set(range(1, 79))
            or any(row.get("status") != "ok" for row in old_results)):
        raise ValueError("Old78 completed successful calls are not intact")
    for row in reservations:
        if row["call_index"] > 78 and not NEW_REQUEST.fullmatch(row["request_id"]):
            raise ValueError("A request outside the authorized unfinished rounds was dispatched")
    return manifest, cfg, snapshots, live


def verify(manifest_path=MANIFEST):
    manifest, cfg, snapshots, states = verify_manifest(manifest_path)
    audit = original.verify_terminal()
    return {**audit, "manual_manifest_sha256": sha(manifest_path), "input_limit": INPUT_LIMIT,
            "old_rounds_preserved": 18, "old_initial_models_preserved": 3,
            "manual_source_sha256": sha(__file__),
            "current_rounds": {str(seed): state["round_completed"] for seed, state in states.items()},
            "verify_only_CPU_readonly": True}


def install_limit():
    if prompts.INPUT_LIMIT not in (20000, INPUT_LIMIT) or transport.INPUT_LIMIT not in (20000, INPUT_LIMIT):
        raise ValueError("Unexpected existing transport/prompt budget override")
    prompts.INPUT_LIMIT = INPUT_LIMIT
    transport.INPUT_LIMIT = INPUT_LIMIT


def assert_owned_limit(client):
    call = client._canonical_owned_call_once
    if (client._e2.INPUT_LIMIT != INPUT_LIMIT or call.__globals__["_e2"] is not client._e2
            or call.__globals__["_base"]._e2 is not client._e2
            or client.max_calls != 300 or client.max_logical_requests != 180):
        raise RuntimeError("The actual pinned owned CLI does not enforce the amended finite budget")


def hydrate_registry(bundle, states):
    for seed in (0, 1, 2):
        data.validate_labels(states[seed]["labels"], seed, bundle=bundle, require_n=950)
        existing = data._REVEALED.get(seed, {})
        if existing and existing != states[seed]["labels"]:
            raise ValueError("In-memory reveal registry differs from the saved boundary")
        data._REVEALED[seed] = dict(states[seed]["labels"])


def readme(status, forecast=None):
    c.readme(status, forecast)
    path = c.safe(c.OUT / "README.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] += " Manual input-limit recovery:34000; original prompt/schema/source20000-char limit unchanged; run elapsed refers only to this manual process."
    if len(lines) > 20:
        raise RuntimeError("Manual README exceeds20 lines")
    c.text(path, "\n".join(lines) + "\n")


def finish(states, bundle, cfg, client, started):
    """Frozen round20 coordination tail; numerical/model helpers are unchanged."""
    selected = []
    for seed in (0, 1, 2):
        rounds = [original.model_base._read_sealed(path) for path in states[seed]["round_model_paths"]]
        selected.append(models.select_best(rounds))
    c.write(c.PHASE / "selected_rounds.json", selected, exclusive=True)
    c.write(c.PHASE / "selection_pre_fill.json", {"utc": c.utc(),
        "selected_rounds_sha256": c.sha(c.PHASE / "selected_rounds.json"),
        "states_before_fill": {str(seed): c.sha(original.state_path(seed)) for seed in (0, 1, 2)},
        "fill_labels_not_yet_revealed": True}, exclusive=True)
    for seed in (0, 1, 2):
        state = states[seed]
        fill = original.queries.final_fill_only_no_reveal(arm="L0", seed=seed,
            labeled_ids=sorted(state["labels"]), bundle=bundle, total_new_revealed=len(state["labels"]) - 500)
        state["labels"].update(data.reveal(fill["selected_ids"], seed, bundle=bundle) if fill["selected_ids"] else {})
        data.validate_labels(state["labels"], seed, bundle=bundle, require_n=2000)
        state.update(final_fill=fill, status="final_labels_ready")
        c.write(original.state_path(seed), state)
    finals = []
    for seed, best in enumerate(selected):
        finals.append(models.finalize(bundle, seed, best, states[seed]["labels"], bundle["test_ids"],
            c.PHASE / "final" / ("L0_seed" + str(seed)), progress=original.progress))
        c.update("running", "winner_TEST_predictions", final_prediction_arms_complete=len(finals))
    arms = original.prediction_arms(finals, bundle)
    original.stats._normalise(arms, bundle["test_ids"])
    c.write(c.PHASE / "committed_prediction_arms.json", arms, exclusive=True)
    original.export_predictions(arms)
    c.verify(cfg)
    marker = c.PHASE / "evaluation_started.json"
    c.write(marker, {"stage": "4.2", "condition": "L0", "scoring_pass": 1,
        "current_L0_scoring_pass": 1, "prior_H4_TEST_scoring_passes": 1,
        "TEST_targets_previously_read_for_this_condition": False,
        "prior_stage4_1_marker_sha256": c.sha(c.DATA / "phase4_1/evaluation_started.json"),
        "utc": c.utc(), "config_sha256": c.sha(c.PHASE / "config.json"),
        "committed_prediction_arms_sha256": c.sha(c.PHASE / "committed_prediction_arms.json"),
        "public_predictions_sha256": c.sha(c.OUT / "predictions.csv")}, exclusive=True)
    c.update("running", "terminal_scoring")
    terminal = original.stats.evaluate(arms, bundle, evaluation_marker=marker)
    c.write(c.PHASE / "terminal_results.json", terminal, exclusive=True)
    original.model_base._write_npz(c.PHASE / "private_pairing_cache.npz", **original.stats.private_pairing_cache())
    elapsed = time.monotonic() - started
    summary = client.summary()
    reports = original.curve_and_report(states, terminal, finals, summary, elapsed)
    readme("complete", c.read(c.PHASE / "pilot_forecast.json"))
    c.update("complete", "stage4_2_complete_waiting_for_continue", results=terminal, final_models=finals,
        rounds_complete=60, rounds_total=60, reports=reports, manual_elapsed_seconds=elapsed,
        elapsed_seconds=None, historical_total_active_seconds=None, transport=summary,
        LLM_calls_global=client.n_calls, LLM_calls_stage4_2=client.n_calls - 60,
        query_summary={str(seed): {"final_labels": len(state["labels"]),
            "revealed_by_queries": 1500 - state["final_fill"]["selected_n"],
            "uniform_fill": state["final_fill"]["selected_n"]} for seed, state in states.items()},
        TEST_targets_read_this_condition=1000, current_L0_scoring_pass=1,
        total_H4_TEST_scoring_passes=2, next_stage_started=False)
    return {"status": "complete", "manual_elapsed_seconds": elapsed, "LLM_calls_global": client.n_calls}


def run(manifest_path=MANIFEST):
    manifest, cfg, snapshots, states = verify_manifest(manifest_path)
    c.verify(cfg)
    with c.RunLock(), c.Awake():
        if (HERE / "run_started.json").exists() or (c.PHASE / "evaluation_started.json").exists():
            raise original.ScientificStop("Manual run/TEST gate already exists; no automatic replay")
        if any(c.digest(states[seed]) != c.digest(snapshots[seed]) for seed in (0, 1, 2)):
            raise original.ScientificStop("Current states differ from the exact authorized round6 boundary")
        started = time.monotonic()
        identity = {**c.process_identity(), "driver": str(Path(__file__).resolve()),
                    "arguments": sys.argv[1:], "executable": sys.executable,
                    "manual_manifest_sha256": sha(manifest_path)}
        c.write(HERE / "process_identity.json", identity, exclusive=True)
        c.write(HERE / "run_started.json", {"utc": c.utc(), "identity": identity,
            "manifest_sha256": sha(manifest_path), "start_round": 7, "end_round": 20}, exclusive=True)
        pointer = {"directory": str(HERE), "driver": str(Path(__file__).resolve()),
                   "manifest": str(Path(manifest_path).resolve()), "identity": identity}
        client = None
        try:
            install_limit()
            bundle = data.load_bundle()
            hydrate_registry(bundle, states)
            c.update("running", "manual_resume_from_round7", error=None, current_manual_run=pointer,
                manual_elapsed_seconds=0., elapsed_seconds=None, historical_total_active_seconds=None,
                historical_recorded_run_elapsed_seconds=manifest["historical_run_elapsed_seconds"],
                LLM_calls_global=78, LLM_calls_stage4_2=18, TEST_targets_read_this_condition=0,
                current_L0_scoring_pass=0, next_stage_started=False)
            readme("running", c.read(c.PHASE / "pilot_forecast.json"))
            c.log("User-authorized manual L0 continuation rounds7..20; input_limit34000 only; old78 successful calls retained")
            with original.wire.StrictSchemaTransport(c.RUNTIME, c.DATA / "transport", c.OUT,
                    cfg["protocol_files"], c.log, max_calls=300, max_logical_requests=180, max_workers=4) as client:
                assert_owned_limit(client)
                if client.n_calls != 78 or len(client.responses) != 78:
                    raise RuntimeError("Shared transport did not restore exactly78 successful calls")
                for round_index in range(7, 21):
                    original.request_round(client, bundle, states, round_index, cfg)
                    c.verify(cfg, full=False)
                    verify_manifest(manifest_path)
                    c.update("running", "manual_loop_round_validation", manual_elapsed_seconds=time.monotonic() - started)
                result = finish(states, bundle, cfg, client, started)
                audit = verify(manifest_path)
                c.write(HERE / "terminal_audit.json", audit, exclusive=True)
                c.write(HERE / "run_stop.json", {**result, "utc": c.utc(), "identity": identity,
                    "next_stage_started": False}, exclusive=True)
                c.log("Manual L0 complete; old source/cache/prefix auditPASS; waiting for continue; no4.3")
                return result
        except BaseException as exc:
            reason = type(exc).__name__ + ": " + str(exc)
            status = "stopped_requires_user" if isinstance(exc, (original.ScientificStop, prompts.PromptBudgetExceeded)) else "failed"
            c.write(HERE / "run_stop.json", {"status": status, "utc": c.utc(), "error": reason,
                "identity": identity, "manual_elapsed_seconds": time.monotonic() - started,
                "input_limit": INPUT_LIMIT, "transport": client.summary() if client else None})
            c.update(status, "manual_stopped", error=reason, manual_elapsed_seconds=time.monotonic() - started,
                elapsed_seconds=None, historical_total_active_seconds=None,
                transport=client.summary() if client else None)
            readme(status, c.read(c.PHASE / "pilot_forecast.json"))
            c.text(HERE / "traceback.txt", traceback.format_exc())
            c.log("STOP manual L0 " + reason)
            return {"status": status, "error": reason}
        finally:
            c.write(HERE / "process_finished.json", {"identity": identity, "utc": c.utc(),
                "manual_elapsed_seconds": time.monotonic() - started})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["verify", "run"])
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    result = verify(args.manifest) if args.action == "verify" else run(args.manifest)
    if sys.stdout is not None:
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
