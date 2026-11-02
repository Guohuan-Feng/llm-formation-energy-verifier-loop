"""Step2 local measurement only, after the revised policy has been frozen.

Read existing L0 cached metadata/LOFO histories and already revealed POOL labels.
Build 3 empty baseline +57 nonempty memory/request views without dispatch,
feature execution, fitting, prediction, scoring, new reveals or TEST target reads.
Return an audit object; only root may persist it and decide whether to launch.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SRC = Path(__file__).resolve().parent
REVISION = SRC.parent
PHASE = REVISION.parent
DATA = PHASE.parent
ROOT = DATA.parents[2]
for directory in (DATA / "src", PHASE / "src", SRC):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
import h4_models as old_models
import l1_data as label_free_data
import l1_counter as counter
import l1_memory_v2 as memory
import l1_prompts_v2 as prompts

COUNTER_RECORD_SHA256 = "f374eab336ae08f9bcb8a58cf0f8346c596ff8eb99e1b37357840f7393e6e99f"
NOTES_PROBE = "".join(chr(0x4e00 + index) for index in range(300))
WHOLE_INPUT_LIMIT = 189000


def _checked(path):
    value = Path(path).resolve(strict=True)
    if "onedrive" in str(value).casefold() or not value.is_relative_to(ROOT.resolve()):
        raise ValueError("Measurement reads must stay under the non-OneDrive research root")
    return value


def _sha(path):
    return hashlib.sha256(_checked(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(_checked(path).read_text(encoding="utf-8-sig"))


def _freeze_gate(config_path, frozen_config_path):
    cfg_path = _checked(config_path)
    gate_path = _checked(frozen_config_path)
    if cfg_path != _checked(PHASE / "config.json") or gate_path != _checked(PHASE / "frozen_config.json"):
        raise ValueError("Only the canonical revised phase4_3 config/freeze files authorize Step2 measurement")
    cfg, gate = _read(cfg_path), _read(gate_path)
    if gate.get("sha256") != _sha(cfg_path):
        raise ValueError("Revised config/frozen hash differs; do not measure or launch")
    if (cfg.get("memory", {}).get("limit_tokens") != 20000 or memory.LIMIT != 20000
            or cfg.get("prompts", {}).get("input_limit") != WHOLE_INPUT_LIMIT):
        raise ValueError("Step2 requires the explicitly frozen metadata20000/whole-input189000 policy")
    if (cfg.get("token_counter", {}).get("token_counter_sha256") != COUNTER_RECORD_SHA256
            or gate.get("token_counter_sha256") != COUNTER_RECORD_SHA256
            or _sha(PHASE / "token_counter.json") != COUNTER_RECORD_SHA256):
        raise ValueError("The original tokenization counter record must remain unchanged")
    expected = cfg.get("source_hashes", {})
    for path in (Path(memory.__file__), Path(prompts.__file__), Path(__file__)):
        source = _checked(path)
        if expected.get(str(source)) != _sha(source):
            raise ValueError("Revised measurement/memory/prompt source is not bound by the frozen config")
    for raw_path, expected_hash in cfg.get("global_context_hashes", {}).items():
        path = Path(raw_path).resolve(strict=True)
        if ("onedrive" in str(path).casefold() or path.name not in {"AGENTS.md", "AGENTS.override.md"}
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash):
            raise ValueError("Frozen global model-instruction bytes changed before local measurement")
    verified = counter.verify()
    if verified["token_counter_sha256"] != COUNTER_RECORD_SHA256:
        raise ValueError("Pinned local counter verification changed")
    return cfg, verified


def measure_all_l0(*, config_path=PHASE / "config.json", frozen_config_path=PHASE / "frozen_config.json",
        notes_probe=NOTES_PROBE, bundle=None):
    """Pure Step2: return measurement; no automatic write, freeze, run or repair."""
    cfg, counter_audit = _freeze_gate(config_path, frozen_config_path)
    if type(notes_probe) is not str or notes_probe != NOTES_PROBE:
        raise ValueError("Use the preselected deterministic300-CJK probe unchanged; no postmeasurement probe selection")
    probe_tokens = counter.count_text(notes_probe)
    if len(notes_probe) != 300 or len(notes_probe.encode("utf-8")) != 900 or probe_tokens != 481:
        raise ValueError("The frozen local notes-probe identity changed")
    b = label_free_data.load_bundle() if bundle is None else bundle
    if b.get("evaluation_truth_loaded", False):
        raise ValueError("Only a label-free bundle may enter local measurement")
    pool = set(b["pool_ids"])
    if pool & set(b["test_ids"]): raise ValueError("POOL/TEST identity overlap")
    index_path = PHASE / "projection_data/projection_index.json"
    index = _read(index_path)
    if (index.get("status") != "complete" or index.get("rounds") != 60
            or index.get("new_LLM_calls") != 0 or index.get("generated_feature_executions") != 0
            or index.get("TEST_targets_or_feedback_used") is not False):
        raise ValueError("Only the completed existing L0 CPU-LOFO projection is allowed")
    records = {(x["seed"], x["round_index"]): x for x in index["records"]}
    if len(records) != 60 or set(records) != {(s, r) for s in (0, 1, 2) for r in range(1, 21)}:
        raise ValueError("Existing L0 history coverage must be exactly3x20")
    hashes = {str(_checked(index_path)): _sha(index_path),
        str(_checked(config_path)): _sha(config_path), str(_checked(frozen_config_path)): _sha(frozen_config_path)}
    rows = []
    for seed in (0, 1, 2):
        state_path = DATA / f"phase4_2/states/seed{seed}.json"
        state = _read(state_path)
        hashes[str(_checked(state_path))] = _sha(state_path)
        # Only members already revealed by the completed old L0 run are read.
        final_labels = state["labels"]
        if not isinstance(final_labels, dict) or not set(final_labels) <= pool:
            raise ValueError("Old L0 label state must contain only revealed POOL members")
        seen = set(b["perseed"][seed]["S0_ids"])
        if len(seen) != 500 or not seen <= set(final_labels): raise ValueError("Exact S0500 not preserved")
        history = []
        for rnd in range(1, 21):
            if rnd == 1:
                feedback_path = DATA / f"phase4_2/initial/L0_seed{seed}/initial_state.json"
            else:
                feedback_path = DATA / f"phase4_2/rounds/L0_seed{seed}/round{rnd-1:02d}/round.json"
            feedback = old_models._read_sealed(_checked(feedback_path))
            hashes[str(_checked(feedback_path))] = _sha(feedback_path)
            if set(feedback["identity"]["train_ids"]) != seen:
                raise ValueError("Before-request feedback membership differs from actual old query reveals")
            labels = {rid: final_labels[rid] for rid in sorted(seen)}
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in labels.values()):
                raise ValueError("Only finite already-revealed POOL labels are allowed")
            mem = memory.build_memory(history, measurement=True, private_ids=b["ids"])
            current = feedback.get("active_features", [])
            request = prompts.build_request(seed, rnd, current, feedback["validation_mae"], feedback["worst20"],
                labels, mem, input_limit=WHOLE_INPUT_LIMIT, carry=0 if rnd == 1 else carry,
                bundle=b, measurement=True)
            ordinary, repair = request["input_budget"], request["format_repair_input_budget"]
            rows.append({"request_id": request["request_id"], "seed": seed, "round_index": rnd,
                "completed_history_rounds": rnd - 1, "empty_initial_baseline": rnd == 1,
                "memory_tokens": mem["token_count"], "memory_JSON_UTF8_bytes": mem["UTF8_bytes"],
                "memory_trimmed": mem["trimmed"], "memory_overflow_20000": mem["overflow"],
                "memory_payload_sha256": mem["payload_sha256"], "metadata_archive_sha256": mem["archive_sha256"],
                "memory_detailed_round_records": request["memory_detailed_round_records"],
                "memory_note_chars": request["memory_note_chars"],
                "memory_notes_chars_by_round": request["memory_notes_chars_by_round"],
                "component_breakdown": mem["component_breakdown"],
                "actual_canonical_prompt_UTF8_bytes": ordinary["canonical_prompt_bytes"],
                "actual_wire_prompt_UTF8_bytes": ordinary["wire_prompt_bytes"],
                "actual_repair_wire_prompt_UTF8_bytes": repair["wire_prompt_bytes"],
                "known_fixed_context_UTF8_bytes": ordinary["known_fixed_context_bytes"],
                "ordinary_conservative_input_bound": ordinary["input_token_upper_bound"],
                "repair_conservative_input_bound": repair["input_token_upper_bound"],
                "ordinary_within189000": ordinary["within_budget"], "repair_within189000": repair["within_budget"],
                "canonical_prompt_sha256": request["canonical_prompt_sha256"],
                "schema_sha256": request["schema_sha256"], "wire_schema_sha256": request["wire_schema_sha256"],
                "current_active_feature_count": request["current_feature_count"],
                "current_complete_source_UTF8_bytes": request["current_complete_source_bytes"],
                "already_revealed_POOL_labels": len(labels)})
            # Advance only through existing L0 query membership; reveal no new label.
            query_path = DATA / f"phase4_2/query_records/L0_seed{seed}_round{rnd:02d}.json"
            query = _read(query_path)
            hashes[str(_checked(query_path))] = _sha(query_path)
            selected = query["query_selection"]["selected_ids"]
            if len(selected) != len(set(selected)) or set(selected) & seen or not set(selected) <= set(final_labels):
                raise ValueError("Old L0 query selection membership changed")
            seen.update(selected)
            carry = query["query_selection"]["carry_out"]
            source = records[(seed, rnd)]
            if source.get("projection_only") is not True or source.get("source_condition") != "L0":
                raise ValueError("Only old L0 surrogate histories may be measured")
            historical = deepcopy(source["history"])
            historical["notes"] = notes_probe
            history.append(historical)
    if len(rows) != 60 or sum(not row["empty_initial_baseline"] for row in rows) != 57:
        raise AssertionError("Step2 must cover exactly57nonempty +3empty initial views")
    passed = all(not row["memory_overflow_20000"] and row["ordinary_within189000"] and row["repair_within189000"] for row in rows)
    return {"status": "PASS" if passed else "FAIL", "stage": "4.3", "revision": memory.VERSION,
        "measurement_only": True, "new_scientific_calls": 0, "new_feature_executions": 0,
        "new_fits_predictions_or_scoring": 0, "new_TEST_HOLDOUT_target_access": 0, "new_POOL_reveals": 0,
        "frozen_config_sha256": _sha(config_path), "counter": counter_audit,
        "external_memory_policy_tokens": 20000, "whole_request_limit": WHOLE_INPUT_LIMIT,
        "counter_record_legacy6000_is_not_policy": True,
        "notes_probe": {"construction": "chr(0x4e00+i) for i=0..299", "Unicode_characters": 300,
            "UTF8_bytes": 900, "local_counter_tokens": 481, "sha256": _sha_text(notes_probe),
            "synthetic_scenario_not_observed_L1_notes": True, "universal_maximum_claimed": False},
        "actual_full_payload_scope": "Complete exact request built from pre-round L0 active code, feedback, revealed POOL labels, catalogue and revised memory; no old-base extrapolation",
        "whole_request_counts_are_UTF8_bounds_not_model_token_measurements": True,
        "nonempty_views": 57, "empty_initial_views": 3,
        "maximum_memory_tokens": max(x["memory_tokens"] for x in rows),
        "maximum_memory_JSON_UTF8_bytes": max(x["memory_JSON_UTF8_bytes"] for x in rows),
        "maximum_actual_canonical_prompt_UTF8_bytes": max(x["actual_canonical_prompt_UTF8_bytes"] for x in rows),
        "maximum_repair_conservative_input_bound": max(x["repair_conservative_input_bound"] for x in rows),
        "memory_overflow_views": sum(x["memory_overflow_20000"] for x in rows),
        "whole_input_overflow_views": sum(not x["repair_within189000"] for x in rows),
        "measurement_does_not_authorize_run": True, "rows": rows, "source_hashes": hashes}


def _sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
