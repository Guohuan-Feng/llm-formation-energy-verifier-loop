"""Explicit ROOT-only reversible publication and hash receipts; no science imports."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import json
import os
import stat
import uuid

MANUAL = Path(__file__).resolve().parent.parent
PHASE = MANUAL.parent
DATA = PHASE.parent
ROOT = DATA.parents[2]
OUT = ROOT / "H4_feature_loop"
MD = PHASE / "memory_content_amendment_L1.md"
OLD_MANIFEST = PHASE / "prior_stages_preservation.json"
CURRENT = PHASE / "prior_stages_preservation_current.json"
SNAPSHOT = MANUAL / "history_snapshot"
SNAP_MANIFEST = MANUAL / "history_snapshot_manifest.json"
PRE_SCORE = MANUAL / "pre_score_public_update.json"
NAMES = ("config.json", "predictions.csv", "llm_calls.jsonl", "metrics.json", "run.log", "README.md")
ADDED = {"current_effective_public_hashes", "history_snapshot_manifest_hash", "amendment_original_sha256",
    "amendment_current_sha256", "original_preservation_manifest_sha256", "public_update_phase"}


def safe(path):
    path = Path(path).resolve()
    if "onedrive" in str(path).casefold() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Publication paths must remain in the non-OneDrive research root")
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding="utf-8-sig"))


def put(path, value, *, raw=False, exclusive=False):
    destination = safe(path)
    safe(destination.parent).mkdir(parents=True, exist_ok=True)
    payload = value if raw else (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if exclusive:
        with safe(destination).open("xb") as stream: stream.write(payload)
    else:
        temporary = safe(MANUAL / "atomic" / (uuid.uuid4().hex + ".tmp"))
        safe(temporary.parent).mkdir(parents=True, exist_ok=True)
        with safe(temporary).open("xb") as stream: stream.write(payload)
        os.replace(safe(temporary), safe(destination))


def utc():
    return datetime.now(timezone.utc).isoformat()


def hashes():
    if {p.name for p in OUT.iterdir()} != set(NAMES): raise ValueError("Public directory must contain exactly six files")
    return {name: {"path": str(safe(OUT / name)), "sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size} for name in NAMES}


def typed(value):
    if type(value) is float: return {"__float_hex__": value.hex()}
    if isinstance(value, dict): return {k: typed(v) for k, v in value.items()}
    if isinstance(value, list): return [typed(v) for v in value]
    return {"__type__": type(value).__name__, "value": value}


def proof(label, left, right):
    a = json.dumps(typed(left), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    b = json.dumps(typed(right), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if a != b: raise ValueError("Object/float.hex comparison failed: " + label)
    return {"object": label, "method": "Every JSON node/type compared recursively; every float uses float.hex()",
        "typed_tree_sha256": hashlib.sha256(a.encode("utf-8")).hexdigest(), "status": "PASS"}


def predictions_check():
    old = (PHASE / "stage4_2_public_snapshot/predictions.csv").read_bytes()
    current = (OUT / "predictions.csv").read_bytes()
    old_rows = list(csv.DictReader(io.StringIO(old.decode("utf-8-sig"))))
    rows = list(csv.DictReader(io.StringIO(current.decode("utf-8-sig"))))
    if len(old) != 298912 or not current.startswith(old) or len(old_rows) != 6000 or len(rows) != 9000:
        raise ValueError("Prediction old6000-row/298912-byte prefix or current9000-row boundary differs")
    return {"old_bytes": len(old), "old_rows": 6000, "current_rows": 9000, "old_prefix_byte_identical": True,
        "old_full_sha256": hashlib.sha256(old).hexdigest(), "current_full_sha256": hashlib.sha256(current).hexdigest()}


def step1_gate():
    summary = read(MANUAL / "step1_read_summary.json")
    verified = read(MANUAL / "step1_original_verify.log")
    if (summary.get("status") != "PASS" or summary.get("unique_manifest_paths") != 1300
            or summary.get("read_failures") or summary.get("immutable_or_prefix_mismatches")):
        raise ValueError("Step1 full1300-path read audit did not pass")
    wanted = {"scientific_invariants": "PASS", "O_preserved": True, "L0_preserved": True,
        "void_archive": "PASS", "rounds_complete": 60, "current_L1_scoring_pass": 0}
    if any(type(verified.get(k)) is not type(v) or verified.get(k) != v for k, v in wanted.items()):
        raise ValueError("Step1 original readonly verification fields did not pass")
    if (PHASE / "evaluation_started.json").exists() or (PHASE / "terminal_results.json").exists():
        raise ValueError("prepare_public must precede the single L1 terminal scoring gate")
    return {"read_summary_sha256": sha(MANUAL / "step1_read_summary.json"),
        "original_verify_log_sha256": sha(MANUAL / "step1_original_verify.log"), "read_paths": 1300, "read_failures": 0}


def snapshot():
    sources = {"original_memory_content_amendment_L1.md": MD, "private_config.json": PHASE / "config.json",
        "private_frozen_config.json": PHASE / "frozen_config.json", "original_prior_stages_preservation.json": OLD_MANIFEST}
    sources.update({"pre_public/" + name: OUT / name for name in NAMES})
    records = {}
    for name, source in sources.items():
        destination = safe(SNAPSHOT / name)
        put(destination, safe(source).read_bytes(), raw=True, exclusive=True)
        safe(destination).chmod(destination.stat().st_mode & ~stat.S_IWRITE)
        records[name] = {"original_path": str(safe(source)), "snapshot_path": str(destination), "sha256": sha(destination),
            "bytes": destination.stat().st_size, "readonly": not bool(destination.stat().st_mode & stat.S_IWRITE)}
        if sha(source) != records[name]["sha256"] or not records[name]["readonly"]: raise ValueError("Snapshot copy/readonly verification failed")
    put(SNAP_MANIFEST, {"status": "PASS", "utc": utc(), "files": records}, exclusive=True)
    safe(SNAP_MANIFEST).chmod(SNAP_MANIFEST.stat().st_mode & ~stat.S_IWRITE)
    return records


def assert_private_unchanged(records):
    for name in ("private_config.json", "private_frozen_config.json", "original_prior_stages_preservation.json"):
        if sha(records[name]["original_path"]) != records[name]["sha256"]: raise ValueError("An immutable private original changed")


def write_composite(records, public_hashes, phase):
    old = read(OLD_MANIFEST)
    if set(old) & ADDED: raise ValueError("Composite extension collides with an original manifest field")
    current = deepcopy(old)
    current.update(current_effective_public_hashes=public_hashes, history_snapshot_manifest_hash=sha(SNAP_MANIFEST),
        amendment_original_sha256=records["original_memory_content_amendment_L1.md"]["sha256"],
        amendment_current_sha256=sha(MD), original_preservation_manifest_sha256=sha(OLD_MANIFEST), public_update_phase=phase)
    proof("all original preservation-manifest fields", old, {k: v for k, v in current.items() if k not in ADDED})
    put(CURRENT, current)


def hash_table(title, old, current):
    return "\n" + title + "\n\n|文件|旧阶段4.2快照 SHA256|当前完整文件 SHA256|\n|---|---|---|\n" + "".join(
        "|" + name + "|" + old[name]["sha256"] + "|" + current[name]["sha256"] + "|\n" for name in NAMES)


def prepare_public():
    gate = step1_gate()
    if PRE_SCORE.exists() or SNAP_MANIFEST.exists(): raise ValueError("Prior publication snapshot/receipt exists; ROOT must inspect it before another prepare")
    cfg, private_cfg = read(OUT / "config.json"), read(PHASE / "config.json")
    proof("pre-update public/private config", cfg, private_cfg)
    md_original = MD.read_bytes().decode("utf-8")
    if "## 公开目录更新" in md_original: raise ValueError("Public-update amendment section already exists")
    preserved = {"O": read(DATA / "phase4_2/stage4_1_public_snapshot/config.json"), "L0": read(PHASE / "stage4_2_public_snapshot/config.json")}
    prior_metrics = read(PHASE / "stage4_2_public_snapshot/metrics.json")
    before_metrics = read(OUT / "metrics.json")
    new_metrics = {**before_metrics, "prior_stage4_1_results": deepcopy(prior_metrics["prior_stage4_1_results"]), "prior_stage4_2_results": deepcopy(prior_metrics["results"])}
    if "preserved_stage_configs" in cfg: raise ValueError("Unexpected pre-existing preserved_stage_configs")
    cfg["preserved_stage_configs"] = preserved
    predictions = predictions_check()
    records = snapshot()
    put(OUT / "config.json", cfg); put(OUT / "metrics.json", new_metrics)
    historical = {name: {"sha256": sha(PHASE / "stage4_2_public_snapshot" / name)} for name in NAMES}
    section = "\n\n## 公开目录更新\n\n用户明确批准方案A，保留并恢复O/L0配置与结果展示块；L1原状态和错误记录保留，终局评分尚待执行。\n1300条清单路径只读检查全部成功、读取失败0；原PermissionError未记录失败文件名，确切原因未知，不能据此次重读成功推断为临时占用。\n原主说明、私有配置、冻结记录、原保全manifest及更新前六公开文件已保留只读快照，见manual_terminal_resume_20261031/history_snapshot_manifest.json。原私有配置、frozen_config和原manifest不变。\npredictions旧6000行/298912字节前缀逐字节相同；当前完整9000行含新增L1预测，所以完整文件hash变化。llm_calls及run.log同样保留原追加前缀。下表区分快照完整hash和更新后完整hash，不改历史hash。\n"
    current = hashes()
    put(MD, (md_original + section + hash_table("### 第二步公开对象更新后哈希", historical, current)).encode("utf-8"), raw=True)
    write_composite(records, current, "step2")
    put(MANUAL / "step2_public_update.json", {"status": "PASS", "utc": utc(), "step1": gate,
        "snapshot_manifest_sha256": sha(SNAP_MANIFEST), "public_config_only_added_key": "preserved_stage_configs",
        "L1_status_and_error_preserved": True, "amendment_sha256": sha(MD), "current_manifest_sha256": sha(CURRENT)}, exclusive=True)
    published = read(OUT / "config.json"); display = published.pop("preserved_stage_configs")
    checks = [proof("public config minus display equals private config", published, private_cfg), proof("O/L0 complete preserved configs", display, preserved)]
    public_metrics = read(OUT / "metrics.json")
    checks += [proof("O results display block", public_metrics["prior_stage4_1_results"], prior_metrics["prior_stage4_1_results"]),
        proof("L0 results display block", public_metrics["prior_stage4_2_results"], prior_metrics["results"])]
    expected_metrics = deepcopy(before_metrics); expected_metrics.update(prior_stage4_1_results=prior_metrics["prior_stage4_1_results"], prior_stage4_2_results=prior_metrics["results"])
    checks.append(proof("all original L1 metrics including status/error", public_metrics, expected_metrics))
    if predictions_check() != predictions: raise ValueError("Predictions changed while publishing display objects")
    assert_private_unchanged(records)
    put(MANUAL / "step3_public_compare.json", {"status": "PASS", "utc": utc(), "object_float_hex_proofs": checks, "predictions": predictions}, exclusive=True)
    original_readme = (OUT / "README.md").read_bytes().decode("utf-8"); lines = original_readme.splitlines(keepends=True)
    positions = [i for i, line in enumerate(lines) if line.startswith("Status:")]
    if len(lines) > 20 or len(positions) != 1: raise ValueError("Expected one README Status line within20 lines")
    index = positions[0]; old_line = lines[index]; newline = "\r\n" if old_line.endswith("\r\n") else "\n" if old_line.endswith("\n") else ""
    suffix = old_line.rstrip("\r\n").split(";", 1)
    lines[index] = "Status: stage4.3 L1运行完成，终局评分待完成。;" + (suffix[1] if len(suffix) == 2 else " stage4.4 waits for explicit continue and has not started.") + newline
    put(OUT / "README.md", "".join(lines).encode("utf-8"), raw=True)
    reread = (OUT / "README.md").read_bytes().decode("utf-8").splitlines(keepends=True)
    if len(reread) > 20 or any(a != b for i, (a, b) in enumerate(zip(original_readme.splitlines(keepends=True), reread)) if i != index): raise ValueError("A non-status README revision line changed")
    put(MANUAL / "step4_readme_pending.json", {"status": "PASS", "utc": utc(), "README_lines": len(reread), "only_Status_line_changed": True, "sha256": sha(OUT / "README.md")}, exclusive=True)
    current = hashes()
    put(MD, MD.read_bytes() + hash_table("### 终评分前公开文件哈希", historical, current).encode("utf-8"), raw=True)
    write_composite(records, current, "pre_score")
    receipt = {"status": "PASS", "utc": utc(), "public_hashes": current, "amendment_sha256": sha(MD), "current_manifest_sha256": sha(CURRENT),
        "history_snapshot_manifest_sha256": sha(SNAP_MANIFEST), "post_allowed": [str(safe(MD)), str(safe(CURRENT))], "L1_TEST_scoring_started": False}
    put(PRE_SCORE, receipt, exclusive=True); safe(PRE_SCORE).chmod(PRE_SCORE.stat().st_mode & ~stat.S_IWRITE)
    return receipt


def finalize_public():
    pre = read(PRE_SCORE); records = read(SNAP_MANIFEST)["files"]
    if pre.get("status") != "PASS" or pre["history_snapshot_manifest_sha256"] != sha(SNAP_MANIFEST): raise ValueError("Pre-score seal changed")
    if not (PHASE / "evaluation_started.json").exists() or not (PHASE / "terminal_results.json").exists(): raise ValueError("Terminal scoring artifacts are required before final publication hashes")
    stop = read(MANUAL / "continuation_stop.json"); metrics = read(OUT / "metrics.json")
    terminal = read(PHASE / "terminal_results.json"); paired = read(PHASE / "paired_comparisons.json")
    if (stop.get("status") != "complete" or stop.get("L1_TEST_target_accessor_calls") != 1 or stop.get("next_stage_started") is not False
            or metrics.get("status") != "complete" or metrics.get("results") != terminal
            or metrics.get("paired_comparisons") != paired or paired.get("registered_L0_minus_O_MAE_check", {}).get("status") != "PASS"):
        raise ValueError("Terminal completion/public results/paired-reference PASS must all precede final hash appendix")
    for name in ("config.json", "llm_calls.jsonl", "predictions.csv", "run.log"):
        if sha(OUT / name) != pre["public_hashes"][name]["sha256"]: raise ValueError("An immutable pre-score public file changed: " + name)
    target = MANUAL / "final_public_update.json"
    if target.exists(): raise ValueError("Final public update already sealed")
    assert_private_unchanged(records)
    original = MD.read_bytes(); previous = original.decode("utf-8")
    if sha(MD) != pre["amendment_sha256"] or "### 终评分后公开文件哈希" in previous: raise ValueError("Pre-score amendment changed or final table already exists")
    current = hashes(); historical = {name: {"sha256": sha(PHASE / "stage4_2_public_snapshot" / name)} for name in NAMES}
    predictions_check()
    put(MD, original + hash_table("### 终评分后公开文件哈希", historical, current).encode("utf-8"), raw=True)
    write_composite(records, current, "post_score")
    result = {"status": "PASS", "utc": utc(), "public_hashes": current, "amendment_sha256": sha(MD), "current_manifest_sha256": sha(CURRENT),
        "pre_score_receipt_sha256": sha(PRE_SCORE), "existing_paths_modified": [str(safe(MD)), str(safe(CURRENT))], "public_scientific_or_report_files_modified": False}
    put(target, result, exclusive=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=("prepare_public", "finalize_public"))
    args = parser.parse_args(); result = prepare_public() if args.action == "prepare_public" else finalize_public()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
