"""Stage4.3-only storage and preservation of both completed H4 conditions."""
from __future__ import annotations
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parent.parent
DATA = PHASE.parent
ROOT = DATA.parents[2]
SRC = PHASE / "src"
OLD_SRC = DATA / "src"
OUT = ROOT / "H4_feature_loop"
ARCHIVE = PHASE / "stage4_2_public_snapshot"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OLD_SRC))
import hashlib
import json
import os
import stat
import uuid
import h4_common as base
import h4_l0_common as prior

RUNTIME = dict(base.RUNTIME)
Awake = base.Awake
utc = base.utc
digest = base.digest
read = base.read
sha = base.sha
PUBLIC_FILES = {"config.json", "predictions.csv", "llm_calls.jsonl", "metrics.json", "run.log", "README.md"}


def safe(path, *, writable=True):
    value = Path(path).resolve()
    if "onedrive" in str(value).casefold() or not value.is_relative_to(ROOT.resolve()):
        raise ValueError("H4.3 path must remain in the non-OneDrive research root")
    if writable and not (value.is_relative_to(PHASE.resolve()) or value.is_relative_to(OUT.resolve())):
        raise ValueError("New stage4.3 writes are confined to phase4_3 and the six public outputs")
    return value


def write(path, value, *, exclusive=False):
    return bytes_write(path, (json.dumps(value, ensure_ascii=False, sort_keys=True,
        indent=2, allow_nan=False) + "\n").encode("utf-8"), exclusive=exclusive)


def text(path, value):
    return bytes_write(path, value.encode("utf-8"))


def bytes_write(path, value, *, exclusive=False):
    destination = safe(path)
    safe(destination.parent).mkdir(parents=True, exist_ok=True)
    if exclusive:
        with destination.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary_dir = safe(PHASE / "atomic")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary = safe(temporary_dir / (uuid.uuid4().hex + ".tmp"))
    try:
        with temporary.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def tree_hashes(directory):
    return {str(path.resolve()): sha(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}


def prefix_matches(ref):
    h = hashlib.sha256()
    with safe(ref["path"], writable=False).open("rb") as stream:
        remaining = ref["size"]
        while remaining:
            block = stream.read(min(4 * 1024 * 1024, remaining))
            if not block:
                return False
            h.update(block)
            remaining -= len(block)
    return h.hexdigest() == ref["sha256"]


def preserve():
    manifest = PHASE / "prior_stages_preservation.json"
    if manifest.exists():
        record = read(manifest)
        verify_preservation(record)
        return record
    import run_h4_l0
    audit = run_h4_l0.verify_terminal()
    if audit["status"] != "complete" or audit["CLI_calls_global"] != 120:
        raise RuntimeError("Stage4.3 must start after completeL0 and exactly120 existing calls")
    if {path.name for path in OUT.iterdir()} != PUBLIC_FILES:
        raise RuntimeError("Expected exactly six completed-stage public files")
    public = {}
    for path in sorted(OUT.iterdir()):
        ref = {"path": str(path.resolve()), "sha256": sha(path), "size": path.stat().st_size}
        public[path.name] = ref
        destination = safe(ARCHIVE / path.name)
        if destination.exists():
            if sha(destination) != ref["sha256"]:
                raise RuntimeError("Existing pre-L1 public snapshot differs")
        else:
            bytes_write(destination, path.read_bytes(), exclusive=True)
        destination.chmod(destination.stat().st_mode & ~stat.S_IWRITE)
    prefixes = {name: public[name] for name in ("llm_calls.jsonl", "predictions.csv", "run.log")}
    for name in ("request_results.jsonl", "transport_events.jsonl"):
        path = DATA / "transport" / name
        prefixes[name] = {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha(path)}
    mutable = {Path(ref["path"]).resolve() for ref in prefixes.values()}
    transport_files = {path: value for path, value in tree_hashes(DATA / "transport").items() if Path(path) not in mutable}
    record = {"utc": utc(), "public": public, "prefixes": prefixes,
        "phase4_1": tree_hashes(DATA / "phase4_1"), "phase4_2": tree_hashes(DATA / "phase4_2"),
        "original_H4_sources": tree_hashes(OLD_SRC), "old_transport_files": transport_files,
        "O_marker_sha256": sha(DATA / "phase4_1/evaluation_started.json"),
        "L0_marker_sha256": sha(DATA / "phase4_2/evaluation_started.json"),
        "O_terminal_sha256": sha(DATA / "phase4_1/terminal_results.json"),
        "L0_terminal_sha256": sha(DATA / "phase4_2/terminal_results.json"),
        "global_calls_before_L1": 120, "prior_TEST_scoring_passes": 2,
        "old_stage_verify": audit, "readonly_public_snapshots": True}
    write(manifest, record, exclusive=True)
    verify_preservation(record)
    return record


def verify_preservation(record=None, *, full=True):
    record = read(PHASE / "prior_stages_preservation.json") if record is None else record
    prior.verify_stage4_1()
    for name, ref in record["public"].items():
        path = ARCHIVE / name
        if sha(path) != ref["sha256"] or path.stat().st_mode & stat.S_IWRITE:
            raise RuntimeError("Completed L0 public snapshot changed or lost readonly flag")
    for ref in record["prefixes"].values():
        if not prefix_matches(ref):
            raise RuntimeError("An old successful-call/log/prediction prefix changed")
    for key, directory in (("phase4_1", DATA / "phase4_1"), ("phase4_2", DATA / "phase4_2"),
                           ("original_H4_sources", OLD_SRC)):
        if full and tree_hashes(directory) != record[key]:
            raise RuntimeError("A prior-stage private tree/source changed: " + key)
    if full:
        for path, expected in record["old_transport_files"].items():
            if sha(path) != expected:
                raise RuntimeError("An old owned CLI response/protocol artifact changed")
    for stage, key in (("phase4_1", "O_marker_sha256"), ("phase4_2", "L0_marker_sha256")):
        if sha(DATA / stage / "evaluation_started.json") != record[key]:
            raise RuntimeError("A prior TEST scoring gate changed")
    return {"O_preserved": True, "L0_preserved": True,
        "old_private_files": len(record["phase4_1"]) + len(record["phase4_2"]), "old_public_snapshots": 6}


def verify(cfg=None, *, full=True):
    cfg = read(PHASE / "config.json") if cfg is None else cfg
    if sha(PHASE / "config.json") != read(PHASE / "frozen_config.json")["sha256"]:
        raise RuntimeError("L1 frozen configuration hash changed")
    if (OUT / "config.json").read_bytes() != (PHASE / "config.json").read_bytes():
        raise RuntimeError("Published and private L1 configs differ")
    for section in ("source_hashes", "input_hashes"):
        for path, expected in cfg[section].items():
            if sha(path) != expected:
                raise RuntimeError("Frozen L1 source/input changed: " + Path(path).name)
    for path, expected in cfg.get("global_context_hashes", {}).items():
        actual = Path(path).resolve(strict=True)
        if "onedrive" in str(actual).casefold() or actual.name not in ("AGENTS.md", "AGENTS.override.md"):
            raise RuntimeError("Unexpected global instruction input")
        if hashlib.sha256(actual.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Frozen global model instructions changed")
    import l1_counter
    counter = l1_counter.verify()
    old = read(ARCHIVE / "config.json")
    if cfg["models"]["parameters"] != old["models"]["parameters"]:
        raise RuntimeError("HGB parameters changed between L0 and L1")
    for seed in ("0", "1", "2"):
        for key in ("S0_ids", "fit_group_mapping"):
            if cfg["data"]["perseed"][seed][key] != old["data"]["perseed"][seed][key]:
                raise RuntimeError("L1 initial membership/template mapping differs from L0")
    if {path.name for path in OUT.iterdir()} != PUBLIC_FILES or len((OUT / "README.md").read_text().splitlines()) > 20:
        raise RuntimeError("Public six-file/README20 boundary violated")
    return {**verify_preservation(full=full), "counter": counter, "status": "PASS"}


def log(message):
    return base.log("L1 " + message)


def update(status, phase, **details):
    with base.LOCK:
        path = OUT / "metrics.json"
        value = read(path) if path.exists() else {}
        value.update(stage="4.3", condition="L1", status=status, phase=phase, updated_utc=utc(), **details)
        write(path, value)
        return value


def readme(status, *, input_limit, forecast=None):
    lines = ["# H4 stage4.3: L1 feature loop with memory", "Status: " + status + "; stage4.4 waits for explicit continue.",
        "Original O and L0 results/markers/private trees and readonly six-file snapshots are preserved; old predictions and calls are not replayed.",
        "Each seed starts at the same L0 S0500;20 rounds of75+carry, then uniform fill to2000, reported separately.",
        "L0 query grammar/POOL-only space groups/template hash mapping/HGB parameters/unweighted fit/TRAIN median remain fixed.",
        "Memory uses pinned tiktoken0.12.0/o200k_base encode_ordinary on exact compact sorted JSON;6000 tokens; not a model-internal-tokenizer claim.",
        "Memory stores TRAIN/validation feature definitions, drop-one refit contributions, failure reasons, query statistics and <=300-character notes.",
        "Over6000: deterministic top30 contributions plus complete latest3 memory records, with all query statistics and notes; deduplicate definitions, never truncate source.",
        "If the required retained memory still exceeds6000 tokens, stop; never silently weaken the retention rule.",
        "Real complete-request UTF8 guard including strict wire/schema/note/global/model instructions+1500 framing+repair is " + str(input_limit) + " bytes.",
        "The L1 limit is set before freeze from measured L0 surrogate memory+round20 base extrapolation with10% rounded-up margin; theoretical upper bounds are records only.",
        "L0 changed its input limit during round7 recovery; L1 pre-freeze setting does not change any previous L0 request or result.",
        "Internal validation labels enter feedback; its members grow with reveals. No TEST/HOLDOUT labels enter prompts, memory or selection.",
        "Winner=min saved round validation rowMAE, earliest tie; persist before fill; original TRAIN cache reused before full2000 refit and TEST1000 prediction.",
        "CLI max4 slots (3 seed dependencies); sandbox2workers/50ms; no CLI/sandbox overlap; global300 counts O60+L060+L1 and all repairs/retries.",
        "One L1 TEST1000 truth read shared by3seeds; prior O1+L01 unchanged, so total H4 terminal scoring passes3.",
        "Row/fullTEST-template-group metrics;2000 shared group bootstraps; arithmetic seed means, not ensemble; paired comparisons use private cached draws only.",
        "L1-L0 compares the complete memory/feedback/query trajectory; source, memory, request-byte ledgers and reports remain under phase4_3."]
    if forecast is not None:
        lines.append("Pilot forecast hours=" + format(forecast["estimated_total_seconds"] / 3600, ".3f") + "; estimate, not measured whole-stage time.")
    text(OUT / "README.md", "\n".join(lines) + "\n")


def process_identity():
    return {**prior.process_identity(), "stage": "4.3", "driver": str(SRC / "run_h4_l1.py")}


class RunLock:
    def __enter__(self):
        path = safe(PHASE / "owner.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        import msvcrt
        try:
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
        except BaseException:
            self.stream.close()
            raise RuntimeError("Another L1 process owns this finite stage")
        return self

    def __exit__(self, *args):
        import msvcrt
        self.stream.seek(0)
        msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()
