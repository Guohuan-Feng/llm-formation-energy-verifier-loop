"""Public Matbench formation-energy pilot data; deterministic label-blind split.

No old research data, fits, LLM calls, or experiment directory writes occur here.
Only standard-library dependencies are needed to prepare/load raw structures.
"""
from __future__ import annotations

import collections
from fractions import Fraction
import functools
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "work" / "p1-data"
RAW_PATH = DATA_DIR / "matbench_mp_e_form.json.gz"
PILOT_PATH = DATA_DIR / "p1_10000.json.gz"
EXPECTED_SHA256 = "dedcb1d4ba2e3e50dbdd45ba5bc647a00e9c2bcf8f8bf556dc8e92caa39eb21f"
N_FULL = 132752
N_PILOT = 10000
SEED = "P1-formation-energy-20261016-v1"


def safe_path(path):
    p = Path(path).resolve()
    if any("onedrive" in s.lower() for s in p.parts):
        raise ValueError(f"OneDrive destination forbidden: {p}")
    for parent in [p, *p.parents]:
        if parent.exists() and parent.is_symlink():
            raise ValueError(f"Symlink destination forbidden: {parent}")
    return p


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash(purpose, value):
    return hashlib.sha256(f"{SEED}|{purpose}|{value}".encode()).hexdigest()


def _round_robin_indices(records, eligible, n, purpose):
    """Maximize distinct groups before a second row; all ordering label-blind."""
    bygroup = collections.defaultdict(list)
    for i in eligible:
        bygroup[records[i]["group"]].append(i)
    for group in bygroup:
        bygroup[group].sort(key=lambda i: _hash(purpose + "-record", records[i]["id"]))
    ordered = sorted(bygroup, key=lambda g: _hash(purpose + "-group", g))
    chosen = []
    level = 0
    while len(chosen) < n:
        added = False
        for group in ordered:
            if len(bygroup[group]) > level:
                chosen.append(bygroup[group][level])
                added = True
                if len(chosen) == n:
                    break
        if not added:
            raise ValueError("Insufficient eligible rows for round-robin selection")
        level += 1
    return sorted(chosen)


def reduced_composition(structure):
    """Exact canonical element counts modulo common multiplier, not element set.

    Integer compositions remain exact; rational occupancies are supported. This
    deliberately does not use chemical ordering conventions or oxidation states.
    """
    amounts = collections.defaultdict(Fraction)
    for site in structure["sites"]:
        for species in site["species"]:
            amounts[species["element"]] += Fraction(str(species.get("occu", 1))).limit_denominator(1000000)
    positive = {e: a for e, a in amounts.items() if a > 0}
    if not positive:
        raise ValueError("Empty composition")
    lcm = math.lcm(*(a.denominator for a in positive.values()))
    integers = {e: int(a * lcm) for e, a in positive.items()}
    gcd = functools.reduce(math.gcd, integers.values())
    return "|".join(f"{e}:{integers[e] // gcd}" for e in sorted(integers))


def _iter_raw_rows(path):
    """Stream the top-level pandas split-orient JSON without retaining 824 MB."""
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise ValueError("Missing data array")
            buffer += chunk
            match = re.search(r'"data"\s*:\s*\[', buffer)
            if match:
                prefix = buffer[:match.start()]
                # Canonical file index is 0..132751; no MP identifiers provided.
                idx_match = re.search(r'"index"\s*:\s*(\[[^\]]*\])', prefix)
                if not idx_match or json.loads(idx_match.group(1)) != list(range(N_FULL)):
                    raise ValueError("Unexpected source index")
                buffer = buffer[match.end():]
                break
        pos = 0
        row_index = 0
        while True:
            while pos < len(buffer) and buffer[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buffer) and buffer[pos] == "]":
                break
            try:
                row, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    raise ValueError(f"Truncated row {row_index}")
                buffer = buffer[pos:] + chunk
                pos = 0
                continue
            yield row_index, row
            row_index += 1
            pos = end
            if pos > 1024 * 1024:
                buffer = buffer[pos:]
                pos = 0
        if row_index != N_FULL:
            raise ValueError(f"Expected {N_FULL} rows, got {row_index}")


def source_metadata():
    return {
        "dataset": "matbench_mp_e_form", "dataset_version": "Matbench v0.1",
        "original_records": N_FULL,
        "target": "Materials Project DFT formation energy per atom",
        "record_id_definition": "mb-eform-NNNNNN denotes original zero-based benchmark row index; no MP material identifier is present in this file",
        "unit": "eV/atom", "structures": "DFT-relaxed Materials Project structures; not proxy CIF assignments",
        "model_input_policy": "Use elements,lattice,fractionalcoordinates only. Ignore structure/site properties including DFT magmom. No target or DFT-derived magnetic feature enters model or LLM packets.",
        "original_retrieval_date": "2019-04-02",
        "official_download_url": "https://ml.materialsproject.org/projects/matbench_mp_e_form.json.gz",
        "actual_download_url": "https://ndownloader.figshare.com/files/17476028",
        "download_fallback_reason": "Official ML endpoint returned HTTP403; author Figshare binary exact-SHA256 match to current official matminer metadata",
        "official_sha256": EXPECTED_SHA256,
        "license": "MIT", "license_source": "https://api.figshare.com/v2/articles/9755432",
        "dataset_doi": "10.6084/m9.figshare.9755432.v1",
        "figshare_description_discrepancy": "Figshare text says formation energy cutoff3.0eV; current matminer text2.5eV. Binary identity verified by exact SHA256; no cleaning/filtering repeated.",
        "metadata_url": "https://raw.githubusercontent.com/hackingmaterials/matminer/6cb5638571a7b3c2d52b9beb8d2c9346d20eaa81/src/matminer/datasets/dataset_metadata.json",
        "benchmark_paper": "https://doi.org/10.1038/s41524-020-00406-3",
        "official_benchmark_protocol": "Full132752 records, original order,5fold shuffled sklearnKFold random_state18012019, selection within each TRAIN fold only",
        "official_protocol_source": "https://hackingmaterials.lbl.gov/automatminer/datasets.html",
        "pilot_comparability": "This10k reduced-composition-disjoint pilot is NOT the official Matbench leaderboard split and cannot establish a leaderboard result.",
        "public_gnn_reference": {
            "source": "https://raw.githubusercontent.com/materialsproject/matbench/936176db18ca4cd7b38cbd957c017a5bac770c6b/docs_src/Leaderboards%20Per-Task/matbench_v0.1_matbench_mp_e_form.md",
            "official_repo_commit": "936176db18ca4cd7b38cbd957c017a5bac770c6b",
            "commit_date": "2024-01-20T09:41:36Z",
            "checked_date": "2026-10-16",
            "coGN": {"mae_eV_per_atom": 0.0170, "fold_std": 0.0003},
            "coNGN": {"mae_eV_per_atom": 0.0178, "fold_std": 0.0004},
            "CGCNN_v2019": {"mae_eV_per_atom": 0.0337, "fold_std": 0.0006},
            "comparison_scope": "Published full official5fold scores for context only; not subset-controlled comparisons",
        },
    }


def prepare():
    """Download if needed, verify exact official binary, freeze pilot manifest."""
    safe_path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    if PILOT_PATH.exists():
        return load_prepare()
    if not RAW_PATH.exists():
        urllib.request.urlretrieve(source_metadata()["actual_download_url"], safe_path(RAW_PATH))
    digest = sha256_file(RAW_PATH)
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Official source SHA256 mismatch: {digest}")
    selected_indices = sorted(sorted(range(N_FULL), key=lambda i: _hash("pilot-record", i))[:N_PILOT])
    selected_set = set(selected_indices)
    records = []
    for i, row in _iter_raw_rows(RAW_PATH):
        if i in selected_set:
            structure, target = row
            group = reduced_composition(structure)
            records.append({
                "id": f"mb-eform-{i:06d}", "source_index": i,
                "structure": structure, "target": float(target), "group": group,
            })
    if len(records) != N_PILOT:
        raise ValueError("Pilot selection incomplete")
    # Keep each composition whole. Greedy hash order exact-target fill, with no
    # target-value access. Oversized group skipped and later smaller groups fill.
    bygroup = collections.defaultdict(list)
    for i, r in enumerate(records):
        bygroup[r["group"]].append(i)
    holdout_groups = set()
    n_holdout = 0
    for group in sorted(bygroup, key=lambda g: _hash("holdout-group", g)):
        size = len(bygroup[group])
        if n_holdout + size <= 2000:
            holdout_groups.add(group)
            n_holdout += size
        if n_holdout == 2000:
            break
    if n_holdout < 2000:
        raise ValueError("Could not fill2000 holdout records with whole composition groups")
    for r in records:
        r["split"] = "HOLDOUT" if r["group"] in holdout_groups else "TRAIN"
    holdout = [i for i, r in enumerate(records) if r["split"] == "HOLDOUT"]
    # Independent deterministic record hash: precisely200 held-out structures.
    verifier = sorted(sorted(holdout, key=lambda i: _hash("verifier-record", records[i]["id"]))[:200])
    train = [i for i, r in enumerate(records) if r["split"] == "TRAIN"]
    verifier_train = _round_robin_indices(records, train, 1024, "verifier-train")
    verifier_set, verifier_train_set = set(verifier), set(verifier_train)
    for i, r in enumerate(records):
        r["verifier"] = i in verifier_set
        r["verifier_train"] = i in verifier_train_set
    metadata = source_metadata()
    metadata.update({
        "raw_path": str(RAW_PATH), "raw_bytes": RAW_PATH.stat().st_size,
        "raw_sha256": digest, "pilot_path": str(PILOT_PATH),
        "seed": SEED, "selection": "10000 smallest SHA256(seed|pilot-record|original-index), no target access",
        "split": "Reduced-composition groups sorted by independent SHA256; whole groups greedily fill2000HOLDOUT; remaining8000TRAIN. No target access.",
        "group_definition": "Sorted element:integer reduced stoichiometry from exact rational site occupancies; polymorphs share group",
        "verifier_selection": "200 smallest independent verifier-record hashes inside HOLDOUT, no target access",
        "n": len(records), "n_groups": len(bygroup),
        "n_train": len(train), "n_train_groups": len({records[i]["group"] for i in train}),
        "n_holdout": len(holdout), "n_holdout_groups": len(holdout_groups),
        "n_verifier": len(verifier), "n_verifier_groups": len({records[i]["group"] for i in verifier}),
        "n_verifier_train": len(verifier_train), "n_verifier_train_groups": len({records[i]["group"] for i in verifier_train}),
        "verifier_train_selection": "1024TRAIN structures by group round-robin; independent SHA256 ordering within and across groups; no target access",
        "composition_overlap_train_holdout": 0,
        "source_indices_sha256": hashlib.sha256(json.dumps(selected_indices, separators=(",", ":")).encode()).hexdigest(),
        "train_ids": [records[i]["id"] for i in train],
        "holdout_ids": [records[i]["id"] for i in holdout],
        "verifier_ids": [records[i]["id"] for i in verifier],
        "verifier_train_ids": [records[i]["id"] for i in verifier_train],
    })
    payload = {"metadata": metadata, "records": records}
    with safe_path(PILOT_PATH).open("wb") as out:
        # Deterministic gzip container as well as deterministic rows/split.
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0) as z:
            z.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())
    return load_prepare()


def load_prepare(as_pymatgen=False):
    """Return frozen pilot records, raw/pymatgen structures, IDs/y/groups/splits.

    Indices are list positions0..9999. Use train_indices/holdout_indices and
    verifier_indices; all three are frozen and independent of target values.
    """
    if not PILOT_PATH.exists():
        return prepare()
    with gzip.open(PILOT_PATH, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    if len(records) != N_PILOT:
        raise ValueError("Wrong pilot record count")
    train = [i for i, r in enumerate(records) if r["split"] == "TRAIN"]
    holdout = [i for i, r in enumerate(records) if r["split"] == "HOLDOUT"]
    verifier = [i for i, r in enumerate(records) if r["verifier"]]
    verifier_train = [i for i, r in enumerate(records) if r["verifier_train"]]
    assert len(train) == 8000 and len(holdout) == 2000 and len(verifier) == 200
    assert not ({records[i]["group"] for i in train} & {records[i]["group"] for i in holdout})
    structures = [r["structure"] for r in records]
    if as_pymatgen:
        from pymatgen.core import Structure
        structures = [Structure.from_dict(s) for s in structures]
    return {
        "records": records, "structures": structures,
        "ids": [r["id"] for r in records], "targets": [r["target"] for r in records],
        "groups": [r["group"] for r in records], "splits": [r["split"] for r in records],
        "source_indices": [r["source_index"] for r in records],
        "train_indices": train, "holdout_indices": holdout, "verifier_indices": verifier,
        "verifier_train_indices": verifier_train, "verifier_test_indices": verifier,
        "metadata": {**payload["metadata"], "pilot_sha256": sha256_file(PILOT_PATH)},
    }


if __name__ == "__main__":
    result = prepare()
    meta = result["metadata"]
    print(json.dumps({k: meta[k] for k in ["n", "n_groups", "n_train", "n_train_groups", "n_holdout", "n_holdout_groups", "n_verifier", "n_verifier_groups", "raw_sha256", "pilot_sha256"]}, indent=2))
