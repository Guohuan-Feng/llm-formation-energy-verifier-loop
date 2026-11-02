"""Pinned, local-only 6000-token memory counter; never a model tokenizer claim."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parent.parent
ROOT = PHASE.parents[3]
VENDOR = PHASE / "vendor"
CACHE = PHASE / "tokenizer_cache"
RECORD = PHASE / "token_counter.json"
ENCODING_NAME = "o200k_base"
TIKTOKEN_VERSION = "0.12.0"
MEMORY_LIMIT = 6000
_ENCODING = None


def safe(path):
    target = Path(path).resolve()
    if "onedrive" in str(target).casefold() or not target.is_relative_to(PHASE.resolve()):
        raise ValueError("L1 counter writes must remain under non-OneDrive phase4_3")
    return target


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load(*, bootstrap=False):
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING
    safe(VENDOR)
    safe(CACHE)
    if not bootstrap and (not RECORD.is_file() or not any(CACHE.iterdir())):
        raise RuntimeError("Frozen local tokenizer/cache is missing; no runtime download permitted")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE)
    sys.path.insert(0, str(VENDOR)) if str(VENDOR) not in sys.path else None
    import tiktoken
    if tiktoken.__version__ != TIKTOKEN_VERSION or not Path(tiktoken.__file__).resolve().is_relative_to(VENDOR.resolve()):
        raise RuntimeError("Pinned private tiktoken package differs")
    _ENCODING = tiktoken.get_encoding(ENCODING_NAME)
    return _ENCODING


def count_text(text):
    if type(text) is not str:
        raise TypeError("Memory counter takes the exact serialized Unicode text")
    return len(_load().encode_ordinary(text))


def measure(payload):
    text = canonical(payload)
    return {"text": text, "token_count": count_text(text), "UTF8_bytes": len(text.encode("utf-8"))}


def _files(directory):
    return {str(path.resolve()): sha(path) for path in sorted(directory.rglob("*")) if path.is_file()}


def bootstrap_record():
    """Root-only step1: package-data download allowed, zero inference/API calls."""
    if RECORD.exists():
        return verify()
    encoding = _load(bootstrap=True)
    sample = canonical({"notes": "Plain text <|endoftext|> and \u4e2d\u6587", "round": 1})
    record = {
        "protocol": "H4.3-memory-local-token-counter-v1",
        "package": "tiktoken", "version": TIKTOKEN_VERSION,
        "encoding": ENCODING_NAME, "counter": "len(encoding.encode_ordinary(exact_memory_JSON_text))",
        "limit_tokens": MEMORY_LIMIT,
        "serialization": {"format": "JSON", "sort_keys": True, "ensure_ascii": False,
            "separators": [",", ":"], "allow_nan": False, "trailing_newline": False,
            "scope": "Entire memory JSON including definitions/contributions/failures/query statistics/all notes; outer prompt label excluded"},
        "special_token_policy": "encode_ordinary: special-looking strings remain ordinary plaintext",
        "claims_to_match_pinned_model_internal_tokenizer": False,
        "whole_request_guard": "Separate actual UTF8 strict-wire+note+global/model instructions+1500 framing; include repair suffix",
        "source_sha256": sha(__file__), "vendor_file_hashes": _files(VENDOR),
        "encoding_cache_hashes": _files(CACHE),
        "sample": {"text": sample, "token_ids": encoding.encode_ordinary(sample)},
        "mergeable_ranks": len(encoding._mergeable_ranks),
        "pattern_sha256": hashlib.sha256(encoding._pat_str.encode()).hexdigest(),
        "documentation": ["https://developers.openai.com/api/docs/guides/advanced-usage",
            "https://developers.openai.com/api/docs/guides/token-counting"],
        "scientific_LLM_calls": 0, "TEST_targets_read": 0,
    }
    destination = safe(RECORD)
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return verify()


def verify():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    if (record["source_sha256"] != sha(__file__) or record["version"] != TIKTOKEN_VERSION
            or record["encoding"] != ENCODING_NAME or record["limit_tokens"] != MEMORY_LIMIT):
        raise RuntimeError("Frozen memory counter identity changed")
    if _files(VENDOR) != record["vendor_file_hashes"] or _files(CACHE) != record["encoding_cache_hashes"]:
        raise RuntimeError("Frozen tokenizer package/data changed")
    encoding = _load()
    if encoding.encode_ordinary(record["sample"]["text"]) != record["sample"]["token_ids"]:
        raise RuntimeError("Frozen memory tokenization test vector changed")
    return {"status": "PASS", "version": TIKTOKEN_VERSION, "encoding": ENCODING_NAME,
        "limit_tokens": MEMORY_LIMIT, "token_counter_sha256": sha(RECORD),
        "vendor_files": len(record["vendor_file_hashes"]),
        "encoding_cache_files": len(record["encoding_cache_hashes"]),
        "scientific_LLM_calls": 0, "TEST_targets_read": 0}


if __name__ == "__main__":
    print(json.dumps(bootstrap_record(), ensure_ascii=False))
