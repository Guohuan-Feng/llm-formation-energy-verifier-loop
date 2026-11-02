"""Stage4.4 storage and preservation; never launches a scientific request."""
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
import sys
import threading
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5]
PHASE = ROOT / 'work/h-data/H4/phase4_4'
PUBLIC = ROOT / 'H4_feature_loop'
_LOCK = threading.RLock()
_PUBLIC_STAT_CACHE = {}


def safe(path, *, write=False):
    path = Path(path).resolve()
    if 'onedrive' in str(path).casefold() or not path.is_relative_to(PHASE if write else ROOT):
        raise ValueError('Stage4.4 paths must resolve inside their authorized local root: ' + str(path))
    return path


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    path = safe(path)
    digest = hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
    except OSError as error:
        if getattr(error, 'filename', None) is None:
            error.filename = str(path)
        raise
    return digest.hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))


def write(path, value, *, exclusive=False):
    path = safe(path, write=True)
    safe(path.parent, write=True).mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else value.encode('utf-8') if isinstance(value, str) else (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if exclusive:
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return
    temporary = safe(PHASE / 'atomic' / (uuid.uuid4().hex + '.tmp'), write=True)
    safe(temporary.parent, write=True).mkdir(parents=True, exist_ok=True)
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, safe(path, write=True))


def event(kind, **details):
    with _LOCK:
        path = safe(PHASE / 'run.log', write=True)
        with path.open('ab') as stream:
            stream.write((json.dumps({'utc': utc(), 'event': kind, **details}, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
            stream.flush()


def public_guard(*, full=False):
    baseline = read(PHASE / 'audit/prior_public_manifest.json')
    for name, identity in baseline['public_readonly'].items():
        path = safe(identity['path'])
        observed = path.stat()
        signature = (observed.st_size, observed.st_mtime_ns)
        if observed.st_size != identity['bytes']:
            if name == 'llm_calls.jsonl':
                rows = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
                attempts = sum(row.get('record_type') == 'reservation' for row in rows)
                if attempts != 183:
                    raise RuntimeError('LLM call guard violated; expected183, observed' + str(attempts))
            raise RuntimeError('A protected public file changed; stage4.4 must stop: ' + str(path))
        if full or _PUBLIC_STAT_CACHE.get(name) != signature:
            if sha(path) != identity['sha256']:
                raise RuntimeError('A protected public file changed; stage4.4 must stop: ' + str(path))
            if name == 'llm_calls.jsonl':
                records = [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
                attempts = sum(item.get('record_type') == 'reservation' for item in records)
                if attempts != 183:
                    raise RuntimeError('LLM call guard violated; expected183, observed' + str(attempts))
            _PUBLIC_STAT_CACHE[name] = signature
    return {'public_files_unchanged': True, 'global_CLI_calls': 183, 'stage4_4_LLM_calls': 0}


def verify_frozen(*, full=True):
    cfg = read(PHASE / 'config.json')
    if sha(PHASE / 'config.json') != read(PHASE / 'frozen_config.json')['sha256']:
        raise RuntimeError('Stage4.4 frozen configuration changed')
    for family in ('source_hashes', 'input_hashes'):
        for path, expected in cfg[family].items():
            if sha(path) != expected:
                raise RuntimeError('Frozen source/input changed: ' + path)
    public_guard(full=True)
    if full:
        for path, expected in read(PHASE / 'audit/prior_stage_files.json')['sha256'].items():
            if sha(path) != expected:
                raise RuntimeError('A completed H4 stage file changed: ' + path)
    return cfg


def permission_details(error):
    return {'error_type': type(error).__name__, 'message': str(error),
            'filename': getattr(error, 'filename', None), 'filename2': getattr(error, 'filename2', None),
            'errno': getattr(error, 'errno', None), 'winerror': getattr(error, 'winerror', None)}


@contextmanager
def prevent_sleep():
    if os.name != 'nt':
        yield
        return
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = kernel.SetThreadExecutionState
    api.argtypes = [ctypes.c_uint]
    api.restype = ctypes.c_uint
    previous = api(0x80000001)
    if not previous:
        raise OSError(ctypes.get_last_error(), 'SetThreadExecutionState failed')
    try:
        yield
    finally:
        api(previous)
