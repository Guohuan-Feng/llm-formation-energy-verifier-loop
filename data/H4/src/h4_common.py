"""H4-only durable storage and immutable stage4.1 protocol checks."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
SRC = Path(__file__).resolve().parent
DATA = ROOT / 'work/h-data/H4'
OUT = ROOT / 'H4_feature_loop'
PHASE = DATA / 'phase4_1'
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
import threading
import uuid

FILES = {'config.json','predictions.csv','llm_calls.jsonl','metrics.json','run.log','README.md'}
LOCK = threading.RLock()
RUNTIME = {'executable': str(ROOT/'work/h-data/H3/runtime_pinned/codex-0.165.0-alpha.1.2.exe'),
 'sha256':'5c1b55966bf9f757dcbf00065c9a6cb862ead55d34b015f1be1939ff18e43943',
 'cli_version':'codex-cli 0.165.0-alpha.1.2','model':'gpt-6.1-sol','reasoning_effort':'high'}

def utc():
    return datetime.now(timezone.utc).isoformat()

def safe(path, *, writable=True):
    p=Path(path).resolve()
    if 'onedrive' in str(p).casefold() or not p.is_relative_to(ROOT.resolve()):
        raise ValueError('Path escapes non-OneDrive research root')
    if writable and not (p.is_relative_to(DATA.resolve()) or p.is_relative_to(OUT.resolve())):
        raise ValueError('H4 writes must remain in H4 private/public destinations')
    return p

def sha(path):
    h=hashlib.sha256()
    with safe(path,writable=False).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(safe(path,writable=False).read_text(encoding='utf-8-sig'))

def bytes_write(path,content,*,exclusive=False):
    with LOCK:
        p=safe(path); safe(p.parent).mkdir(parents=True,exist_ok=True)
        if exclusive:
            with p.open('xb') as f: f.write(content); f.flush(); os.fsync(f.fileno())
        else:
            tempdir=safe(DATA/'atomic'); tempdir.mkdir(parents=True,exist_ok=True)
            temp=safe(tempdir/(uuid.uuid4().hex+'.tmp'))
            with temp.open('xb') as f: f.write(content); f.flush(); os.fsync(f.fileno())
            os.replace(temp,safe(p))

def write(path,value,*,exclusive=False):
    bytes_write(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8'),exclusive=exclusive)

def text(path,value,*,exclusive=False):
    bytes_write(path,value.encode('utf-8'),exclusive=exclusive)

def log(message):
    with LOCK:
        p=safe(OUT/'run.log')
        with p.open('a',encoding='utf-8',newline='\n') as f:
            f.write('['+utc()+'] '+message+'\n'); f.flush(); os.fsync(f.fileno())

def update(status,phase,percent=None,**extra):
    with LOCK:
        p=OUT/'metrics.json'; m=read(p) if p.exists() else {}
        m.update(status=status,stage='4.1',phase=phase,updated_utc=utc(),**extra)
        if percent is not None:
            m['progress_percent']=max(m.get('progress_percent',0),percent)
            m['progress_milestones_reached']=sorted(set(m.get('progress_milestones_reached',[]))|{x for x in (25,50,75) if percent>=x})
        write(p,m)
        return m

def snapshot_old():
    folders=[p for p in ROOT.iterdir() if p.is_dir() and p!=OUT and
             (p.name[0:1] in ('E','F','G','H') or p.name=='P1')]
    paths=[p for folder in folders for p in folder.rglob('*') if p.is_file()]
    paths.extend(ROOT.glob('*.py'))
    return {str(p.resolve()):sha(p) for p in sorted(set(paths))}

def verify(cfg=None,*,full=True):
    cfg=read(OUT/'config.json') if cfg is None else cfg
    frozen=read(DATA/'frozen_config.json')
    if sha(OUT/'config.json')!=frozen['sha256']:
        raise ValueError('Frozen H4 config changed')
    for section in ('source_hashes','input_hashes'):
        for path,expected in cfg[section].items():
            if sha(path)!=expected: raise ValueError('Frozen '+section+' changed: '+Path(path).name)
    if full and snapshot_old()!=cfg['old_readonly_hashes']:
        raise ValueError('Old experiment/source file set or bytes changed')
    if {p.name for p in OUT.iterdir()}!=FILES: raise ValueError('Public H4 must contain exactly six files')
    if len((OUT/'README.md').read_text(encoding='utf-8').splitlines())>20:
        raise ValueError('README exceeds20 lines')
    if sha(RUNTIME['executable'])!=RUNTIME['sha256']: raise ValueError('Pinned CLI changed')
    for item in cfg['prompt_request_files']:
        if sha(item['path'])!=item['sha256']: raise ValueError('Frozen prompt request bytes changed')
    return {'status':'PASS','utc':utc(),'old_files':len(cfg['old_readonly_hashes']),
            'sources':len(cfg['source_hashes']),'inputs':len(cfg['input_hashes']),
            'prompt_requests':len(cfg['prompt_request_files']),'old_hashes_checked':full}

def readme(status,*,forecast=None):
    lines=['# H4 stage4.1: O one-shot feature generation',
      'Status: '+status+'; stage4.1 only; stages4.2/4.3/4.4 wait for explicit continue.',
      'Seeds0/1/2 reuse H2/H3 frozen S0500 plus uniform-random1500 POOL labels,2000 total; full synthetic TEST1000.',
      'Each seed has20 independent candidates, each with exactly300 sampled labeled examples plus TRAIN2000 summary, <=20 functions.',
      'As requested, examples/summary may include internal validation labels; this validation is not an untouched-label holdout.',
      'No TEST/HOLDOUT label enters prompts, features, queries or selection; original label-free H3 geometry/Magpie132 cache is reused.',
      'Fixed template-group80/20 internal split per seed; candidate selection uses row MAE, ties earliest candidate.',
      'HGB uses original H3 unweighted fit and frozen parameters; TRAIN-only median imputation.',
      'Magpie132 plus candidate functions; failure>20% on inner-fit TRAIN drops a function for validation.',
      'After selection, recompute the failure mask on fullTRAIN2000; reuse cached TRAIN feature values, refit, predict TEST once.',
      'H4-private H3b batch sandbox copy only;2 workers,50ms GO-to-result per function/structure; timeout/error→NaN.',
      'Pinned CLI gpt-6.1-sol/high, no tools/network, at most4 calls; no claim of reproducible LLM output seed.',
      'All60 prompts,300-example memberships, schemas, data/split/source hashes frozen before the first call; <=20000 input bound.',
      'Global H4 call cap300 including repairs/retries;429 retries after60/120/240s; finite manual recovery only.',
      'Terminal scoring: one TEST1000 read for all3 seeds; row/fullTEST-template group weights,2000 shared group bootstraps.',
      'Per-seed metrics and arithmetic means of metrics, not an ensemble; private paired draws retained for later authorized comparisons.',
      'Pilot8 requests plus their candidate validations estimate the full stage; stop if estimate exceeds10 hours.',
      'H4A and all previous experiments remain read-only. Stage4.1 makes no query or real-data transfer test.']
    if forecast is not None: lines.append('Pilot stage forecast: '+format(forecast['estimated_total_seconds']/3600,'.3f')+' h; measured pilot, not measured whole-stage runtime.')
    if len(lines)>20: raise ValueError('README exceeds20')
    text(OUT/'README.md','\n'.join(lines)+'\n')

class Awake:
    def __enter__(self):
        if os.name=='nt':
            if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000000|0x1|0x2):
                raise OSError('SetThreadExecutionState failed')
        return self
    def __exit__(self,*args):
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

class RunLock:
    def __enter__(self):
        p=safe(DATA/'stage4_1_owner.lock'); p.parent.mkdir(parents=True,exist_ok=True)
        self.f=p.open('a+b'); self.f.seek(0,2)
        if self.f.tell()==0: self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        import msvcrt
        try: msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except BaseException: self.f.close(); raise RuntimeError('Another H4 stage4.1 process owns the run')
        return self
    def __exit__(self,*args):
        import msvcrt
        self.f.seek(0); msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1); self.f.close()
