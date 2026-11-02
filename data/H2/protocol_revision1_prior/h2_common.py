"""H2 file safety, finite budget accounting and lean progress artifacts."""
from __future__ import annotations
import ctypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'H2_formal_dataset'
DATA = ROOT/'work'/'h-data'/'H2'
BUDGET_SECONDS = 24*3600

def safe(path):
    p=Path(path).resolve()
    if 'onedrive' in str(p).casefold() or not p.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Forbidden H2 destination: {p}')
    return p

def setup():
    for path in [OUT,DATA,DATA/'cache',DATA/'cache'/'tmp']:
        safe(path).mkdir(parents=True,exist_ok=True)
    for key,child in [('TEMP','tmp'),('TMP','tmp'),('TORCH_HOME','torch'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg')]:
        p=safe(DATA/'cache'/child);p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    os.chdir(safe(DATA))

def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    p=safe(path);tmp=safe(p.with_suffix(p.suffix+'.tmp'))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def gz_write(path,value):
    with safe(path).open('wb') as f:
        with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
            z.write(json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8'))
def gz_read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
def log(message):
    line=now()+' '+str(message)
    with safe(OUT/'run.log').open('a',encoding='utf-8') as f:
        f.write(line+'\n');f.flush();os.fsync(f.fileno())
    print(line,flush=True)
def append_jsonl(path,records):
    with safe(path).open('a',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        f.flush();os.fsync(f.fileno())
def snapshot_old():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ['P1','H1_synthetic_pilot'] for p in sorted((ROOT/folder).iterdir()) if p.is_file()}
def source_hashes(names):return {n:sha(ROOT/n) for n in names}
def verify_config(cfg):
    assert cfg['old_artifact_hashes']==snapshot_old(),'Old H1/P1 artifacts changed'
    for n,d in cfg.get('benchmark_source_hashes',{}).items():assert sha(ROOT/n)==d,f'Frozen benchmark source changed: {n}'
    assert sha(DATA/'frozen_plan.json.gz')==cfg['frozen_plan_sha256']

class Budget:
    """Count active phase time; intentional gaps between phases cost no compute."""
    def __init__(self,phase):
        self.phase=phase;self.started=time.perf_counter();self.path=DATA/'budget.json'
        state=read(self.path) if self.path.exists() else {'budget_seconds':BUDGET_SECONDS,'spent_seconds':0.,'phases':[]}
        if state.get('active_phase'):
            raise RuntimeError('Prior phase was interrupted. Save/audit its budget before an explicit full-phase recomputation.')
        self.previous=float(state['spent_seconds']);self.state=state
        state['active_phase']={'name':phase,'started_utc':now()};write(self.path,state)
    def elapsed(self):return time.perf_counter()-self.started
    def remaining(self):return BUDGET_SECONDS-self.previous-self.elapsed()
    def close(self,status):
        t=self.elapsed();self.state['spent_seconds']=self.previous+t
        self.state['phases'].append({'name':self.phase,'elapsed_seconds':t,'status':status,'completed_utc':now()})
        self.state.pop('active_phase',None);write(self.path,self.state)
    def heartbeat(self):
        self.state['active_phase']['elapsed_seconds']=self.elapsed();write(self.path,self.state)

class Awake:
    def __enter__(self):
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):raise RuntimeError('Sleep prevention failed')
        return self
    def __exit__(self,*args):ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def metrics_update(**values):
    p=OUT/'metrics.json';m=read(p) if p.exists() else {};m.update(values);m['updated_utc']=now();write(p,m);return m
