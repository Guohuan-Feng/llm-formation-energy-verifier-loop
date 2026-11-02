"""Stage4.2 storage; frozen stage4.1 code and artifacts remain read-only."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import os
import stat
import ctypes
import ctypes.wintypes as w
import h4_common as old

ROOT=old.ROOT; DATA=old.DATA; OUT=old.OUT; SRC=old.SRC
PHASE=DATA/'phase4_2'
ARCHIVE=PHASE/'stage4_1_public_snapshot'
safe=old.safe;sha=old.sha;read=old.read;write=old.write;text=old.text
digest=old.digest;utc=old.utc;log=old.log;Awake=old.Awake
RUNTIME=old.RUNTIME; FILES=old.FILES

def tree_hashes(directory):
    return {str(p.resolve()):sha(p) for p in sorted(Path(directory).rglob('*')) if p.is_file()}

def preserve_stage4_1():
    manifest=PHASE/'stage4_1_preservation.json'
    if manifest.exists():
        verify_stage4_1(read(manifest)); return read(manifest)
    old.verify()
    safe(PHASE).mkdir(parents=True,exist_ok=True)
    public={p.name:{'path':str(p.resolve()),'sha256':sha(p),'size':p.stat().st_size} for p in OUT.iterdir()}
    for name,item in public.items():
        p=safe(ARCHIVE/name)
        if p.exists():
            if sha(p)!=item['sha256']: raise ValueError('Preexisting archive differs')
        else: old.bytes_write(p,Path(item['path']).read_bytes(),exclusive=True)
        p.chmod(p.stat().st_mode & ~stat.S_IWRITE)
    config=read(OUT/'config.json')
    record={'created_utc':utc(),'public':public,'private_phase4_1':tree_hashes(DATA/'phase4_1'),
        'original_source_hashes':config['source_hashes'],'original_input_hashes':config['input_hashes'],
        'old_readonly_hashes':config['old_readonly_hashes'],
        'stage4_1_marker_sha256':sha(DATA/'phase4_1/evaluation_started.json'),
        'global_CLI_calls_before_stage4_2':60,
        'archive_readonly':True,'stage4_1_TEST_not_rescored':True}
    write(manifest,record,exclusive=True)
    return record

def prefix_matches(path,record):
    import hashlib
    with safe(path,writable=False).open('rb') as stream:
        payload=stream.read(record['size'])
    return len(payload)==record['size'] and hashlib.sha256(payload).hexdigest()==record['sha256']

def verify_stage4_1(record=None):
    record=read(PHASE/'stage4_1_preservation.json') if record is None else record
    for name,item in record['public'].items():
        if sha(ARCHIVE/name)!=item['sha256']: raise ValueError('Stage4.1 archived public bytes changed')
        if (ARCHIVE/name).stat().st_mode & stat.S_IWRITE: raise ValueError('Stage4.1 archive lost read-only flag')
    if tree_hashes(DATA/'phase4_1')!=record['private_phase4_1']:
        raise ValueError('Stage4.1 private candidates/final/evaluation records changed')
    for section in ('original_source_hashes','original_input_hashes','old_readonly_hashes'):
        for path,expected in record[section].items():
            if sha(path)!=expected: raise ValueError('Old source/input/result changed: '+Path(path).name)
    for name in ('llm_calls.jsonl','run.log','predictions.csv'):
        if not prefix_matches(OUT/name,record['public'][name]):
            raise ValueError('Stage4.1 public prefix changed: '+name)
    return {'stage4_1_preserved':True,'old_files':len(record['old_readonly_hashes']),
        'stage4_1_private_files':len(record['private_phase4_1']),'stage4_1_public_files':6}

def update(status,phase,**details):
    with old.LOCK:
        p=OUT/'metrics.json';m=read(p) if p.exists() else {}
        m.update(stage='4.2',condition='L0',status=status,phase=phase,updated_utc=utc(),**details)
        write(p,m);return m

def readme(status,forecast=None):
    lines=['# H4 stage4.2: L0 feature loop without memory',
        'Status: '+status+'; stage4.2 only; stage4.3 waits for explicit continue.',
        'Stage4.2 is newly frozen before its first call; stage4.1 public snapshot, source, candidates/final and TEST marker remain unchanged.',
        'Each seed starts at frozen S0500, then20 rounds of75+carry labels; round20 remaining budget fills uniformly to2000.',
        'POOL-only final_structure space groups use pymatgen SpacegroupAnalyzer(symprec=0.01,angle_tolerance=5.0); versions and5000 number/symbol records are hashed.',
        'Query fields use H4A label-free grammar plus template_spacegroup number closed interval; this is the substituted final geometry, not original template symmetry.',
        'Existing4.1 template split roles are preserved; new groups use the same seed/hash cutoff, fixed before querying; no per-round reshuffle.',
        'Magpie-only S0 supplies initial validation feedback; it is excluded from20-round winner selection.',
        'Each request sees the complete current active feature code, validation MAE, worst20 labeled validation rows, labeled summary and unlabeled catalogue.',
        'Validation labels may enter feedback by design; validation members grow with reveals, so round MAEs use different sample memberships.',
        'HGB parameters/unweighted fit/TRAIN-only median imputation match4.1; inner-fit feature failures>20% are dropped.',
        'Next-round code is the complete surviving feature set after this scientific failure mask, never budget truncation; raw sets remain in the ledger.',
        'Select minimum saved round validation rowMAE, earliest tie; final fill never reselects or changes round scores.',
        'Winner cached TRAIN cells are retained; only missing final TRAIN members are executed before full2000 refit and TEST1000 prediction.',
        'H4 batch sandbox remains2 workers/50ms per function and structure,100-structure checkpoints; no concurrent CLI and sandbox execution.',
        'Pinned gpt-6.1-sol/high CLI, tools/network disabled, <=4 slots; globalH4 cap300 includes original60 and all repairs/retries.',
        'L0 terminal gate reads TEST1000 once for3seeds; prior O scoring once remains unchanged, so total H4 scoring passes become2.',
        'Row and fullTEST template-group weights;2000 shared group bootstraps; arithmetic means of seed metrics, not an ensemble.',
        'Later stages/H5 and old experiments remain inactive/read-only; full artifacts and curve are under work/h-data/H4/phase4_2.']
    if forecast is not None:lines.append('Pilot forecast: '+format(forecast['estimated_total_seconds']/3600,'.3f')+' hours; estimate, not measured whole-stage runtime.')
    if len(lines)>20: raise ValueError('README exceeds20 lines')
    text(OUT/'README.md','\n'.join(lines)+'\n')

def process_identity():
    class FT(ctypes.Structure): _fields_=[('low',w.DWORD),('high',w.DWORD)]
    api=ctypes.windll.kernel32;api.GetCurrentProcess.restype=w.HANDLE
    api.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(FT)]*4;api.GetProcessTimes.restype=w.BOOL
    values=[FT() for _ in range(4)]
    if not api.GetProcessTimes(api.GetCurrentProcess(),*[ctypes.byref(x) for x in values]):raise OSError('GetProcessTimes failed')
    ticks=values[0].high<<32|values[0].low
    from datetime import datetime,timezone
    return {'pid':os.getpid(),'parent_pid':os.getppid(),'creation_time_100ns':ticks,
        'creation_utc':datetime.fromtimestamp(ticks/1e7-11644473600,timezone.utc).isoformat(),
        'driver':str((SRC/'run_h4_l0.py').resolve()),'arguments':sys.argv[1:],'stage':'4.2','utc':utc()}

class RunLock:
    def __enter__(self):
        p=safe(PHASE/'owner.lock');p.parent.mkdir(parents=True,exist_ok=True);self.f=p.open('a+b');self.f.seek(0,2)
        if self.f.tell()==0:self.f.write(b'0');self.f.flush()
        self.f.seek(0);import msvcrt
        try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except BaseException:self.f.close();raise RuntimeError('Another L0 driver owns this stage')
        return self
    def __exit__(self,*args):
        import msvcrt
        self.f.seek(0);msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1);self.f.close()

def verify(cfg=None,*,full=True):
    cfg=read(PHASE/'config.json') if cfg is None else cfg
    if sha(PHASE/'config.json')!=read(PHASE/'frozen_config.json')['sha256']:
        raise ValueError('L0 frozen config changed')
    if sha(OUT/'config.json')!=sha(PHASE/'config.json'):raise ValueError('Public L0 config differs from frozen private copy')
    for section in ('source_hashes','input_hashes'):
        for path,expected in cfg[section].items():
            if sha(path)!=expected:raise ValueError('Frozen L0 source/input changed: '+Path(path).name)
    if {p.name for p in OUT.iterdir()}!=FILES:raise ValueError('Public experiment requires exactly6 files')
    if len((OUT/'README.md').read_text().splitlines())>20:raise ValueError('README exceeds20 lines')
    audit=verify_stage4_1() if full else {'stage4_1_full_check_deferred':True}
    return {'status':'PASS','utc':utc(),**audit,'sources':len(cfg['source_hashes']),'inputs':len(cfg['input_hashes'])}
