"""H4 persistent CPU batch sandbox; reuse H3's restricted numeric language.

One primitive-tree validation/transfer/readonly freeze per structure, then the
same parent GO-to-result50ms hard deadline separately for every function.
Generated functions execute only in parent-owned memory/process-limited workers.
No LLM call or filesystem mutation happens on import. Checkpoints, if requested,
are confined to work/h-data/H4 after resolving directory links.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
import hashlib,json,math,multiprocessing as mp,os,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import h3_sandbox as old

ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'work/h-data/H4'
POLICY_VERSION='H4-batched-input-original50ms-v1'
sys.dont_write_bytecode=True


def safe(path):
    p=Path(path).resolve();root=DATA.resolve()
    if 'onedrive' in str(p).casefold() or not p.is_relative_to(root):raise ValueError('H4 sandbox output must remain under non-OneDrive H4 data')
    return p


def _child(connection,sources,memory_mb):
    sys.dont_write_bytecode=True
    os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    try:
        if connection.recv()!='INIT':return
        if os.name!='nt':
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(memory_mb*1024**2,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
        started=time.perf_counter();import numpy as np
        functions=[]
        for source in sources:
            if source is None:functions.append(None);continue
            old.validate_feature(source)
            namespace={'__builtins__':old.SAFE_BUILTINS,
                'math':old._Namespace({name:getattr(math,name) for name in old.MATH_FUNCTIONS|old.MATH_CONSTANTS}),
                'np':old._numeric_namespace(np)}
            exec(compile(source,'<restricted-H4-feature>','exec'),namespace)
            functions.append(namespace['feature'])
        connection.send({'phase':'ready','child_initialize_seconds':time.perf_counter()-started})
        structure=None
        while True:
            packet=connection.recv()
            if packet is None:return
            if packet[0]=='PREPARE':
                t=time.perf_counter();structure=old._readonly(packet[1])
                connection.send({'phase':'prepared','prepare_seconds':time.perf_counter()-t});continue
            if packet[0]!='GO' or structure is None:raise RuntimeError('Invalid batch sandbox protocol')
            index=packet[1];function=functions[index]
            if function is None:raise RuntimeError('Rejected feature must not be dispatched')
            t=time.perf_counter()
            try:
                value=function(structure)
                if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):
                    raise TypeError('Feature must return exactly one real numeric scalar')
                value=float(value)
                if not math.isfinite(value):result={'status':'nonfinite','error':'Nonfinite feature result','value':None}
                else:result={'status':'ok','error':'','value':value}
            except BaseException as exc:
                result={'status':'exception','error':f'{type(exc).__name__}: {str(exc)[:400]}','value':None}
            result.update(phase='result',feature_index=index,execution_seconds=time.perf_counter()-t)
            connection.send(result)
    except (EOFError,BrokenPipeError):pass
    except BaseException as exc:
        try:connection.send({'phase':'startup_error','error':f'{type(exc).__name__}: {str(exc)[:400]}'})
        except Exception:pass
    finally:connection.close()


class _Session:
    def __init__(self,sources,memory_mb,startup_timeout,prepare_timeout):
        self.sources=sources;self.memory_mb=memory_mb;self.startup_timeout=startup_timeout;self.prepare_timeout=prepare_timeout
        self.context=mp.get_context('spawn');self.process=None;self.connection=None;self.job=None
        self.startup_count=0;self.timeout_count=0;self.startup_failures=0
    def close(self):
        p=self.process
        if self.job is not None:self.job.close();self.job=None
        if p is not None:
            if p.is_alive():p.terminate()
            p.join(timeout=1)
            if p.is_alive():p.kill();p.join(timeout=1)
            if p.is_alive():raise RuntimeError('H4 worker could not be reaped')
        if self.connection is not None:self.connection.close()
        self.process=None;self.connection=None
    def recv(self,timeout):
        if not self.connection.poll(max(0.,timeout)):raise TimeoutError('Sandbox IPC deadline exceeded')
        return self.connection.recv()
    def start(self):
        t=time.perf_counter();parent,child=self.context.Pipe(duplex=True)
        self.process=self.context.Process(target=_child,args=(child,self.sources,self.memory_mb),name='H4-batch-sandbox',daemon=True)
        self.connection=parent;self.process.start();child.close()
        try:
            if os.name=='nt':self.job=old._WindowsJob(self.process.pid,self.memory_mb)
            parent.send('INIT');packet=self.recv(self.startup_timeout)
            if packet.get('phase')!='ready':raise RuntimeError(packet.get('error','H4 initialization failed'))
        except BaseException:self.close();self.startup_failures+=1;raise
        self.startup_count+=1;return time.perf_counter()-t
    def prepare(self,plain):
        t=time.perf_counter();self.connection.send(('PREPARE',plain));packet=self.recv(self.prepare_timeout)
        if packet.get('phase')!='prepared':raise RuntimeError(packet.get('error','Input preparation failed'))
        return time.perf_counter()-t
    def evaluate(self,index,record,validation,parent_input_seconds=None):
        import numpy as np
        n=len(self.sources);values=np.full(n,np.nan);status=np.full(n,'ast_rejected',dtype='<U32')
        execution=np.full(n,np.nan);window=np.full(n,np.nan);errors=[]
        timing={'startup_seconds':0.,'prepare_seconds':0.,'parent_input_seconds':0.}
        wall=time.perf_counter()
        try:
            if parent_input_seconds is None:
                t=time.perf_counter();plain=old._plain_input(record)
                if type(plain) is not dict:raise ValueError('Feature input must be a structure dictionary')
                timing['parent_input_seconds']=time.perf_counter()-t
            else:
                # Private caller has already validated/copied this primitive tree.
                plain=record;timing['parent_input_seconds']=parent_input_seconds
        except Exception as exc:
            status[:]='invalid_input';errors.append({'row':index,'feature':None,'status':'invalid_input','error':str(exc)[:400]})
            return self._packet(index,values,status,execution,window,timing,errors,wall)
        prepared=False
        for feature,source in enumerate(self.sources):
            if source is None:continue
            try:
                if self.process is None or not self.process.is_alive():
                    self.close();timing['startup_seconds']+=self.start();prepared=False
                if not prepared:timing['prepare_seconds']+=self.prepare(plain);prepared=True
            except Exception as exc:
                self.close();prepared=False;status[feature]='startup_failure'
                errors.append({'row':index,'feature':feature,'status':'startup_failure','error':f'{type(exc).__name__}: {str(exc)[:400]}'})
                if self.startup_failures>=3:raise RuntimeError('Three sandbox initialization failures; finite batch stopped')
                continue
            execution_started=time.perf_counter()
            try:
                self.connection.send(('GO',feature))
                packet=self.recv(.05-(time.perf_counter()-execution_started));window[feature]=time.perf_counter()-execution_started
                if window[feature]>.05:raise TimeoutError('Execution result arrived after50ms deadline')
                if packet.get('phase')!='result' or packet.get('feature_index')!=feature:raise RuntimeError('Feature result protocol mismatch')
                status[feature]=packet['status'];execution[feature]=packet['execution_seconds']
                if packet['status']=='ok':values[feature]=float(packet['value'])
                else:errors.append({'row':index,'feature':feature,'status':packet['status'],'error':packet['error']})
            except TimeoutError as exc:
                window[feature]=time.perf_counter()-execution_started;status[feature]='timeout';self.timeout_count+=1
                errors.append({'row':index,'feature':feature,'status':'timeout','error':str(exc)})
                self.close();prepared=False
            except (EOFError,BrokenPipeError,OSError) as exc:
                status[feature]='worker_exit';errors.append({'row':index,'feature':feature,'status':'worker_exit','error':type(exc).__name__})
                self.close();prepared=False
            except Exception as exc:
                status[feature]='protocol_failure';errors.append({'row':index,'feature':feature,'status':'protocol_failure','error':str(exc)[:400]})
                self.close();prepared=False
        return self._packet(index,values,status,execution,window,timing,errors,wall)
    @staticmethod
    def _packet(index,values,status,execution,window,timing,errors,wall):
        return {'index':index,'values':values,'status':status,'execution_seconds':execution,'execution_window_seconds':window,
            **timing,'wall_seconds':time.perf_counter()-wall,'errors':errors}


class FeatureBatchRunner:
    def __init__(self,sources,n_workers=4,timeout_ms=50,memory_limit_mb=512,startup_timeout_seconds=15.,prepare_timeout_seconds=3.):
        if timeout_ms!=50:raise ValueError('Frozen deadline is exactly50ms')
        if not isinstance(n_workers,int) or not 1<=n_workers<=16:raise ValueError('CPU workers must be1..16')
        if not 128<=memory_limit_mb<=2048:raise ValueError('Worker memory must be128..2048MiB')
        items=list(sources.items()) if isinstance(sources,dict) else [(f'feature_{i}',source) for i,source in enumerate(sources)]
        if not items or len(items)>100:raise ValueError('Batch requires1..100 features')
        self.names=[name for name,_ in items];self.sources=[];self.validation=[]
        for name,source in items:
            try:details=old.validate_feature(source);self.sources.append(source);self.validation.append({'name':name,'valid':True,'details':details})
            except old.FeatureValidationError as exc:self.sources.append(None);self.validation.append({'name':name,'valid':False,'error':str(exc)})
        self.n_workers=n_workers;self.memory_mb=memory_limit_mb;self.startup_timeout=startup_timeout_seconds;self.prepare_timeout=prepare_timeout_seconds
        self.closed=False;self.sessions=[];self.lock=threading.Lock();self.local=threading.local()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):
        for session in self.sessions:session.close()
        self.closed=True
    def _evaluate(self,index,record,parent_input_seconds=None):
        session=getattr(self.local,'session',None)
        if session is None:
            session=_Session(self.sources,self.memory_mb,self.startup_timeout,self.prepare_timeout);self.local.session=session
            with self.lock:self.sessions.append(session)
        return session.evaluate(index,record,self.validation,parent_input_seconds)
    def evaluate_many(self,records,*,checkpoint_dir=None,checkpoint_every=100,progress=None):
        import numpy as np
        if self.closed:raise RuntimeError('FeatureBatchRunner is closed')
        if checkpoint_every!=100:raise ValueError('H4 checkpoint interval is fixed100 structures')
        checkpoint=safe(checkpoint_dir) if checkpoint_dir is not None else None
        if checkpoint is not None:checkpoint.mkdir(parents=True,exist_ok=True)
        started=time.perf_counter();iterator=iter(records);packets={};ids=[];seen=set();submitted=0;done=0;errors=[]
        exhausted=False
        try:
            with ThreadPoolExecutor(max_workers=self.n_workers,thread_name_prefix='H4-worker-owner') as pool:
                while not exhausted:
                    # Decode/primitive-copy100 inputs only while no feature is in flight.
                    # The same-GIL producer cannot consume a feature's50ms IPC window.
                    chunk=[]
                    for unused in range(100):
                        try:record=next(iterator)
                        except StopIteration:exhausted=True;break
                        if type(record) is dict and 's' in record and 'id' in record:rid=str(record['id']);structure=record['s']
                        else:rid=str(submitted);structure=record
                        if rid in seen:raise ValueError('Duplicate structure identity in batch')
                        seen.add(rid);ids.append(rid);index=submitted;submitted+=1
                        t=time.perf_counter()
                        try:
                            plain=old._plain_input(structure)
                            if type(plain) is not dict:raise ValueError('Feature input must be a structure dictionary')
                            chunk.append((index,plain,time.perf_counter()-t))
                        except Exception as exc:
                            n=len(self.names);packet={'index':index,'values':np.full(n,np.nan),'status':np.full(n,'invalid_input',dtype='<U32'),
                                'execution_seconds':np.full(n,np.nan),'execution_window_seconds':np.full(n,np.nan),
                                'startup_seconds':0.,'prepare_seconds':0.,'parent_input_seconds':time.perf_counter()-t,
                                'wall_seconds':time.perf_counter()-t,'errors':[{'row':index,'feature':None,'status':'invalid_input','error':str(exc)[:400]}]}
                            packets[index]=packet;errors.extend(packet['errors']);done+=1
                    pending={pool.submit(self._evaluate,index,plain,prepare):index for index,plain,prepare in chunk}
                    while pending:
                        ready,_=wait(pending,return_when=FIRST_COMPLETED)
                        for future in ready:
                            index=pending.pop(future);packet=future.result();packets[index]=packet;errors.extend(packet['errors']);done+=1
                    if chunk or done:
                        summary={'completed':done,'submitted':submitted,'wall_seconds':time.perf_counter()-started,'workers':self.n_workers}
                        if checkpoint is not None:self._checkpoint(checkpoint,ids,packets,summary)
                        if progress is not None and done%100==0:progress(summary)
                    if not chunk and exhausted:break
        finally:self.close()
        n=len(ids);f=len(self.names)
        values=np.full((n,f),np.nan);status=np.full((n,f),'pending',dtype='<U32');execution=np.full((n,f),np.nan);window=np.full((n,f),np.nan)
        timing={key:np.zeros(n) for key in ('startup_seconds','prepare_seconds','parent_input_seconds','wall_seconds')}
        for index,packet in packets.items():
            values[index]=packet['values'];status[index]=packet['status'];execution[index]=packet['execution_seconds'];window[index]=packet['execution_window_seconds']
            for key in timing:timing[key][index]=packet[key]
        result={'ids':np.array(ids),'feature_names':np.array(self.names),'values':values,'status':status,
            'execution_seconds':execution,'execution_window_seconds':window,**timing,'errors':errors,'validation':self.validation,
            'performance':{'policy_version':POLICY_VERSION,'original_AST_policy_version':old.POLICY_VERSION,'wall_seconds':time.perf_counter()-started,
                'n_structures':n,'n_features':f,'n_workers':self.n_workers,'n_sandbox_processes_started':sum(s.startup_count for s in self.sessions),
                'n_timeouts':sum(s.timeout_count for s in self.sessions),'status_counts':dict(Counter(status.flat)),
                'deadline_ms':50,'deadline_includes':'parentGO dispatch, child execution, resultIPC and scheduling; unchanged fromH3',
                'prepare_once_per_structure':True,'input_decode_and_primitive_copy':'batch100 barrier with no inflight generated feature',
                'generated_code_in_parent':False,'memory_mb_per_worker':self.memory_mb}}
        if checkpoint is not None:
            self._checkpoint(checkpoint,ids,packets,{'completed':done,'submitted':submitted,'wall_seconds':result['performance']['wall_seconds'],'workers':self.n_workers,'complete':True})
        return result
    def _checkpoint(self,directory,ids,packets,summary):
        import numpy as np
        n=len(ids);f=len(self.names);values=np.full((n,f),np.nan);status=np.full((n,f),'pending',dtype='<U32')
        for i,packet in packets.items():values[i]=packet['values'];status[i]=packet['status']
        temp=safe(directory/'checkpoint.tmp.npz');target=safe(directory/'checkpoint.npz')
        with temp.open('wb') as handle:np.savez_compressed(handle,ids=np.array(ids),values=values,status=status,feature_names=np.array(self.names));handle.flush();os.fsync(handle.fileno())
        os.replace(temp,target)
        temp=safe(directory/'progress.tmp.json');target=safe(directory/'progress.json')
        temp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,target)
