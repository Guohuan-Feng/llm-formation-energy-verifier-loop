"""CPU software tests: fake features/fitter only, never real fit/execute/targets."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[5]
PHASE = ROOT/'work/h-data/H4/phase4_4'
sys.path.insert(0, str(PHASE/'src'))
import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import p44_models as m


class FakeSandbox:
    calls = 0
    def __init__(self, sources, **kwargs):
        assert kwargs['n_workers'] == 2 and kwargs['timeout_ms'] == 50
        self.names = list(sources)
    def __enter__(self): return self
    def __exit__(self, *unused): pass
    def evaluate_many(self, records, *, checkpoint_dir, checkpoint_every, progress):
        FakeSandbox.calls += 1
        rows = list(records); assert checkpoint_every == 100
        m.c.safe(checkpoint_dir, write=True)
        n = len(rows)
        values = np.ones((n, 2)); status = np.full((n, 2), 'ok', dtype='<U32')
        if n == 5: values[:2, 0] = np.nan; values[0, 1] = np.nan
        else: values[:, 1] = np.nan
        status[~np.isfinite(values)] = 'timeout'
        return {'ids': np.asarray([r['id'] for r in rows]), 'feature_names': np.asarray(self.names),
                'values': values, 'status': status, 'errors': [], 'validation': [], 'performance': {}}


class Tests(unittest.TestCase):
    def setUp(self):
        base = m.c.safe(PHASE/'software/tmp', write=True); base.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=base)
        self.phase = Path(self.tmp.name).resolve()
        self.change = patch.object(m, 'PHASE', self.phase); self.change.start()
    def tearDown(self):
        self.change.stop(); self.tmp.cleanup()
    def fixture(self):
        ids = [f'software_train{i}' for i in range(5)]+['software_test0','software_test1']
        b = {'index': {rid:i for i,rid in enumerate(ids)}, 'magpie132': np.ones((7,132)),
             'structures_by_id': {rid:{} for rid in ids}, 'source_hashes': {'mock':'mock'}}
        f = [{'name':name,'description':'mock','source':'def feature(s):\n    return 1.0\n'} for name in ('bad','boundary')]
        return b, ids[:5], ids[5:], f
    def arm(self, fitter):
        b,tr,te,f = self.fixture()
        return m._hgb_arm({},b,domain='real',group='O',seed=0,train_ids=tr,predict_ids=te,
            labels={rid:float(i) for i,rid in enumerate(tr)},features=f,model_fitter=fitter,sandbox_factory=FakeSandbox)
    def test_train_failure_mask_and_holdout_diagnostic_only(self):
        seen = {}
        def fit(x,y,z):
            seen['shape'] = (x.shape,z.shape)
            return np.array([.1,.2]), {'mock':True}, b'software-model-only'
        value = self.arm(fit)
        self.assertEqual(seen['shape'], ((5,133),(2,133)))
        self.assertEqual(value['feature_selection']['kept_names'], ['boundary'])
        self.assertEqual(value['feature_selection']['prediction_failure_rates'][1], 1.)
        self.assertFalse(value['evaluation_target_values_read'])
    def test_successful_cache_skips_fitter_and_feature_execution(self):
        self.arm(lambda x,y,z:(np.zeros(len(z)),{},b'mock'))
        with patch.object(m,'_fit',side_effect=AssertionError('no fit')):
            previous = FakeSandbox.calls
            self.arm(lambda *a: (_ for _ in ()).throw(AssertionError('no refit')))
        self.assertEqual(previous,FakeSandbox.calls)
    def test_cached_artifact_tamper_rejects_without_refit(self):
        value = self.arm(lambda x,y,z:(np.zeros(len(z)),{},b'mock'))
        m.c.write(value['artifacts']['model']['path'],b'changed')
        with self.assertRaisesRegex(RuntimeError,'artifact changed'):
            self.arm(lambda *a: (_ for _ in ()).throw(AssertionError('no refit')))
    def test_target_identity_overlap_rejected_before_features(self):
        b,tr,te,f = self.fixture()
        with self.assertRaises(ValueError):
            m._hgb_arm({},b,domain='real',group='O',seed=0,train_ids=tr,predict_ids=[tr[0]],
                labels={rid:0. for rid in tr},features=f,model_fitter=lambda *a:None,sandbox_factory=FakeSandbox)
    def test_output_escape_rejected(self):
        with self.assertRaises(ValueError): m.c.safe(ROOT/'work/h-data/H3b/forbidden.json',write=True)
    @unittest.skipUnless(os.name=='nt','Windows read-only byte lock')
    def test_read_only_byte_lock_does_not_modify_file(self):
        path=self.phase/'toy_existing_global.lock';m.c.write(path,b'0')
        before=(m.c.sha(path),path.stat().st_mtime_ns,path.stat().st_size)
        with m._read_only_global_gpu_lock(path):
            # The rb descriptor really owns an exclusive byte-range lock: a
            # second handle cannot even read that byte while ownership is held.
            with self.assertRaises(PermissionError): path.read_bytes()
        self.assertEqual(before,(m.c.sha(path),path.stat().st_mtime_ns,path.stat().st_size))
    @unittest.skipUnless(os.name=='nt','Windows read-only byte lock')
    def test_lock_checks_immutability_also_on_exception(self):
        path=self.phase/'toy_existing_global.lock';m.c.write(path,b'0');stamp=path.stat().st_mtime_ns
        with self.assertRaisesRegex(RuntimeError,'bytes/mtime changed'):
            with m._read_only_global_gpu_lock(path):
                os.utime(path,ns=(stamp,stamp+1000000000));raise ValueError('simulated worker failure')
    def test_new_cg_plan_uses_only_train_targets_and_never_old_checkpoint(self):
        old=m.c.read(m.CG_SOURCE_PLAN)
        b={'perseed_O_ids':{str(seed):old['arms'][f'CGCNN_synthetic_trainCHGNet_n2000_seed{seed}']['train_ids'] for seed in range(3)},
           'synthetic_test_ids':old['arms']['CGCNN_synthetic_trainCHGNet_n2000_seed0']['test_ids']}
        ids=set().union(*map(set,b['perseed_O_ids'].values()));labels={rid:0. for rid in ids}
        path=m._new_cg_plan({'software_only':True},b,labels);new=m.c.read(path)
        self.assertEqual(set(new['arms']),{f'CGCNN_n2000_seed{s}' for s in range(3)})
        self.assertEqual(set(m.c.read(new['targets_path'])),ids)
        self.assertFalse(set(m.c.read(new['targets_path']))&set(b['synthetic_test_ids']))
        self.assertNotIn('technical_revision',new)
        self.assertFalse(new['parent_provenance']['old_checkpoint_or_predictions_loaded'])


if __name__=='__main__': unittest.main()
