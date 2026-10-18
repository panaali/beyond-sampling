"""Four saved-baseline comparisons plus fixed synthetic lifecycle failures."""
from pathlib import Path
from dataclasses import asdict,FrozenInstanceError
import ast
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
import warnings
import numpy as np
from adapter import ObservedEstimator

HERE=Path(__file__).resolve().parent
OUT=HERE/'controls'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

class SentinelError(Exception): pass

class FailingStream:
    def __init__(self,base,fail_at):self.base=base;self.fail_at=fail_at;self.attempts=0
    def standard_normal(self,shape):
        k=self.attempts;self.attempts+=1
        if k==self.fail_at:raise SentinelError('frozen stream failure')
        return self.base.standard_normal(shape)

def main():
    assert not OUT.exists(),'preserve all attempts'
    assert 0<resource.getrlimit(resource.RLIMIT_CPU)[0]<=20
    assert 0<resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    OUT.mkdir();start=time.process_time()
    r=dict(complete=False,rows=[],exception_controls=[],nested_control=None,original_Q_estimator_calls=0,
           synthetic_Q_calls=0,framework_forward_calls=0,Gaussian_RNG_calls=0)
    def save():
        r.update(cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        p=OUT/'results.json.tmp';p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
        with p.open('rb') as f:os.fsync(f.fileno())
        os.replace(p,OUT/'results.json')
    def event(phase,**kw):
        with (OUT/'progress.jsonl').open('a') as f:
            f.write(json.dumps(dict(phase=phase,**kw),allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    def cap():assert time.process_time()-start<17,'17CPU cap, no automatic retry'
    save();event('attempt',stage='preflight')
    try:
        p=read(HERE/'protocol.json');g=read(HERE/'execution_gate.json')
        assert g['clearance'] is True and g['protocol_sha256']==sha(HERE/'protocol.json')
        # Verify the frozen sets separately: execution metadata cannot override protocol pins.
        for pins in (p['pins'],g['pins']):
            for path,digest in pins.items():assert sha(path)==digest,path
        required=[str(HERE/x) for x in ('adapter.py','controls.py','protocol.json','source_manifest.json')]
        required += [p['prior_results'],p['prior_saved_review'],g['source_review']['path']]
        assert all(path in g['pins'] for path in required)
        sr=g['source_review']
        assert len(sr['publication'])>=8 and all(c in '0123456789abcdef' for c in sr['publication'])
        assert g['pins'][sr['path']]==sr['sha256']
        reviewed=read(sr['path'])
        assert reviewed['status']=='passed' and reviewed['source_review_complete'] is True
        for path in (str(HERE/'adapter.py'),str(HERE/'controls.py'),str(HERE/'protocol.json')):
            assert reviewed['pins'][path]==sha(path)
        prior_review=read(p['prior_saved_review'])
        assert prior_review['complete'] is True
        assert prior_review['pins'][p['prior_saved_review_result_key']]==sha(p['prior_results'])
        manifest=read(HERE/'source_manifest.json')
        for path,digest in manifest['pins'].items():assert sha(path)==digest,path
        r['runtime']=dict(executable=sys.executable,prefix=sys.prefix,numpy_version=np.__version__,numpy_file=np.__file__)
        r.update(protocol_sha256=sha(HERE/'protocol.json'),gate=g);save();assert r['runtime']==p['runtime']
        oldp=read(p['prior_protocol'])
        for path,digest in oldp['pins'].items():assert sha(path)==digest,path
        old=read(p['prior_results']);assert old['complete'] and len(old['rows'])==4
        old_helpers=module(p['prior_driver'],'prior_composition_helpers')
        qloader=module(oldp['q_loader'],'pinned_Q_loader');qenv,qtree=qloader.load(Path(oldp['q_source']).read_text());Q=qenv['Q']
        fixtures=read(oldp['q_fixtures']);toy=next(x for x in fixtures if x['name']=='zero_recurrent')
        model=tuple(np.array(toy[k],dtype=np.float64) for k in ('Wi','Wh','Wo'))
        setup=read(p['fixtures']);assert len(setup['cases'])==4
        prototypes=np.array(setup['prototypes'],dtype=np.float64)
        alloc=np.array(read(oldp['allocation'])['plain'],dtype=np.float64)
        src=Path(oldp['cv_source']).read_text()
        est=ObservedEstimator(src,Q,oldp['precompute'],alloc)
        prior_ast=ast.dump(ast.parse(read(p['prior_trees'])['composition']),include_attributes=False)
        assert est.numeric_ast_sha256==hashlib.sha256(prior_ast.encode()).hexdigest()==p['numeric_ast_sha256']
        assert qtree==read(p['prior_trees'])['Q']
        r['numeric_ast_sha256']=est.numeric_ast_sha256;event('result',stage='preflight');save()
        snapshots=[];cache=None
        for case,baseline in zip(setup['cases'],old['rows'],strict=True):
            cap();assert case['name']==baseline['name'];event('attempt',name=case['name'])
            stream=old_helpers.Stream(prototypes,case['delayed']);row=dict(name=case['name']);r['rows'].append(row);save()
            r['original_Q_estimator_calls']+=1;save()
            ws=[]
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');value=est(model,case['budget'],stream)
            finally:
                row.update(snapshot=None if est.last is None else asdict(est.last),requests=stream.requests,
                           warnings=sorted((x.category.__name__,str(x.message)) for x in ws));save()
            row.update(estimate_hex=float(value[0]).hex(),cost=value[1]);save()
            prior=baseline['arms'][0]
            row['parity']=all(row[k]==prior[k] for k in ['estimate_hex','cost','requests','warnings'])
            snap=est.last
            row['diagnostics_match']=snap.status=='complete' and snap.quota==tuple(baseline['nh']) and snap.kept==baseline['kept'] and snap.source_drawn==snap.completed_candidates==snap.requested_candidates==baseline['drawn'] and snap.reported_cost==baseline['raw_cost']
            for frozen,record in snapshots:assert asdict(frozen)==record
            snapshots.append((snap,asdict(snap)))
            if cache is None:cache=est._env['_pc']
            row['cache_identity_preserved']=est._env['_pc'] is cache
            save();event('result',name=case['name'],parity=row['parity'],diagnostics_match=row['diagnostics_match'])
            assert row['parity'] and row['diagnostics_match'] and row['cache_identity_preserved']
        # Frozen failure cases use synthetic Q or fail before Q; never original Q replay.
        for name,fail_at,q_fail,expected_requested,expected_completed,expected_kept in [
            ('rng_before_return',0,False,50000,0,None),
            ('rng_after_one_batch',1,False,100000,50000,None),
            ('Q_after_draw',None,True,50000,50000,584),
        ]:
            cap();event('attempt',name=name);q_calls=[0]
            def qstub(m,x):
                q_calls[0]+=1
                if q_fail:raise SentinelError('frozen Q failure')
                return np.zeros(len(x))
            observer=ObservedEstimator(src,qstub,oldp['precompute'],alloc)
            base=old_helpers.Stream(prototypes,fail_at==1)
            stream=base if fail_at is None else FailingStream(base,fail_at)
            caught=None
            try:observer(model,1000,stream)
            except Exception as e:caught=type(e).__name__
            snap=observer.last
            row=dict(name=name,caught=caught,snapshot=None if snap is None else asdict(snap),synthetic_Q_calls=q_calls[0],completed_requests=base.requests)
            r['exception_controls'].append(row);r['synthetic_Q_calls']+=q_calls[0];save()
            assert caught=='SentinelError' and snap.status=='failure' and snap.error_type=='SentinelError'
            assert (snap.requested_candidates,snap.completed_candidates,snap.kept)==(expected_requested,expected_completed,expected_kept)
            assert snap.reported_cost is None and observer._live is None
            assert snap.source_drawn==(50000 if q_fail else None) and q_calls[0]==int(q_fail)
            row['passed']=True;save();event('result',name=name,passed=True)
        # Reentrancy is rejected without replacing the outer observation. The
        # outer synthetic-Q result is a lifecycle fixture, not accuracy evidence.
        event('attempt',name='nested_and_immutable');holder={};nested=[]
        def nested_q(m,x):
            r['synthetic_Q_calls']+=1
            try:holder['est'](m,0,holder['stream'])
            except RuntimeError as e:nested.append(str(e))
            return np.zeros(len(x))
        observer=ObservedEstimator(src,nested_q,oldp['precompute'],alloc)
        stream=old_helpers.Stream(prototypes,False);holder.update(est=observer,stream=stream)
        value=observer(model,0,stream);snap=observer.last
        frozen_error=None
        try:snap.kept=999
        except Exception as e:frozen_error=type(e).__name__
        r['nested_control']=dict(snapshot=asdict(snap),nested_errors=nested,immutability_exception=frozen_error,
                                 estimate_hex=value[0].hex(),cost=value[1],requests=stream.requests);save()
        assert nested==['nested observation unsupported'] and frozen_error=='FrozenInstanceError'
        assert snap.status=='complete' and snap.kept==28 and snap.completed_candidates==50000 and len(stream.requests)==1
        assert observer._live is None
        for frozen,record in snapshots:assert asdict(frozen)==record
        event('result',name='nested_and_immutable',passed=True)
        assert len(r['rows'])==4 and len(r['exception_controls'])==3 and r['original_Q_estimator_calls']==4 and r['synthetic_Q_calls']==2
        r['complete']=True;event('complete');save()
    except BaseException as e:
        r.update(failure=repr(e),traceback=traceback.format_exc());event('failure',error=repr(e));save();raise

if __name__=='__main__':main()
