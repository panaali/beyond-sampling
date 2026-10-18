"""Frozen target-byte extraction and two independent planar constructions only."""
from fractions import Fraction
from math import gcd
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import traceback

import core

HERE=Path(__file__).resolve().parent
RESULT=HERE/'target_k2_result.json'
PROGRESS=HERE/'target_k2_progress.jsonl'
PROTOCOL=HERE/'target_k2_protocol.json'
GATE=HERE/'target_k2_gate.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(obj):
    temp=RESULT.with_suffix('.tmp')
    with temp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,RESULT)
def event(phase,**kw):
    with PROGRESS.open('a') as f:
        f.write(json.dumps(dict(phase=phase,**kw),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def pair(x):return None if x is None else [x.numerator,x.denominator]
def primitive(x,y):
    g=gcd(abs(x),abs(y));assert g>0
    return x//g,y//g

def planar(wi,wh,D):
    """Independent Fraction slope arrangement; no core geometry/rank calls."""
    halves=[]
    for s in (-1,1):
        h=[max(s*w,0) for w in wi]
        alpha=[sum(w*x for w,x in zip(row,h,strict=True)) for row in wh]
        beta=[w*D for w in wi]
        roots=sorted(set(Fraction(-a,b) for a,b in zip(alpha,beta,strict=True) if b))
        bounds=[None]+roots+[None];rows=[]
        for lo,hi in zip(bounds[:-1],bounds[1:],strict=True):
            if lo is None and hi is None:r=Fraction(0)
            elif lo is None:r=hi-1
            elif hi is None:r=lo+1
            else:r=(lo+hi)/2
            bits=tuple(int(a+b*r>0) for a,b in zip(alpha,beta,strict=True))
            rays=[(0,-1) if lo is None else primitive(s*lo.denominator,lo.numerator),
                  (0,1) if hi is None else primitive(s*hi.denominator,hi.numerator)]
            hidden=tuple((s*a,b) if on else (0,0) for a,b,on in zip(alpha,beta,bits,strict=True))
            history=(tuple(int(s*w>0) for w in wi),bits)
            rows.append(dict(bounds=[pair(lo),pair(hi)],representative=pair(r),
                             history=history,rays=tuple(sorted(rays)),hidden=hidden))
        halves.append(dict(sign=s,alpha=alpha,beta=beta,roots=[pair(r) for r in roots],rows=rows))
    return halves

def main():
    assert not RESULT.exists() and not PROGRESS.exists() and not RESULT.with_suffix('.tmp').exists()
    p=json.loads(PROTOCOL.read_text());g=json.loads(GATE.read_text())
    assert g['clearance'] is True and g['protocol_sha256']==sha(PROTOCOL)
    for name,digest in {**p['pins'],**g['pins']}.items():assert sha(name)==digest,(name,'pin drift')
    assert p['requires_independent_control_saved_review'] is True
    prior_review=g['control_saved_review']
    assert len(prior_review['commit'])>=8 and all(c in '0123456789abcdef' for c in prior_review['commit'])
    assert g['pins'][prior_review['path']]==prior_review['sha256']==sha(prior_review['path'])
    review=json.loads(Path(prior_review['path']).read_text())
    assert review['complete'] is True and review['pass_all'] is True
    assert 0<resource.getrlimit(resource.RLIMIT_CPU)[0]<=20
    assert 0<resource.getrlimit(resource.RLIMIT_AS)[0]<=2147483648
    start=time.process_time();result=dict(status='running',protocol_sha256=sha(PROTOCOL),gate=g,
        model_calls=0,mass_calls=0,random_calls=0,checkpoint_decodes=0,
        resource_limits=dict(cpu=resource.getrlimit(resource.RLIMIT_CPU),address_space=resource.getrlimit(resource.RLIMIT_AS)))
    def cap():assert time.process_time()-start<16,'internal CPU cap'
    save(result);event('attempt',stage='provenance')
    try:
        import numpy as np
        import torch
        assert sys.version==p['python_version'] and np.__version__==p['numpy_version'] and torch.__version__==p['torch_version']
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        assert torch.get_num_threads()==torch.get_num_interop_threads()==1
        cap();result['import_cpu_seconds']=time.process_time()-start
        result['checkpoint_decodes']+=1;save(result)
        state=torch.load(p['checkpoint'],map_location='cpu',weights_only=True)
        expected={'rnn.weight_ih_l0':(16,1),'rnn.weight_hh_l0':(16,16),'linear.weight':(10,16)}
        assert set(state)==set(expected)
        meta=json.loads(Path(p['prior_metadata']).read_text())
        assert meta['checkpoint_sha256']==p['pins'][p['checkpoint']]
        result['checkpoint_arrays']={};converted={}
        for key,shape in expected.items():
            v=state[key]
            assert tuple(v.shape)==shape and v.dtype==torch.float32 and v.device.type=='cpu' and bool(torch.isfinite(v).all())
            digest=hashlib.sha256(v.detach().numpy().tobytes()).hexdigest()
            assert digest==meta['stored_weights'][key]['sha256']
            result['checkpoint_arrays'][key]=dict(shape=shape,dtype=str(v.dtype),sha256=digest)
            converted[key]=v.detach().numpy().astype(np.float64)
        arrays=dict(Wi=converted['rnn.weight_ih_l0'][:,0],Wh=converted['rnn.weight_hh_l0'],Wo=converted['linear.weight'])
        result['archive_arrays']={}
        with np.load(p['weights_archive'],allow_pickle=False) as old:
            assert set(old.files)==set(arrays)
            for name,v in arrays.items():
                o=old[name]
                assert o.dtype==np.float64 and o.shape==v.shape and o.tobytes()==v.tobytes()
                digest=hashlib.sha256(v.tobytes()).hexdigest()
                assert digest==meta['archives']['weights.npz']['arrays'][name]['bytes_sha256']
                result['archive_arrays'][name]=dict(shape=v.shape,dtype=str(v.dtype),sha256=digest)
        wi_rat=[float(x).as_integer_ratio() for x in arrays['Wi']]
        wh_rat=[[float(x).as_integer_ratio() for x in row] for row in arrays['Wh']]
        ratios=wi_rat+[x for row in wh_rat for x in row]
        D=max(b for a,b in ratios)
        assert D>0 and D&(D-1)==0 and all(D%b==0 and b&(b-1)==0 for a,b in ratios)
        wi=tuple(a*(D//b) for a,b in wi_rat)
        wh=tuple(tuple(a*(D//b) for a,b in row) for row in wh_rat)
        assert all(Fraction(x,D)==Fraction(a,b) for x,(a,b) in zip(wi,wi_rat,strict=True))
        assert all(Fraction(x,D)==Fraction(a,b) for row,rr in zip(wh,wh_rat,strict=True) for x,(a,b) in zip(row,rr,strict=True))
        result['weights']=dict(D=D,wi=wi,wh=wh,wi_ratios=wi_rat,wh_ratios=wh_rat,
                               checkpoint_sha256=sha(p['checkpoint']),archive_sha256=sha(p['weights_archive']))
        save(result);event('result',stage='provenance',coefficients=432,recurrence_coefficients=272)
        cap();event('attempt',stage='independent_planar')
        result['planar']=planar(wi,wh,D)
        independent=[r for half in result['planar'] for r in half['rows']]
        histories=[r['history'] for r in independent]
        assert len(histories)==len(set(histories))
        save(result);event('result',stage='independent_planar',count=len(independent))
        cap();event('attempt',stage='cone_k2');L=core.Ledger();result['cone_levels']=[]
        try:
            states=core.initial(wi,wh,D,L)
            for depth in (1,2):
                if depth==2:states=[c for S in states for c in core.extend(S,wi,wh,D,L)]
                level=dict(depth=depth,states=[core.encode(S) for S in states],certificates=[])
                result['cone_levels'].append(level);save(result)
                for S in states:cap();level['certificates'].append(core.certify_state(S,wi,wh,D,L))
        finally:
            result['cone_counts']=L.counts;result['max_retained_bits']=L.max_bits;save(result)
        actual={S.history:S for S in states};assert len(actual)==len(states)==len(independent)
        result['comparisons']=[]
        for ref in independent:
            S=actual[ref['history']]
            row=dict(history=S.history,rays_pass=S.C.rays==ref['rays'],hidden_pass=S.hidden==ref['hidden'])
            result['comparisons'].append(row)
            assert row['rays_pass'] and row['hidden_pass'],'independent planar/cone disagreement'
        result['count']=len(states);result['author_reported_k2']=34
        result['author_count_agrees']=len(states)==34
        result['status']='complete';event('result',stage='cone_k2',count=len(states),author_count_agrees=result['author_count_agrees'])
    except BaseException as e:
        result['status']='failure';result['failure']=repr(e);result['traceback']=traceback.format_exc()
        event('failure',error=repr(e));raise
    finally:
        result['cpu_seconds']=time.process_time()-start
        result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;save(result)
        print(json.dumps(dict(status=result['status'],cpu_after_final_save=time.process_time()-start)),flush=True)

if __name__=='__main__':main()
