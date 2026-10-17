"""Exact compatibility gate for scalar versus batched radial setup."""
from pathlib import Path
from collections import defaultdict
import hashlib,itertools,json,math,sys,time
import numpy as np
SOURCE=Path('/workspace/work/recover-small_h2_56/m16/approaches/batched_triangle_fallback')
assert hashlib.sha256((SOURCE/'batch.py').read_bytes()).hexdigest()=='c597c7f058992d8f34b6c5c5950ccbc586b168e84714f227c039f49adeb9390c'
assert hashlib.sha256(Path('/workspace/clean-neurons/m16/approaches/small_ladder_review/portable_cost/t6/triangle.py').read_bytes()).hexdigest()=='454a72f7657d4098efa1fca1a3c62065a5ee449c8418af3b9b3f79df5e7a45e9'
sys.path.insert(0,str(SOURCE))
from batch import Grouped,Engine,EDGE
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule


def cases():
    out=[];eye=np.eye(3,6);anchor=.2*np.ones(6)/6+.8*np.eye(6)[0]
    for width in (1.,.1,.01,.001,.0001,.00001,.0000001,0.):
        b=np.ones((3,6))+width*eye;b/=b.sum(axis=1)[:,None]
        out.extend(b[list(p)] for p in itertools.permutations(range(3)))
    for width in (.1,.01,.001,.0001,.00001,.0000001,0.):
        b=np.stack([anchor,anchor+width*(eye[1]-eye[0]),anchor+width*(eye[2]-eye[0])])
        out.extend(b[list(p)] for p in itertools.permutations(range(3)))
    out.extend([eye,np.stack([eye[0],eye[0],eye[1]]),np.tile(anchor,(3,1))])
    fixture=SOURCE.parent/'directed_chebyshev/t6_fixture.json'
    assert hashlib.sha256(fixture.read_bytes()).hexdigest()=='736986404087b1d98e902d470d14ddc4e800b43e1437cad812a249fc5fc2b040'
    rays=np.array(json.loads(fixture.read_text())['rays'])
    for n in (4,8):
        y,rest,w=outer_rule(n,Audit())
        for ray in rays:
            prefix=y@ray[:,:3].T
            out.extend(prefix[:,None,:]+rest[:,None,None]*ray[:,3:].T[None,:,:])
    return np.asarray(out)


def scalar_radial(v):
    center=v.sum(axis=0);center/=math.sqrt(float(center@center))
    lo=float(np.min(v@center))**2
    signs=np.sign(v.sum(axis=0));projection=float(np.min(v@signs))
    if projection>0:lo=max(lo,projection*projection/float(signs@signs))
    hi=float(np.max(np.sum(v*v,axis=1)));lo=min(lo,hi)
    return lo,hi,(lo+hi)/2,(hi-lo)/(hi+lo)


def vector_radial(v):
    center=v.sum(axis=1)
    norm=np.sqrt(np.matmul(center[:,None,:],center[:,:,None])[:,0,0])
    center=center/norm[:,None]
    lo=np.min(np.matmul(v,center[:,:,None])[:,:,0],axis=1)**2
    signs=np.sign(v.sum(axis=1));projection=np.min(np.matmul(v,signs[:,:,None])[:,:,0],axis=1)
    signsq=np.matmul(signs[:,None,:],signs[:,:,None])[:,0,0]
    lo=np.where(projection>0,np.maximum(lo,projection*projection/signsq),lo)
    hi=np.max(np.sum(v*v,axis=2),axis=1);lo=np.minimum(lo,hi)
    return np.stack((lo,hi,(lo+hi)/2,(hi-lo)/(hi+lo)),axis=1)


if __name__=='__main__':
    start=time.process_time();v=cases();scalar=np.array([scalar_radial(x) for x in v]);vector=vector_radial(v)
    mismatch=np.flatnonzero(np.any(scalar!=vector,axis=1))
    out=dict(cases=len(v),mismatches=len(mismatch),component_mismatches=np.sum(scalar!=vector,axis=0).tolist(),
             maximum_absolute_difference=np.max(abs(scalar-vector),axis=0).tolist(),
             examples=[dict(index=int(i),scalar=scalar[i].tolist(),vector=vector[i].tolist()) for i in mismatch[:16]],
             source_sha256={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in ('batch.py','check.py','pilot.py')},
             cpu_seconds=time.process_time()-start,exact_radial_gate_pass=len(mismatch)==0)
    (Path(__file__).resolve().parent/'radial_probe.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='examples'}))
