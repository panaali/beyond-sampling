"""Pinned, read-only T75 dependencies and validation utilities."""
import hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
SOURCE=Path(os.environ.get('T90_SOURCE_ROOT','/workspace/work/admin-t90-claim'))
EXPECTED={'m16/approaches/understand_m43/est75.py':'e7af0d22877b85369b03617549be78aacf97a47852209ffa2d8467aae70e8dbd','m16/approaches/understand_m43/est_mech.py':'428cb90edc234b74626af47c076222d87a3112575d468e3715922375f9f1da63','m16/approaches/arc_m43_repro/exact.py':'8eeb865ddd2900fa0f390e86eaa05ddba87a6d16fa3f1b5913a57a9d98e8590c','m16/approaches/arc_m43_repro/m43.py':'e4cca6147b5962b709f5d91b0749ed8a25a6f11fdef6b8346b8a16e292e11887'}
for path,digest in EXPECTED.items():
    if hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()!=digest:raise RuntimeError(f'frozen source drift: {path}')
sys.path.insert(0,str(SOURCE/'m16/approaches/arc_m43_repro'));sys.path.insert(0,str(SOURCE/'m16/approaches/understand_m43'))
import est75 as E75
import exact as EX
from m43 import Ops,OPS_PER_SAMPLE
from zoo43 import noisy_copies
assert E75.RECIPE==dict(K=(4,4),rule='adaptive',eps=.1) and OPS_PER_SAMPLE==72

def model_hash(model):return hashlib.sha256(b''.join(np.ascontiguousarray(w).tobytes() for w in model)).hexdigest()

def load(power,seed):
    import torch
    state=torch.load(HERE/f'data/H4_T3_n2p{power}_seed{seed}.pth',map_location='cpu',weights_only=True)
    return tuple(state[k].detach().numpy().astype(float)[:,0] if i==0 else state[k].detach().numpy().astype(float) for i,k in enumerate(('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight')))

def estimate(model,**recipe):
    E75.STATS.update(cut=0,resolved=0,cells=0);ops=Ops();start=time.process_time();value=E75.est75(model,ops,**(recipe or E75.RECIPE));elapsed=time.process_time()-start
    return dict(value=value,operations=ops.n,stats=dict(E75.STATS),cpu_seconds=elapsed)

def truth(model,strict=True):
    ops=Ops();stats={};oc,of=EX.clip,EX.frac;start=time.process_time()
    if strict:EX.clip,EX.frac=E75.clip,E75.frac
    try:value=EX.exact_acc(model,ops,stats)
    finally:EX.clip,EX.frac=oc,of
    return dict(value=value,operations=ops.n,correct_leaves=stats['leaves'],cpu_seconds=time.process_time()-start)

def independent(model):
    sys.path.insert(0,str(HERE.parent/'compiled_geometry'))
    from geometry_native import geometry
    start=time.process_time();result=geometry(model);mass=[]
    for key in ('good','bad'):
        rays=result[key]
        if len(rays):
            numerator=abs(np.linalg.det(rays));denominator=1+np.sum(rays[:,0]*rays[:,1],1)+np.sum(rays[:,0]*rays[:,2],1)+np.sum(rays[:,1]*rays[:,2],1)
            mass.append(float(np.sum(2*np.arctan2(numerator,denominator)/(4*np.pi))))
        else:mass.append(0.)
    return dict(value=mass[0],partition=sum(mass),counts=result['counts'].tolist(),cpu_seconds=time.process_time()-start)

def manifest():
    rows=[]
    for power in (29,30):
        for seed in range(5):
            p=HERE/f'data/H4_T3_n2p{power}_seed{seed}.pth';rows.append(dict(power=power,seed=seed,url=f'https://storage.googleapis.com/arc-ml-public/alg/zoo/2nd_argmax_4_3_{2**power}_{seed}.pth',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    dependencies=dict(EXPECTED)
    for path in ('zoo43.py','eval75.json'):
        p=SOURCE/'m16/approaches/understand_m43'/path;dependencies[str(p.relative_to(SOURCE))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(checkpoints=rows,source_root=str(SOURCE),frozen_sources=dependencies,recipe=E75.RECIPE)
