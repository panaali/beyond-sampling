from collections import defaultdict
import hashlib,json,time,sys
from pathlib import Path
import mpmath as mp
import numpy as np
import torch
from scipy.special import ndtr
from native import NativeBasis
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport'))
from polynomial import interval_integrals,StagePieces
from engine import grids
start=time.process_time();stats=defaultdict(float);backend=NativeBasis(stats)
rows=[(c-h,c+h,.3,.6,ds,.15) for c in (0.,-1.,1.,-5.,5.) for h in (1e-10,1e-6,.1,1.) for ds in (-.1,.1)]
arr=np.array(rows).T;lo,hi,s,t,ds,dt=arr
corners=tuple(np.broadcast_to(np.eye(4)[k],(len(lo),4)) for k in range(4))
python_stats=defaultdict(float)
old,b0=interval_integrals(*arr[:2],corners,*arr[2:],python_stats)
new,b1=backend.integrate(*arr,stats)
error=float(np.max(abs(old-new)));assert error<=3e-12
mp.mp.dps=60;independent=[]
for index,(l,r,ss,tt,dd,ee) in enumerate(rows):
    l,r,ss,tt,dd,ee=map(mp.mpf,(l,r,ss,tt,dd,ee));c=(l+r)/2;h=(r-l)/2
    for k in range(4):
        def f(v):
            aa=ss+dd*v;bb=tt+ee*v
            value=((1-aa)*(1-bb),aa*(1-bb),(1-aa)*bb,aa*bb)[k]
            return h*value*mp.exp(-(c+h*v)**2/2)/mp.sqrt(2*mp.pi)
        exact=mp.quad(f,[-1,1]);err=float(abs(mp.mpf(new[index,k])-exact));assert err<=3e-12
        independent.append(dict(interval=index,basis=k,error=err,quadrature_only_bound=float(b1[index]),floating_error_exceeds_bound=bool(err>b1[index])))
for bad in ('zero_width','nan'):
    values=[x[:1].copy() for x in arr]
    if bad=='zero_width':values[1]=values[0].copy()
    else:values[0][0]=np.nan
    try:backend.integrate(*values,stats)
    except ArithmeticError:pass
    else:raise AssertionError('invalid input accepted')
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
axes,states,_=grids(model,9,6.,stats,2.);Native=backend.stage_class(StagePieces);matrix_rows=[]
for stage in (0,1,2):
    oldpieces=StagePieces(model,states[stage],axes[stage+1],6.,python_stats,time.process_time()+10)
    newpieces=Native(model,states[stage],axes[stage+1],6.,stats,time.process_time()+10)
    for u in (-3.,-.17,0.,.17,3.):
        q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)))
        aa,_=oldpieces.matrices(u,q,python_stats,time.process_time()+10);bb,_=newpieces.matrices(u,q,stats,time.process_time()+10)
        err=max(float(np.max(abs((x-y).toarray()))) for x,y in zip(aa,bb));assert err<=3e-12
        defect=max(float(np.max(abs(np.asarray(x.sum(axis=1))-1))) for x in bb)
        matrix_rows.append(dict(stage=stage,u=u,error=err,row_defect=defect))
# One fixed batch benchmark; this is not a model-input sweep.
big=[np.tile(x,1000) for x in arr];bcorners=tuple(np.broadcast_to(np.eye(4)[k],(len(big[0]),4)) for k in range(4))
a0=time.process_time();interval_integrals(*big[:2],bcorners,*big[2:],defaultdict(float));python_cpu=time.process_time()-a0
a0=time.process_time();backend.integrate(*big,defaultdict(float));native_cpu=time.process_time()-a0
out=dict(intervals=len(rows),native_python_error=error,independent=independent,matrices=matrix_rows,python_work=dict(python_stats),native_work=dict(stats),benchmark=dict(intervals=len(big[0]),python_cpu=python_cpu,native_cpu=native_cpu),cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256((HERE/'kernel.cpp').read_bytes()).hexdigest())
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(parity=error,independent_max=max(r['error'] for r in independent),matrix_max=max(r['error'] for r in matrix_rows),benchmark=out['benchmark'],cpu=out['cpu_seconds'])))
