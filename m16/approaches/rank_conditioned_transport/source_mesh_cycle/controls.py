from collections import defaultdict
import hashlib,json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import betainc,betaincc,ndtr
import torch
from mesh import mixture_cdf,positive_cdf,source_mesh
from engine import grids
from breakpoints import prepare
from driver import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(marginal=[],grids=[],constant=[],topology=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for T,S in ((4,[0,1,2]),(4,[1,2]),(4,[0]),(7,[0,1,2,3,4,5,6])):
    # Independent condition on target q: target-first contributes its CDF;
    # a non-target first slot is above with1/(T-1), otherwise below.
    for x in (.1,.3,.5,.8):
        def integrand(q):
            below=min(x/q,1.) if q else 1.
            above=max((x-q)/(1-q),0.) if q<1 else 0.
            other=((T-2)*below+above)/(T-1)
            conditional=((float(q<=x) if j==0 else other) for j in S)
            return sum(conditional)/len(S)*T*(T-1)*q**(T-2)*(1-q)
        direct=quad(integrand,0,1,points=[x],epsabs=1e-13)[0]
        err=abs(direct-float(mixture_cdf(x,T,S)));assert err<1e-12
        out['marginal'].append(dict(T=T,labels=S,x=x,error=err))
    for sign in (-1,1):
        vals=positive_cdf(np.linspace(0,6,101),sign,T,S,6.)
        assert abs(vals[0])<1e-14 and abs(vals[-1]-1)<1e-14 and np.min(np.diff(vals))>=0
for mode in ('physical','reversed'):
    stats=defaultdict(float);axes,states,_=grids(model,9,6.,stats,2.)
    aa,ss,meta=source_mesh(mode)(model,axes,states,6.,stats)
    for k in range(len(axes)):
        if k!=1:assert all(np.array_equal(x,y) for x,y in zip(aa[k],axes[k]))
    scaled=(model[0]*1e-16,model[1],model[2]);ax,st,_=grids(scaled,9,6.,stats,2.)
    bb,_,_=source_mesh(mode)(scaled,ax,st,6.,stats)
    err=max(np.max(abs(x-y/1e-16)) for x,y in zip(aa[1],bb[1]))
    assert err<1e-12
    out['grids'].append(dict(mode=mode,scale_error=float(err),metadata=meta,work=dict(stats)))
    for name,wo in (('zero',np.zeros_like(model[2])),('duplicate',np.tile([.3,-.7],(4,1)))):
        p,stats=estimate((model[0],model[1],wo),3,2,mode,seconds=15.)
        expected=(1-stats['omitted_target_mass_bound'])/4
        err=abs(p-expected);assert err<1e-10
        out['constant'].append(dict(mode=mode,control=name,value=p,error=err,stats=stats))
for mode in ('baseline','physical','reversed'):
    data=prepare(model,9,seconds=10.,mesh_builder=None if mode=='baseline' else source_mesh(mode))
    out['topology'].append(dict(mode=mode,panels=len(data['points'])-1,work=dict(data['stats'])))
    assert len(data['points'])-1<=2000
out['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(max_marginal_error=max(r['error'] for r in out['marginal']),constant_error=max(r['error'] for r in out['constant']),topology=[(r['mode'],r['panels']) for r in out['topology']],cpu=out['cpu_seconds'])))
