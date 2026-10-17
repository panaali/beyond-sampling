from collections import defaultdict
from pathlib import Path
import sys,time,json
import numpy as np
from scipy.special import ndtr
import torch
from cache import PenultimateCache
from driver import estimate
from engine import terminal,grids
from breakpoints import prepare
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(rows=[]);states=np.array([[0.,0.],[.1,0.],[0.,.2],[.3,.2]])
cases=[('trained',model),('zero',(model[0],model[1],np.zeros_like(model[2]))),('duplicate',(model[0],model[1],np.tile([.3,-.7],(4,1)))),('Wh0',(model[0],np.zeros((2,2)),model[2]))]
for name,m in cases:
    cache=PenultimateCache(m,states);uvals=np.unique(np.r_[cache.knots,-3.,-.13,.17,2.9])
    errors=[];stats=defaultdict(float)
    for u in uvals:
        q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)))
        actual=cache.values(float(u),q)
        h3=np.maximum(states@m[1].T+m[0]*u,0.)
        wanted=np.array([terminal(m,h,float(u),q,2,6.,stats) for h in h3])
        error=float(np.max(abs(actual-wanted)));errors.append(error)
        if error>1e-10:
            out['failure']=dict(name=name,u=float(u),error=error,actual=actual.tolist(),wanted=wanted.tolist())
            (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');raise ArithmeticError(out['failure'])
    out['rows'].append(dict(name=name,queries=len(uvals),max_error=max(errors),work=dict(cache.stats)))
cache=PenultimateCache(model,states)
scaled=(model[0]*1e-16,model[1],model[2]);other=PenultimateCache(scaled,states*1e-16)
out['scale_error']=max(float(np.max(abs(cache.values(u)-other.values(u)))) for u in (-3.,-.13,0.,.17,2.9));assert out['scale_error']<1e-10
for name,m in cases[1:3]:
    value,stats=estimate(m,3,2,True,seconds=10)
    err=abs(value-(1-stats['omitted_target_mass_bound'])/4);assert err<1e-10
    out[name+'_full']=dict(error=err,stats=stats)
data=prepare(model,9);cache=PenultimateCache(model,data['states'][2]);union=np.unique(np.r_[data['points'],cache.knots])
out['topology']=dict(fixed_panels=len(data['points'])-1,cache_knots=len(cache.knots),cache_pieces=len(cache.left),union_panels=len(union)-1,work=dict(cache.stats))
assert len(union)-1<=2000
out['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(max_error=max(r['max_error'] for r in out['rows']),scale_error=out['scale_error'],topology=out['topology'],cpu=out['cpu_seconds'])))
