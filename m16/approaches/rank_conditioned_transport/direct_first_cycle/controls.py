from collections import defaultdict
from pathlib import Path
import json,time
import numpy as np
from scipy.special import ndtr
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator
import torch
from core import moving_knots,crossing_lines,direct_label_zero
from driver import estimate
from breakpoints import prepare
from engine import grids,fixed_target
from polynomial import conditional_expectations
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(primitive=[],aligned=[],constant=[],topology=[])
data=prepare(model,9);stats=data['stats'];wi,wh,_=model
axes=data['axes'][2];xx,yy=np.meshgrid(*axes,indexing='ij')
v=np.stack((.4+.2*np.sin(xx+yy),.5+.1*np.cos(xx-yy)),axis=-1)
interp=RegularGridInterpolator(axes,v,bounds_error=True)
for u in (-3.,-.13,0.,.17,2.9):
    q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)));h=np.maximum(wi*u,0.)
    lo,hi,_=conditional_expectations(model,h[None,:],axes,v,u,q,6.,stats,time.process_time()+5)
    base=wh@h;cuts=[-6.,u,6.]
    for d in range(2):cuts.extend(((axes[d]-base[d])/wi[d]).tolist())
    cuts=np.unique(np.clip(cuts,-6,6));numeric=np.zeros((2,2));calls=[0]
    for l,r in zip(cuts[:-1],cuts[1:]):
        side=int((l+r)/2>u)
        for k in range(2):
            def f(x):
                calls[0]+=1
                return float(interp(np.maximum(base+wi*x,0.))[0,k])*np.exp(-x*x/2)/np.sqrt(2*np.pi)
            numeric[side,k]+=quad(f,l,r,epsabs=1e-13)[0]
    numeric/=((1-2*ndtr(-6))*np.array([q,1-q]))[:,None]
    error=float(np.max(abs(numeric-np.stack((lo[0],hi[0])))));assert error<2e-12
    out['primitive'].append(dict(u=u,error=error,independent_integrand_calls=calls[0]))
# Target-first h1 equals an original stage1 node at each chosen u.
for u in [0.]+[sign*6*(1+1e-12)*(k/8)**2 for sign in (-1,1) for k in (1,3,5,6)]:
    q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)))
    direct,_=direct_label_zero(data,q,{},time.process_time()+5)
    old,_=fixed_target(model,0,q,data['axes'],data['states'],8,6.,stats,time.process_time()+5,'polynomial_piece_cached',{},data['piece_cache'],data['terminal_cache'])
    error=abs(direct-old);assert error<2e-11
    out['aligned'].append(dict(u=u,error=error))
# Shared-scale invariance at a fixed u.
scaled=(wi*1e-16,wh,model[2]);other=prepare(scaled,9)
q=.61;p,_=direct_label_zero(data,q,{},time.process_time()+5);r,_=direct_label_zero(other,q,{},time.process_time()+5)
out['scale_error']=abs(p-r);assert out['scale_error']<2e-11
for name,wo in (('zero',np.zeros_like(model[2])),('duplicate',np.tile([.3,-.7],(4,1)))):
    value,work=estimate((wi,wh,wo),3,2,True,seconds=10)
    error=abs(value-(1-work['omitted_target_mass_bound'])/4);assert error<1e-10
    out['constant'].append(dict(name=name,error=error,stats=work))
knots=moving_knots(model,axes,6.,4.,stats)
union=np.unique(np.r_[data['points'],knots]);assert len(union)-1<=2000
out['topology'].append(dict(fixed_panels=len(data['points'])-1,moving_panels=len(knots)-1,union_panels=len(union)-1))
# Check order at two probes inside every non-tiny moving topology panel.
checks=0
for l,r in zip(knots[:-1],knots[1:]):
    if r-l<1e-12:continue
    sign=1 if (l+r)/2>0 else -1
    lines=crossing_lines(model,axes,sign,6.)
    left=lines[:,0]*(.75*l+.25*r)+lines[:,1];right=lines[:,0]*(.25*l+.75*r)+lines[:,1]
    for i in range(len(lines)):
        for j in range(i):
            a,b=left[i]-left[j],right[i]-right[j]
            if min(abs(a),abs(b))>1e-11:assert a*b>=0
    checks+=1
out['line_order_panels_checked']=checks;out['work']=dict(stats);out['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(primitive_error=max(x['error'] for x in out['primitive']),aligned_error=max(x['error'] for x in out['aligned']),scale=out['scale_error'],constant=max(x['error'] for x in out['constant']),topology=out['topology'],cpu=out['cpu_seconds'])))
