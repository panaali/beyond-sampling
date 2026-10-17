import json,time
from pathlib import Path
import numpy as np
from accounting import Audit,determinant,gauss_rule,volume,enumerate_correct
HERE=Path(__file__).resolve().parent
DD=Path('/workspace/clean-neurons/m16/approaches/exact_ladder/cones.py')
start=time.process_time();checks=[]
for n in (4,8,12,16,20):
 a=Audit();x,w=gauss_rule(n,a);xx,ww=np.polynomial.legendre.leggauss(n);xx=(xx+1)/2;ww/=2
 errors=[abs(float(w@(x**k))-1/(k+1)) for k in range(2*n)]
 assert max(errors)<2e-14
 assert np.max(np.abs(x-np.sort(xx)))<2e-14
 v,b=volume([np.eye(4)],n,a);assert abs(v-1/16)<=b+1e-14
 checks.append(dict(order=n,weight_sum=float(w.sum()),polynomial_error=max(errors),node_error=float(np.max(np.abs(x-xx))),weight_error=float(np.max(np.abs(w-ww))),orthant_mass=v,integration_bound=b,cost=a.report()))
rng=np.random.default_rng(52001);det_errors=[]
for i in range(20):
 m=rng.normal(size=(4,4))
 if i%2:m[:,3]=m[:,0]+1e-8*m[:,3]
 a=Audit();v=determinant(m,a);ref=np.linalg.det(m);det_errors.append(abs(v-ref));assert abs(v-ref)<1e-12
import torch
state=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
wi=state['rnn.weight_ih_l0'].numpy().astype(float).ravel();wh=state['rnn.weight_hh_l0'].numpy().astype(float)
ties=[]
for name,wo in [('zero',np.zeros((4,2))),('duplicate',np.ones((4,2)))]:
 a=Audit();rays,s=enumerate_correct((wi,wh,wo),DD,a);v,b=volume(rays,16,a)
 assert abs(v-.25)<b+1e-12
 ties.append(dict(name=name,accuracy=v,bound=b,simplices=len(rays),geometry=s,cost=a.report()))
out=dict(rule_checks=checks,max_det_error=max(det_errors),det_cases=len(det_errors),tie_checks=ties,cpu_seconds=time.process_time()-start)
(HERE/'checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(max_rule_error=max(c['polynomial_error'] for c in checks),max_det_error=max(det_errors),ties=[t['accuracy'] for t in ties],cpu=out['cpu_seconds'])))
