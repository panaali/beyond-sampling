"""Proposal regression and public GL24 tie ownership checks."""
import json,time
from pathlib import Path
import numpy as np
import torch
from ..accounting import Audit
from .geometry import enumerate_correct
from . import estimator
HERE=Path(__file__).resolve().parent;start=time.process_time()
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T5_seed0.pth',weights_only=True,map_location='cpu');base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
original=json.loads((HERE/'results.json').read_text());rng=np.random.default_rng(1)
noise=tuple(x+rng.normal(size=x.shape)*np.sqrt(np.mean(x*x))*.0025 for x in base)
out=dict(geometry_regressions=[],tie_controls=[])
for name,model in [('base',base),('rms0.0025',noise)]:
 a=Audit();rays,stats,_=enumerate_correct(model,a);previous=next(r for r in original['models'] if r['name']==name)
 assert stats['gate_cells']==previous['geometry']['gate_cells'] and len(rays)==previous['correct_simplices']
 assert a.report()==previous['geometry_cost']
 out['geometry_regressions'].append(dict(name=name,identical_geometry_counts=True,identical_cost_ledger=True,gate_cells=stats['gate_cells'],correct_simplices=len(rays),cost=a.report()))
class ForbiddenRNG:
 def __getattribute__(self,name):raise AssertionError('randomness forbidden')
for name,wo in [('zero',np.zeros((5,2))),('duplicate',np.ones((5,2)))]:
 captured=[];real=estimator.calculate
 def capture(*args,**kwargs):
  value=real(*args,**kwargs);captured.append(value);return value
 estimator.calculate=capture
 try:p,used=estimator.est((base[0],base[1],wo),1,ForbiddenRNG())
 finally:estimator.calculate=real
 stats=captured[0][1]
 assert abs(p-.2)<stats['integration_only_bound']+1e-12
 assert used==stats['affine_only_forward_equivalents'] and used>1
 assert not stats['cost']['black_box_kernels'] and not stats['cost']['validation_kernel_calls']
 out['tie_controls'].append(dict(name=name,accuracy=p,expected=.2,budget=1,reported_forward_equivalents=used,stats=stats))
out['cpu_seconds']=time.process_time()-start;(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(geometry=[r['name'] for r in out['geometry_regressions']],ties=[r['accuracy'] for r in out['tie_controls']],cpu=out['cpu_seconds'])))
