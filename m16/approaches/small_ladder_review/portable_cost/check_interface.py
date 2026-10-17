"""Bounded-rank production controls: no SVD validation, sampling or RNG access."""
import json,time
from pathlib import Path
import numpy as np
import torch
import estimator
HERE=Path(__file__).resolve().parent
class ForbiddenRNG:
 def __getattribute__(self,name):raise AssertionError('RNG must not be accessed')
start=time.process_time();rows=[]
state=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',weights_only=True,map_location='cpu')
wi=state['rnn.weight_ih_l0'].numpy().astype(float).ravel();wh=state['rnn.weight_hh_l0'].numpy().astype(float)
for name,wo in [('zero',np.zeros((4,2))),('duplicate',np.ones((4,2)))]:
 model=(wi,wh,wo);capture=[];real_calculate=estimator.calculate
 def save(*args,**kwargs):
  result=real_calculate(*args,**kwargs);capture.append(result);return result
 estimator.calculate=save
 p,used=estimator.est(model,1,ForbiddenRNG())
 estimator.calculate=real_calculate
 stats=capture[0][1]
 assert abs(p-.25)<stats['integration_only_bound']+1e-12
 assert used>1 and used==stats['affine_only_forward_equivalents']
 assert not stats['cost']['black_box_kernels'] and not stats['cost']['validation_kernel_calls']
 rows.append(dict(control=name,accuracy=p,advisory_budget=1,returned_cost=used,stats=stats))
out=dict(controls=rows,cpu_seconds=time.process_time()-start)
(HERE/'interface_checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['cpu_seconds'],accuracy=[r['accuracy'] for r in rows],cost=[r['returned_cost'] for r in rows])))
