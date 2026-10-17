"""Frozen same81-DOF root-support test, with model controls before scoring."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from engine import estimate

HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
start=time.process_time()
w=torch.load(path,weights_only=True,map_location='cpu')
wi=w['rnn.weight_ih_l0'].numpy().astype(float).ravel()
wh=w['rnn.weight_hh_l0'].numpy().astype(float)
wo=w['linear.weight'].numpy().astype(float)
model=wi,wh,wo
out=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('space.py','engine.py')},
    reference=.41017126172687207,controls=[],rows=[])

def save():
    out['total_cpu_seconds']=time.process_time()-start
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')

for name,readout in [('zero',np.zeros_like(wo)),('identical',np.ones_like(wo))]:
    value,stats=estimate((wi,wh,readout))
    out['controls'].append(dict(name=name,estimate=value,error=abs(value-1/3),stats=stats));save()
    if abs(value-1/3)>1e-10:raise ArithmeticError('constant readout failed')
for order in (64,128):
    value,stats=estimate(model,quadrature_order=order)
    out['rows'].append(dict(order=order,estimate=value,error=value-out['reference'],stats=stats));save()
    print(json.dumps(dict(order=order,estimate=value,error=value-out['reference'],cpu=stats['cpu_seconds'])),flush=True)
base=out['rows'][0]['estimate']
for scale in (np.array([.01,100.]),np.array([1e-16,1e-16])):
    transformed=(wi*scale,wh*scale[:,None]/scale[None,:],wo/scale[None,:])
    value,stats=estimate(transformed)
    out['controls'].append(dict(name='coordinate_scale',scale=scale.tolist(),estimate=value,error=abs(value-base),stats=stats));save()
    if abs(value-base)>1e-9:raise ArithmeticError('scale control failed')
out['complete']=True;save()
