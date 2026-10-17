"""Same operator, fixed vectorized angular assembly and cold full cost."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from adapter import estimate,dense_estimate
from operator_space import Space

HERE=Path(__file__).resolve().parent
start=time.process_time();out=dict(primitives=[],models=[])
s=Space(8,4,64);fast=Space(8,4,64,vectorized=True)
c=np.sin(np.arange(s.size)*.31+.4)/(1+np.arange(s.size))
for normal in ([0.,0.,1.],[.6,-.7,.2],[1.,.1,.7],[-.1,.2,1e-6],
               [1.,0.,0.],[-np.sqrt(3),1.,0.]):
    for sign in (1,-1):
        v=sign*np.array(normal)
        error=float(np.max(abs(s.cut(c,v)-fast.cut(c,v))))
        out['primitives'].append(dict(normal=v.tolist(),error=error))
        assert error<1e-12
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth',weights_only=True,map_location='cpu')
h2=(w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),w['rnn.weight_hh_l0'].numpy().astype(float),w['linear.weight'].numpy().astype(float))
z=np.load('/workspace/clean-neurons/m16/approaches/arc_m43_repro/m43_weights.npz')
h4=tuple(z[k].astype(float) for k in ('Wi','Wh','Wo'));h4=(h4[0].ravel(),h4[1],h4[2])
for name,model in [('H2T3',h2),('H4T3',h4)]:
    old,oldstats=dense_estimate(model)
    value,stats=estimate(model,vectorized=True)
    row=dict(name=name,dense=dict(estimate=old,stats=oldstats),
             real_operator=dict(estimate=value,stats=stats),difference=abs(value-old))
    out['models'].append(row)
    assert row['difference']<1e-9
    print(json.dumps(dict(name=name,difference=row['difference'],dense_cpu=oldstats['cpu_seconds'],
        real_cpu=stats['cpu_seconds'],dense_accounted_ops=oldstats['accounted_scalar_arithmetic_upper'],
        real_accounted_ops=stats['accounted_scalar_arithmetic_upper'])),flush=True)
out['total_cpu_seconds']=time.process_time()-start
(HERE/'vectorization.json').write_text(json.dumps(out,indent=2)+'\n')
