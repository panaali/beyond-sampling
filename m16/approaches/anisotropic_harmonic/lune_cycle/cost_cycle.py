"""Fixed-rule equivalence and cold same-session hemisphere/lune comparison."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch
from engine import estimate
from space import Space

HERE=Path(__file__).resolve().parent

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

start=time.process_time();out=dict(primitive_controls=[],models=[])
s=Space(8,4,64);fast=Space(8,4,64,True)
for normal in ([1.,0.,0.],[-np.sqrt(3),1.,0.],[0.,0.,1.],[.6,-.7,.2],[1.,.1,.7],[-.1,.2,1e-6]):
    original=s.matrix(normal);optimized=fast.matrix(normal)
    error=float(np.max(abs(original-optimized)))
    out['primitive_controls'].append(dict(normal=normal,error=error))
    if error>1e-12:raise ArithmeticError('optimized matrix mismatch')
old_space=load_module('old_supported_space',HERE.parent/'space.py')
old_engine=load_module('old_supported_engine',HERE.parent/'engine.py')
old_engine.Space=old_space.Space  # imports are local; source/canonical code is unchanged
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth',weights_only=True,map_location='cpu')
h2=(w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),w['rnn.weight_hh_l0'].numpy().astype(float),w['linear.weight'].numpy().astype(float))
z=np.load('/workspace/clean-neurons/m16/approaches/arc_m43_repro/m43_weights.npz')
h4=tuple(z[k].astype(float) for k in ('Wi','Wh','Wo'));h4=(h4[0].ravel(),h4[1],h4[2])
for name,model,truth in [('H2T3',h2,.41017126172687207),('H4T3',h4,.9845503813585573)]:
    row=dict(name=name,reference=truth,estimates=[])
    value,stats=old_engine.estimate(model,8,4)
    row['estimates'].append(dict(kind='hemisphere',estimate=value,error=value-truth,stats=stats))
    for optimized,order in ((False,64),(True,64),(True,128)):
        value,stats=estimate(model,8,4,order,optimized=optimized)
        row['estimates'].append(dict(kind='optimized_lune' if optimized else 'lune',estimate=value,error=value-truth,stats=stats))
    delta=abs(row['estimates'][1]['estimate']-row['estimates'][2]['estimate'])
    row['fixed_rule_equivalence_error']=delta
    if delta>1e-11:raise ArithmeticError('optimized model mismatch')
    out['models'].append(row);out['cpu_seconds']=time.process_time()-start
    (HERE/'cost_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(name=name,estimates=[dict(kind=r['kind'],error=r['error'],cpu=r['stats']['cpu_seconds']) for r in row['estimates']])),flush=True)
