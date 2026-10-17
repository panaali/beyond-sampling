"""Frozen two-model matched-dimension basis and cost discriminator."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from driver import estimate,fourier_estimate

HERE=Path(__file__).resolve().parent
start=time.process_time()
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
w=torch.load(path,weights_only=True,map_location='cpu')
h2=(w['rnn.weight_ih_l0'].numpy().astype(float).ravel(),
    w['rnn.weight_hh_l0'].numpy().astype(float),w['linear.weight'].numpy().astype(float))
z=np.load('/workspace/clean-neurons/m16/approaches/arc_m43_repro/m43_weights.npz')
h4=tuple(z[k].astype(float) for k in ('Wi','Wh','Wo'));h4=(h4[0].ravel(),h4[1],h4[2])
out=dict(rows=[],controls=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('poly_space.py','driver.py')})
for name,model,truth in [('H2T3',h2,.41017126172687207),('H4T3',h4,.9845503813585573)]:
    row=dict(name=name,reference=truth)
    try:
        for kind,fn,order in (('fourier',fourier_estimate,64),('polynomial',estimate,64),('refinement',estimate,128)):
            value,stats=fn(model,quadrature_order=order,seconds=100-(time.process_time()-start))
            row[kind]=dict(estimate=value,error=value-truth,stats=stats)
        row['refinement_change']=abs(row['refinement']['estimate']-row['polynomial']['estimate'])
        old,new=row['fourier'],row['polynomial']
        row['absolute_error_ratio']=abs(new['error']/old['error'])
        row['squared_error_cpu_ratio']=(new['error']**2*new['stats']['cpu_seconds'])/(old['error']**2*old['stats']['cpu_seconds'])
        row['expansion_gate']=row['absolute_error_ratio']<1 and row['squared_error_cpu_ratio']<1
        for kind,changed in [('scaled',(model[0]*1e-16,model[1],model[2])),
                             ('zero',(model[0],model[1],np.zeros_like(model[2]))),
                             ('duplicate',(model[0],model[1],np.tile(model[2][0],(3,1))))]:
            check,checkstats=estimate(changed,seconds=100-(time.process_time()-start))
            target=new['estimate'] if kind=='scaled' else 1/3
            out['controls'].append(dict(name=name,kind=kind,error=abs(check-target),stats=checkstats))
            if abs(check-target)>1e-9:raise ArithmeticError('model control failed')
    except Exception as exc:row['failure']=repr(exc)
    out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('fourier','polynomial','refinement')}),flush=True)
    if 'failure' in row:raise SystemExit(1)
out['expansion_gate']=all(r['expansion_gate'] for r in out['rows'])
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
