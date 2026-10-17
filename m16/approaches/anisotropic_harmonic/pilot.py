"""Constant/tie controls and frozen smallest trained-model experiment."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from engine import estimate
HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
wi,wh,wo=(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=(wi.ravel(),wh,wo)
start=time.process_time()
result=dict(reference=.41017126172687207,controls=[],rows=[])
for kind,output in (('zero',np.zeros_like(wo)),('identical',np.ones_like(wo))):
    value,stats=estimate((model[0],wh,output),4,1,seconds=18-(time.process_time()-start))
    result['controls'].append(dict(kind=kind,error=value-1/3,stats=stats))
    assert abs(value-1/3)<1e-10
for N,M in ((4,0),(4,1),(8,1),(16,1),(8,2)):
    try:
        value,stats=estimate(model,N,M,seconds=18-(time.process_time()-start))
        row=dict(estimate=value,error=value-result['reference'],stats=stats)
    except (ArithmeticError,ValueError,TimeoutError) as error:
        row=dict(N=N,M=M,failure=str(error))
    result['rows'].append(row)
    result['total_cpu_seconds']=time.process_time()-start
    (HERE/'pilot_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    if 'failure' in row:break
