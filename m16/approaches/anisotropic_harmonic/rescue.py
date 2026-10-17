"""Frozen tangential rescue and corrected isotropic reference, no input draws."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from engine import estimate
HERE=Path(__file__).resolve().parent
REVIEW=Path('/workspace/work/recover-small_h2_910/m16/approaches/harmonic_negative_review')
sys.path.insert(0,str(REVIEW))
from diagnose import evaluate
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T3_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
wi,wh,wo=(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=(wi.ravel(),wh,wo)
start=time.process_time(); deadline=start+18
result=dict(reference=.41017126172687207,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            independent_evaluator_sha256=hashlib.sha256((REVIEW/'diagnose.py').read_bytes()).hexdigest(),rows=[])
for N,M,Q in ((4,2,None),(8,4,None),(8,8,None),(8,4,128)):
    try:
        value,stats=estimate(model,N,M,Q,seconds=deadline-time.process_time())
        row=dict(method='supported',estimate=value,error=value-result['reference'],stats=stats)
    except (ArithmeticError,ValueError,TimeoutError) as error:
        row=dict(method='supported',N=N,M=M,Q=Q,failure=str(error))
    result['rows'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    result['total_cpu_seconds']=time.process_time()-start
    (HERE/'rescue_results.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'failure' in row:raise SystemExit(1)
for D in (2,4,8):
    try:
        info=evaluate(model,D,True,True,deadline)
        row=dict(method='isotropic_corrected',D=D,estimate=info['estimate'],
                 error=info['estimate']-result['reference'],stats=info)
    except (ArithmeticError,ValueError,TimeoutError) as error:
        row=dict(method='isotropic_corrected',D=D,failure=str(error))
    result['rows'].append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    result['total_cpu_seconds']=time.process_time()-start
    (HERE/'rescue_results.json').write_text(json.dumps(result,indent=2)+'\n')
    if 'failure' in row:break
