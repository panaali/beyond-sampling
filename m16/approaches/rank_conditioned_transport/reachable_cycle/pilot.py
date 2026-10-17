"""Frozen trained diagnostic; shared nodes across all six meshes."""
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
from diagnostic import evaluate

HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
weights=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(weights[k].numpy().astype(float) for k in (
    'rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time()
out=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    reference=.30773074302392744,rows=[],
    diagnostic_source_sha256=hashlib.sha256((HERE/'diagnostic.py').read_bytes()).hexdigest())
rules=[(8,8),(16,8)] if len(sys.argv)==1 else [(16,16)]
filename='pilot.json' if len(sys.argv)==1 else 'q_refinement.json'
for prefix,q in rules:
    try:result=evaluate(model,prefix,q,seconds=100-(time.process_time()-start))
    except Exception as exc:result=dict(prefix_order=prefix,q_order_per_half=q,failure=repr(exc))
    out['rows'].append(result);out['total_cpu_seconds']=time.process_time()-start
    (HERE/filename).write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in (
        'prefix_order','q_order_per_half','source_quadrature_estimate','cpu_seconds','failure')}),flush=True)
    if 'failure' in result:raise SystemExit(1)
    for r in result['rows']:
        print(json.dumps({k:r[k] for k in ('hidden_order','mode','signed','absolute','squared')}),flush=True)
