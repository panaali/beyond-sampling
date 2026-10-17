"""Frozen complete-panel comparisons of terminal mesh placement at h9."""
import hashlib
import json
from pathlib import Path
import time
import torch
from panel import estimate_panel
from spatial import terminal_mesh

HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time()
output=dict(reference=.30773074302392744,rows=[],
    checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                   for name in ('engine.py','polynomial.py','panel.py','breakpoints.py','spatial.py')})
for order in (2,4):
    for mode in ('curvature','gate'):
        try:
            value,stats=estimate_panel(model,9,order,'complete',
                seconds=100-(time.process_time()-start),mesh_builder=terminal_mesh(mode))
            row=dict(mode=mode,estimate=value,error=value-output['reference'],stats=stats)
        except (ValueError,ArithmeticError,TimeoutError) as error:
            row=dict(order=order,mode=mode,failure=str(error))
        output['rows'].append(row)
        output['total_cpu_seconds']=time.process_time()-start
        (HERE/'panel_mesh_results.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
        if 'failure' in row:raise SystemExit(1)
