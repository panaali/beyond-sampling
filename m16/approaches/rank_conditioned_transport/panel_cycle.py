"""Frozen h9 complete panels vs equal-count uniform panels, two GL orders."""
import json
from pathlib import Path
import time
import torch
from panel import estimate_panel
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time()
output=dict(reference=.30773074302392744,rows=[])
for order in (2,4):
    for scheme in ('complete','uniform'):
        try:
            value,stats=estimate_panel(model,9,order,scheme,seconds=100-(time.process_time()-start))
            row=dict(estimate=value,error=value-output['reference'],stats=stats)
        except (ValueError,ArithmeticError,TimeoutError) as error:
            row=dict(order=order,scheme=scheme,failure=str(error))
        output['rows'].append(row)
        output['total_cpu_seconds']=time.process_time()-start
        (HERE/'panel_results.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
        if 'failure' in row:raise SystemExit(1)
