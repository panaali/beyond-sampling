"""Frozen same-breakpoint q measure rescue; h5 GL2/4 only."""
import json
from pathlib import Path
import time
import torch
from adapter import estimate

HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2][[0,1,3,2]]
start=time.process_time();out=dict(rows=[])
for order in (2,4):
    row=dict(order=order)
    try:
        value,stats=estimate(model,5,order,True,seconds=100-(time.process_time()-start),target_rule='q')
        row.update(estimate=value,stats=stats)
    except Exception as exc:row['failure']=repr(exc)
    out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
    (HERE/'q_panels.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    if 'failure' in row:raise SystemExit(1)
out['refinement_change']=abs(out['rows'][0]['estimate']-out['rows'][1]['estimate'])
(HERE/'q_panels.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(refinement_change=out['refinement_change'],mass_defects=[r['stats']['raw_mass_defect_upper'] for r in out['rows']]),indent=2))
