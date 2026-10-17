"""Frozen cache/direct parity and complete-panel refinement gates."""
import hashlib
import json
from pathlib import Path
import time
import torch
from adapter import estimate

HERE=Path(__file__).resolve().parent
path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2][[0,1,3,2]]
start=time.process_time();out=dict(rows=[],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    fixture='original seed0 with readout rows2/3 swapped, not a new trained checkpoint')

def run(h,order,cached):
    row=dict(hidden_order=h,order=order,cached=cached)
    try:
        value,stats=estimate(model,h,order,cached,seconds=100-(time.process_time()-start))
        row.update(estimate=value,stats=stats)
    except Exception as exc:row['failure']=repr(exc)
    out['rows'].append(row);out['total_cpu_seconds']=time.process_time()-start
    (HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
    if 'failure' in row:raise SystemExit(1)
    return value

coarse=run(5,2,True);direct=run(5,2,False);fine=run(5,4,True)
out['h5_cache_parity']=abs(coarse-direct);out['h5_refinement_change']=abs(coarse-fine)
out['h9_gate_passed']=out['h5_cache_parity']<=1e-11 and out['h5_refinement_change']<=1e-7
if out['h9_gate_passed']:
    c9=run(9,2,True);f9=run(9,4,True)
    out['h9_refinement_change']=abs(c9-f9)
out['total_cpu_seconds']=time.process_time()-start
(HERE/'pilot.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
