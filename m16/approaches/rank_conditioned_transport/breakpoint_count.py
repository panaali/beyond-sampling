import json
from pathlib import Path
import time
import numpy as np
import torch
from breakpoints import prepare
HERE=Path(__file__).resolve().parent
p=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(p,map_location='cpu',weights_only=True)
wi,wh,wo=(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=wi.ravel(),wh,wo
start=time.process_time();rows=[]
for h in (9,17,33):
    data=prepare(model,h,seconds=18-(time.process_time()-start))
    row=dict(h=h,complete_panels=len(data['points'])-1,grid_panels=len(data['grid_points'])-1,
             terminal_panels=len(data['terminal_points'])-1,
             minimum_width=float(np.diff(data['points']).min()),
             maximum_width=float(np.diff(data['points']).max()),stats=dict(data['stats']))
    rows.append(row)
    print(json.dumps({k:v for k,v in row.items() if k!='stats'}),flush=True)
(HERE/'breakpoint_count_results.json').write_text(json.dumps(dict(rows=rows,total_cpu_seconds=time.process_time()-start),indent=2)+'\n')
