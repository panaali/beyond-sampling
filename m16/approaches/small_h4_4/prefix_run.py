"""Cycle3: exact prefix boundaries, bounded original-model convergence probe."""
import json,time
from pathlib import Path
import numpy as np
import torch
from estimator import angular_estimate,prefix_breaks
start=time.process_time();out=Path(__file__).with_name('prefix_results.json');data=dict(rules=[])
w=torch.load('/workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth',weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
for order,subdivision in ((2,1),(4,1),(8,1),(8,2)):
    row=dict(order=order,subdivision=subdivision)
    try:
        p,s=angular_estimate(base,order,boundary_mode='prefix',subdivision=subdivision,max_cpu=110-(time.process_time()-start))
        row.update(estimate=p,stats=s)
    except (ArithmeticError,TimeoutError) as e:row['error']=str(e)
    data['rules'].append(row);data['total_cpu']=time.process_time()-start
    out.write_text(json.dumps(data,indent=2)+'\n');print(order,subdivision,row.get('estimate',row.get('error')),flush=True)
