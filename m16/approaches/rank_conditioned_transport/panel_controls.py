import json
from pathlib import Path
import time
import numpy as np
import torch
from engine import estimate
from breakpoints import prepare
from panel import density_nodes,stream,estimate_panel
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();deadline=start+18
old,oldstats=estimate(model,hidden_order=5,target_order=32,target_rule='split_value',target_value_limit=4,
    mesh_power=2.,transition='polynomial_piece_cached',terminal_reuse=True,seconds=deadline-time.process_time())
data=prepare(model,5,seconds=deadline-time.process_time())
q,weights,omitted=density_nodes(4,6.,4.,32,(np.array([-4.,0.]),np.array([0.,4.])))
new,contributions,bound=stream(data,q,weights,deadline)
assert abs(new-old)<1e-12
constant=(model[0],model[1],np.zeros_like(model[2]))
value,stats=estimate_panel(constant,hidden_order=3,order=4,seconds=deadline-time.process_time())
expected=(1-stats['omitted_target_mass_bound'])/4
assert abs(value-expected)<1e-8
output=dict(streaming_difference=new-old,streaming_work=dict(data['stats']),
            constant=dict(estimate=value,expected_retained=expected,error=value-expected,stats=stats),
            total_cpu_seconds=time.process_time()-start)
(HERE/'panel_controls_results.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(streaming_difference=new-old,constant_error=value-expected,total_cpu=output['total_cpu_seconds'])))
