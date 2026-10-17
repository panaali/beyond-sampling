import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from cone_estimator import cone_estimate
p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--geometry',required=True);a=p.parse_args()
w=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
start=time.process_time();data=dict(controls=[]);out=Path(__file__).with_name('cone_controls.json')
for name,model,truth in [('input_scale',(base[0]*1e-16,base[1],base[2]),.4770131259941187),('output_scale',(base[0],base[1],base[2]*1e-16),.4770131259941187),('zero_logits',(base[0],base[1],np.zeros_like(base[2])),.25),('identical_logits',(base[0],base[1],np.tile(base[2][0],(4,1))),.25)]:
    value,stats=cone_estimate(model,20,a.geometry,max_cpu=110-(time.process_time()-start),check_partition=True)
    data['controls'].append(dict(name=name,estimate=value,reference=truth,difference=value-truth,stats=stats))
    data['total_cpu']=time.process_time()-start;out.write_text(json.dumps(data,indent=2)+'\n');print(name,value,value-truth,flush=True)
