"""Cycle2: original refinement, scale/tie controls, preserved volume controls."""
import json,time
from pathlib import Path
import numpy as np
import torch
from estimator import angular_estimate
from stable_volume import mass
start=time.process_time()
output=Path(__file__).with_name('refinement.json');data=dict(controls=[],rules=[])
def save():
    data['total_cpu']=time.process_time()-start
    output.write_text(json.dumps(data,indent=2)+'\n')

for n in (8,16,20):
    p,b=mass(np.eye(4),n)
    data['controls'].append(dict(name='positive_orthant',order=n,estimate=p,truth=1/16,error=p-1/16,analytic_quadrature_bound=b))
    a=np.eye(4);a[:,0]=(a[:,0]+a[:,1])/2
    c=np.eye(4);c[:,1]=(c[:,0]+c[:,1])/2
    pa,ba=mass(a,n);pc,bc=mass(c,n)
    data['controls'].append(dict(name='split_positive_orthant',order=n,estimate=pa+pc,truth=1/16,error=pa+pc-1/16,analytic_quadrature_bound=ba+bc))
save()
w=torch.load('/workspace/clean-neurons/work/checkpoints/H4_T4_seed0.pth',weights_only=True,map_location='cpu')
base=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));base=(base[0].ravel(),*base[1:])
for name,m,order,offset in [('original16',base,16,0.),('original8_shifted',base,8,np.pi/16),('input_scale', (base[0]*1e-16,base[1],base[2]),2,0.),('output_scale',(base[0],base[1],base[2]*1e-16),2,0.),('zero_logits',(base[0],base[1],np.zeros_like(base[2])),2,0.)]:
    row=dict(name=name,order=order,offset=offset)
    try:
        p,s=angular_estimate(m,order,offset,max_cpu=100-(time.process_time()-start))
        row.update(estimate=p,stats=s)
    except (ArithmeticError,TimeoutError) as e:
        row['error']=str(e)
    data['rules'].append(row);save();print(name,row.get('estimate',row.get('error')),flush=True)
