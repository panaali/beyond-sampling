import json,time
from pathlib import Path
import numpy as np
import torch
from interpolation import InterpolatedPair
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
states=[]
for prefix in ((-1.,1.),(1.,-1.),(1.,1.)):
 h=np.zeros(2)
 for x in prefix:h=np.maximum(model[1]@h+model[0]*x,0.)
 states.append(h)
states=np.array(states);start=time.process_time();out={'tie_rows':[],'scale_rows':[]}
for name,wo in (('constant',np.zeros((4,2))),('duplicate',np.tile(model[2][0],(4,1)))):
 obj=InterpolatedPair((model[0],model[1],wo),states)
 for label in (0,1):
  for u in (-1.,0.,1.):
   value,error=obj.values(u,label);defect=float(np.max(abs(value-(1. if label==0 else 0.))));out['tie_rows'].append(dict(name=name,label=label,u=u,defect=defect));assert defect<1e-10
base=InterpolatedPair(model,states);scaled=InterpolatedPair((model[0]*1e-16,model[1],model[2]),states*1e-16)
for label in (0,1):
 for u in (-1.,0.,1.):
  x,_=base.values(u,label);y,_=scaled.values(u,label);defect=float(np.max(abs(x-y)));out['scale_rows'].append(dict(label=label,u=u,defect=defect));assert defect<1e-10
out['domain']=base.domain;out['total_cpu_seconds']=time.process_time()-start
(HERE/'interpolation_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['total_cpu_seconds'],ties=max(r['defect'] for r in out['tie_rows']),scale=max(r['defect'] for r in out['scale_rows']))))
