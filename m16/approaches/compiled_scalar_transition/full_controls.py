import json,time
from pathlib import Path
import numpy as np
import torch
from driver import estimate
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=a[0].ravel(),a[1],a[2]
start=time.process_time();out=dict(rows=[])
for name,m in [('zero',(model[0],model[1],np.zeros_like(model[2]))),('duplicate',(model[0],model[1],np.tile([.3,-.7],(4,1)))),('base',model),('scaled',(model[0]*1e-16,model[1],model[2]))]:
    p,stats=estimate(m,3,2,True,seconds=15.)
    if name in ('zero','duplicate'):assert abs(p-(1-stats['omitted_target_mass_bound'])/4)<1e-10
    out['rows'].append(dict(name=name,value=p,stats=stats))
assert abs(out['rows'][2]['value']-out['rows'][3]['value'])<3e-12
p,stats=estimate(model,3,2,False,seconds=15.)
out['python']=dict(value=p,stats=stats)
out['base_difference']=abs(p-out['rows'][2]['value']);assert out['base_difference']<3e-12
out['scale_difference']=abs(out['rows'][2]['value']-out['rows'][3]['value'])
out['cpu_seconds']=time.process_time()-start
(HERE/'full_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(parity=out['base_difference'],scale=out['scale_difference'],cpu=out['cpu_seconds'])))
