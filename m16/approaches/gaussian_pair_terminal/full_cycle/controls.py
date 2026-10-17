import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtr
from driver import estimate,setup,pair_label
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
start=time.process_time();out={'constant_rows':[],'scale_rows':[]}
for name,wo in (('constant',np.zeros((4,2))),('duplicate',np.tile(model[2][0],(4,1)))):
 for paired in (False,True):
  try:value,stats=estimate((model[0],model[1],wo),2,paired,seconds=17.-(time.process_time()-start))
  except Exception as exc:
   out['failure']=dict(name=name,paired=paired,error=repr(exc),cpu_seconds=time.process_time()-start)
   (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
   raise
  error=value-stats['raw_target_mass']/4
  out['constant_rows'].append(dict(name=name,paired=paired,value=value,error=error,stats=stats))
  (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
  assert abs(error)<1e-10
base=setup(model);scaled=setup((model[0]*1e-16,model[1],model[2]));deadline=time.process_time()+5.
for u in (-1.,0.,1.):
 q=float((ndtr(u)-ndtr(-6))/(1-2*ndtr(-6)))
 for label in (0,1):
  x,xe=pair_label(base,q,label,{},deadline);y,ye=pair_label(scaled,q,label,{},deadline)
  error=abs(x-y);out['scale_rows'].append(dict(u=u,label=label,base=x,scaled=y,error=error));assert error<1e-10
out['total_cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=out['total_cpu_seconds'],constant=max(abs(r['error']) for r in out['constant_rows']),scale=max(r['error'] for r in out['scale_rows']))))
