"""Two frozen nonzero-prefix controls, no outer integration."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import torch
from terminal import MiddleTarget
from interpolation import InterpolatedMiddle
from controls import scalar_reference

HERE=Path(__file__).resolve().parent
start=time.process_time();path=Path('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth')
w=torch.load(path,map_location='cpu',weights_only=True)
arrays=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'))
model=arrays[0].ravel(),arrays[1],arrays[2]
prefixes=(-.25,.25);states=np.maximum(np.array(prefixes)[:,None]*model[0],0.)
old=InterpolatedMiddle(model,states);approx,error=old.values(1.);rows=[]
for i,(x0,h) in enumerate(zip(prefixes,states)):
    obj=MiddleTarget(model,h,1.);value,info=obj.probabilities();swap,_=obj.probabilities(swap=True)
    direct_start=time.process_time();direct,direct_error,warnings=scalar_reference(model,h,1.,2)
    rows.append(dict(prefix=x0,target=1.,need=1,hidden=h.tolist(),exact_pair=value.tolist(),
                     old_interpolated=approx[i].tolist(),old_minus_pair=(approx[i]-value).tolist(),
                     axes_max=float(np.max(abs(value-swap))),direct_scalar=direct.tolist(),
                     direct_max=float(np.max(abs(value-direct))),direct_error_diagnostic=direct_error,
                     direct_warnings=warnings,direct_cpu_seconds=time.process_time()-direct_start,
                     pair_info=info,pair_work=dict(obj.stats)))
    assert rows[-1]['axes_max']<1e-10 and rows[-1]['direct_max']<1e-7
out=dict(checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),domain=old.domain,
         old_work=dict(old.stats),old_penultimate_work=dict(old.penultimate.stats),
         old_kernel_error_diagnostic=error,cpu_seconds=time.process_time()-start,
         population_weighted_error=False)
(HERE/'nonzero_prefix.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('old_work','old_penultimate_work')}))
