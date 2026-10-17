import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from interpolation import InterpolatedPair
HERE=Path(__file__).resolve().parent
w=torch.load('/workspace/clean-neurons/work/checkpoints/H2_T4_seed0.pth',map_location='cpu',weights_only=True)
a=tuple(w[k].numpy().astype(float) for k in ('rnn.weight_ih_l0','rnn.weight_hh_l0','linear.weight'));model=a[0].ravel(),a[1],a[2]
reference=json.loads((HERE/'pilot.json').read_text());prefixes=[(-1.,1.),(1.,-1.),(1.,1.)]
states=np.array([next(r['hidden'] for r in reference['rows'] if tuple(r['prefix'])==prefix) for prefix in prefixes])
start=time.process_time();obj=InterpolatedPair(model,states);cache={};rows=[]
for ref in reference['rows']:
 key=(ref['u'],ref['label'])
 if key not in cache:cache[key]=obj.values(*key)
 values,bound=cache[key];value=values[prefixes.index(tuple(ref['prefix']))];truth=np.array(ref['values'])
 rows.append(dict(prefix=ref['prefix'],hidden=ref['hidden'],label=ref['label'],u=ref['u'],jointly_consistent_target=ref['jointly_consistent_target'],exact_pair=truth.tolist(),interpolated_pair=value.tolist(),signed_error=(value-truth).tolist(),interpolation_kernel_quadrature_only_bound=bound))
cycle=time.process_time()-start;out=dict(rows=rows,domain=obj.domain,work=dict(obj.stats),total_cpu_seconds=cycle,reference_pilot_sha256=hashlib.sha256((HERE/'pilot.json').read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((HERE/'interpolation.py').read_bytes()).hexdigest(),source_hashes={f:hashlib.sha256((HERE.parent/'rank_conditioned_transport'/f).read_bytes()).hexdigest() for f in ('engine.py','polynomial.py')},summaries=[])
for subset in ('all','jointly_consistent'):
 for label in (0,1):
  errors=np.array([r['signed_error'] for r in rows if r['label']==label and (subset=='all' or r['jointly_consistent_target'])])
  out['summaries'].append(dict(subset=subset,label=label,queries=len(errors),max_absolute_by_need=np.max(abs(errors),axis=0).tolist(),mean_absolute_by_need=np.mean(abs(errors),axis=0).tolist(),signed_mean_by_need=np.mean(errors,axis=0).tolist()))
(HERE/'interpolation_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu=cycle,summaries=out['summaries'])))
