"""Frozen broader reference-only transfer validation, with every case retained."""
import json,time,resource
from pathlib import Path
import numpy as np
from inventory import load
from suffix import calculate
BASE=Path(__file__).parent;ROOT=Path('/workspace/clean-neurons/m16/approaches')
start=time.process_time();base,_=load(4);rng=np.random.default_rng(47021);cases=[]
refs=json.loads((ROOT/'small_h2_34/stress_t4.json').read_text())['rows']
for sigma in (.02,.10):
 for i in range(4):cases.append((f'T4_rms{sigma}_copy{i}',tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base)))
for scale in (1e-16,1e16):cases.append((f'T4_scale{scale}',(base[0]*scale,base[1],base[2]*scale)))
for name,wo in (('zero',np.zeros_like(base[2])),('duplicate',np.ones_like(base[2]))):cases.append(('T4_'+name,(base[0],base[1],wo)))
cases=[(name,model,ref['reference'],ref['reference_bound']) for (name,model),ref in zip(cases,refs)]
for file in ('crossseed_1_2.json','crossseed_3_4.json'):
 for ref in json.loads((ROOT/'small_h2_56'/file).read_text())['models']:
  model,sha=load(5,ref['seed']);assert sha==ref['sha256'];integral=ref['integrals'][0]
  cases.append((f'T5_seed{ref["seed"]}',model,integral['estimate'],integral['conditional_quadrature_error_bound']))
out={'cases':[],'preparation_cpu':time.process_time()-start}
for name,model,reference,bound in cases:
 row=dict(name=name,reference=reference,reference_bound=bound)
 try:
  p,stats=calculate(model,16);q,refined=calculate(model,32)
  variance=reference*(1-reference)/stats['forward_equivalents']
  row.update(estimate=p,reference_difference=p-reference,refinement_difference=abs(p-q),stats=stats,
             mse_ratio=(p-reference)**2/variance,reference_ceiling_mse_ratio=(abs(p-reference)+bound)**2/variance)
 except Exception as exc:row['failure']=repr(exc)
 out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (BASE/'transfer.json').write_text(json.dumps(out,indent=2)+'\n')
 print(name,row.get('estimate'),row.get('reference_difference'),row.get('mse_ratio'),row.get('refinement_difference'),row.get('failure'),flush=True)
