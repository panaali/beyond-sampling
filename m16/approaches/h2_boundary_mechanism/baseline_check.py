import json,time
from pathlib import Path
import numpy as np
from inventory import load,noisy
from last_input import calculate
BASE=Path(__file__).parent;start=time.process_time();models={};a,_=load(4);b,_=load(5)
rng=np.random.default_rng(47001);models['T4_copy0']=a
for i in range(1,9):models[f'T4_copy{i}']=tuple(w+.05*rng.normal(size=w.shape) for w in a)
for sigma in (0.,.0025,.02):models[f'T5_sigma{sigma}']=b if sigma==0 else noisy(b,sigma)
rng=np.random.default_rng(47021)
for sigma in (.02,.10):
 for i in range(4):models[f'T4_rms{sigma}_copy{i}']=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in a)
for scale in (1e-16,1e16):models[f'T4_scale{scale}']=(a[0]*scale,a[1],a[2]*scale)
for name,wo in (('zero',np.zeros_like(a[2])),('duplicate',np.ones_like(a[2]))):models['T4_'+name]=(a[0],a[1],wo)
for seed in range(1,5):models[f'T5_seed{seed}']=load(5,seed)[0]
out={'cases':[]}
for file in ('benchmark.json','transfer.json'):
 for original in json.loads((BASE/file).read_text())['cases']:
  name=original['name'];p,s=calculate(models[name]);truth=original['reference'];mse=(p-truth)**2;var=truth*(1-truth)
  row=dict(name=name,estimate=p,reference=truth,error=p-truth,stats=s,mse_ratio=mse*s['forward_equivalents']/var,
           mse_ratio_at_two_input_budget=mse*original['stats']['forward_equivalents']/var,
           two_input_error=original['reference_difference'],two_input_mse_ratio=original['mse_ratio'])
  out['cases'].append(row);print(name,p,row['error'],row['mse_ratio'],row['mse_ratio_at_two_input_budget'],flush=True)
out['cpu_seconds']=time.process_time()-start
(BASE/'baseline_check.json').write_text(json.dumps(out,indent=2)+'\n')
