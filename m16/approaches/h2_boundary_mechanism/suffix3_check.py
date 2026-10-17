"""Frozen 28-case validation; references only enter this scoring script."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from inventory import load,noisy
from suffix3 import calculate
BASE=Path(__file__).parent

def cases():
 models={};a,_=load(4);b,_=load(5)
 rng=np.random.default_rng(47001);models['T4_copy0']=a
 for i in range(1,9):models[f'T4_copy{i}']=tuple(w+.05*rng.normal(size=w.shape) for w in a)
 for sigma in (0.,.0025,.02):models[f'T5_sigma{sigma}']=b if sigma==0 else noisy(b,sigma)
 rng=np.random.default_rng(47021)
 for sigma in (.02,.10):
  for i in range(4):models[f'T4_rms{sigma}_copy{i}']=tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in a)
 for scale in (1e-16,1e16):models[f'T4_scale{scale}']=(a[0]*scale,a[1],a[2]*scale)
 for name,wo in (('zero',np.zeros_like(a[2])),('duplicate',np.ones_like(a[2]))):models['T4_'+name]=(a[0],a[1],wo)
 for seed in range(1,5):models[f'T5_seed{seed}']=load(5,seed)[0]
 refs=[]
 for file in ('benchmark.json','transfer.json'):refs+=json.loads((BASE/file).read_text())['cases']
 return [(r['name'],models[r['name']],r) for r in refs]

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 start=time.process_time();items=cases();out={'cases':[],'preparation_cpu':time.process_time()-start}
 for name,model,old in items:
  if args.pilot and name not in ('T4_copy0','T5_sigma0.0','T4_rms0.02_copy0'):continue
  row=dict(name=name,reference=old['reference'],reference_bound=old.get('reference_bound',0.),two_input_error=old['reference_difference'])
  try:
   p,s=calculate(model,8,True);q,r=calculate(model,16,True);truth=old['reference'];variance=truth*(1-truth)
   row.update(estimate=p,reference_difference=p-truth,refinement_difference=abs(p-q),refined_value=q,stats=s,
    refined_stats=r,mse_ratio=(p-truth)**2*s['forward_equivalents']/variance)
  except Exception as exc:row['failure']=repr(exc)
  out['cases'].append(row);out['cpu_seconds']=time.process_time()-start;out['peak_rss_kb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  (BASE/('suffix3_pilot.json' if args.pilot else 'suffix3_check.json')).write_text(json.dumps(out,indent=2)+'\n')
  print(name,row.get('estimate'),row.get('reference_difference'),row.get('mse_ratio'),row.get('refinement_difference'),row.get('stats',{}).get('triangles'),row.get('failure'),flush=True)
