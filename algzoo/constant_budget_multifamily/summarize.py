"""Independent saved-panel arithmetic and coefficient-count inventory."""
import json,math
from pathlib import Path
HERE=Path(__file__).parent
panels=[];failures=[]
for p in sorted(HERE.glob('results_H*_seed*.json')):
 d=json.loads(p.read_text());H=d['H'];n=d['n_test'];base=d['model_accuracy'];entry=dict(H=H,seed=d['seed'],model_accuracy=base,cells=len(d['rows']),cpu=d['compute']['cpu_seconds'],fits=[])
 for r in d['rows']:
  key=f"{r['family']}_{r['budget']}";coef=d['coefficients'][key]
  assert len(coef)==r['budget'] and all(len(x)==H for x in coef) and r['fitted_scalars']==H*r['budget']
  assert math.isclose((r['replacement_accuracy']-.1)/(base-.1),r['gap_recovered'],abs_tol=1e-14)
  assert r['passes_reconstruction_threshold']==(r['gap_recovered']>=.9)
  assert r['solver']['converged']==(r['solver']['gradient_inf']<=r['solver']['threshold'])
  assert sum(x>=.8 for x in r['r2'])==r['n_r2_at_least_0_8']
  count=round(n*r['replacement_accuracy']);assert abs(count/n-r['replacement_accuracy'])<1e-14
  entry['fits'].append(dict(family=r['family'],budget=r['budget'],accuracy=r['replacement_accuracy'],gap=r['gap_recovered'],converged=r['solver']['converged'],first_segment_mse=r['solver']['segments'][0]['mse'],final_mse=r['solver']['train_mse']))
  if not r['solver']['converged']:failures.append(dict(H=H,seed=d['seed'],family=r['family'],budget=r['budget'],gradient=r['solver']['gradient_inf'],threshold=r['solver']['threshold'],last_message=r['solver']['segments'][-1]['message']))
 panels.append(entry)
out=dict(panels=panels,unconverged=failures,fit_count=sum(p['cells'] for p in panels),converged_count=sum(p['cells'] for p in panels)-len(failures),recorded_cpu=sum(p['cpu'] for p in panels),complete_six_panels=len(panels)==6 and all(p['cells']==15 for p in panels))
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('panels','unconverged')})
for p in panels:print(p['H'],p['seed'],p['model_accuracy'],[(r['family'],round(r['accuracy'],6),round(r['gap'],6),r['converged']) for r in p['fits'] if r['budget']==20])
