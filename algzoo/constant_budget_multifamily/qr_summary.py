"""Audit rescue coverage and publish all fixed final curves without test selection."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).parent
panels=[];count=0;passed=0;failures=[];cpu=0.;modelpasses=0;maxtransform=0.;objectivecalls=0
for p in sorted(HERE.glob('results_H*_seed*.json')):
 old=json.loads(p.read_text());H,seed=old['H'],old['seed'];qpath=HERE/f'qr_H{H}_seed{seed}.json'
 if not qpath.exists():continue
 q=json.loads(qpath.read_text());assert q['original_sha256']==hashlib.sha256(p.read_bytes()).hexdigest();qr={(r['family'],r['budget']):r for r in q['rows']};entry=dict(H=H,seed=seed,source_accuracy=old['model_accuracy'],rows=[])
 for original in old['rows']:
  key=original['family'],original['budget'];r=original
  if not original['solver']['converged']:
   if key not in qr:continue
   r=qr[key];count+=1;solver=r['solver'];assert solver['threshold']==original['solver']['threshold'];assert solver['converged']==(solver['original_gradient_inf']<=solver['threshold']);assert solver['rank']==original['budget'];assert abs(solver['initial_mse']-original['solver']['train_mse'])<1e-10
   assert math.isclose(r['gap_recovered'],(r['replacement_accuracy']-.1)/(old['model_accuracy']-.1),abs_tol=1e-14)
   coefficients=q['coefficients'][f'{key[0]}_{key[1]}'];assert len(coefficients)==key[1] and all(len(x)==H for x in coefficients)
   if solver['converged']:passed+=1
   else:failures.append(dict(H=H,seed=seed,family=key[0],budget=key[1],solver=solver))
   maxtransform=max(maxtransform,solver['initial_transform_error'],solver['final_transform_error']);objectivecalls+=solver['objective_evals']+solver['original_gradient_evaluations']
  entry['rows'].append(dict(family=key[0],budget=key[1],accuracy=r['replacement_accuracy'],accuracy_ci=r['accuracy_ci'],gap=r['gap_recovered'],converged=r['solver']['converged'],n_r2_at_least_0_8=r['n_r2_at_least_0_8'],mean_r2=sum(r['r2'])/len(r['r2']),rescued=not original['solver']['converged']))
 cpu+=q['cpu_seconds'];modelpasses+=q['hidden_forward_sequences'];panels.append(entry)
out=dict(panels=panels,rescue_count=count,rescue_converged=passed,unresolved=failures,complete=count==25 and len(panels)==6 and all(len(p['rows'])==15 for p in panels),additional_cpu=cpu,additional_hidden_passes=modelpasses,max_transform_prediction_discrepancy=maxtransform,additional_objective_gradient_evaluations=objectivecalls)
(HERE/'qr_summary.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('panels','unresolved')})
for p in panels:print(p['H'],p['seed'],[(r['family'],r['accuracy'],r['gap'],r['converged']) for r in p['rows'] if r['budget']==20])
