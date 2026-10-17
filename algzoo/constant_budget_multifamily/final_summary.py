"""Audit prescribed original->QR->active chain; never select by test accuracy."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).parent
correction_path=HERE/'correction_H32_seed2.json';correction=json.loads(correction_path.read_text());correction_rows={(r['family'],r['budget']):r for r in correction['rows']}
for name,sha in correction['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
correction_count=0
panels=[];converged=0;count=0;cpu=0.;passes=0;steps=0;maximum_increase=0.;unresolved=[]
for p in sorted(HERE.glob('results_H*_seed*.json')):
 old=json.loads(p.read_text());H,seed=old['H'],old['seed'];qp=HERE/f'qr_H{H}_seed{seed}.json';q=json.loads(qp.read_text());qr={(r['family'],r['budget']):r for r in q['rows']};ap=HERE/f'active_H{H}_seed{seed}.json';a=json.loads(ap.read_text()) if ap.exists() else None;active={(r['family'],r['budget']):r for r in a['rows']} if a else {}
 if a:
  for name,sha in a['source_hashes'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
  cpu+=a['cpu_seconds'];passes+=a['hidden_forward_sequences']
 entry=dict(H=H,seed=seed,source_accuracy=old['model_accuracy'],rows=[])
 for r0 in old['rows']:
  key=r0['family'],r0['budget'];r=r0;stage='original'
  if not r['solver']['converged']:
   r=qr[key];stage='qr'
   if not r['solver']['converged']:
    if key not in active:continue
    r=active[key];stage='active';count+=1;assert r['solver']['threshold']==r0['solver']['threshold'];assert r['solver']['converged']==(r['solver']['original_gradient_inf']<=r0['solver']['threshold']);coef=a['coefficients'][f'{key[0]}_{key[1]}'];assert len(coef)==key[1] and all(len(z)==H for z in coef)
    for neuron in r['solver']['neurons']:
     for step in neuron['steps']:assert step['loss']<=step['old_loss'];steps+=1
    maximum_increase=max(maximum_increase,r['solver']['global_mse_increase'])
  if not r['solver']['converged'] and (H,seed)==(32,2):
   r=correction_rows[key];stage='correction';correction_count+=1;assert r['solver']['threshold']==r0['solver']['threshold'];assert r['solver']['converged']==(r['solver']['original_gradient_inf']<=r0['solver']['threshold'])
   coef=correction['coefficients'][f'{key[0]}_{key[1]}'];assert len(coef)==key[1] and all(len(z)==H for z in coef)
   for neuron in r['solver']['neurons']:
    for step in neuron['steps']:assert step['loss']<=step['old_loss']
  ok=r['solver']['converged'];converged+=ok
  if not ok:unresolved.append(dict(H=H,seed=seed,family=key[0],budget=key[1],stage=stage,gradient=r['solver']['original_gradient_inf']))
  assert math.isclose(r['gap_recovered'],(r['replacement_accuracy']-.1)/(old['model_accuracy']-.1),abs_tol=1e-14)
  entry['rows'].append(dict(family=key[0],budget=key[1],stage=stage,accuracy=r['replacement_accuracy'],accuracy_ci=r['accuracy_ci'],gap=r['gap_recovered'],r2_count=r['n_r2_at_least_0_8'],converged=ok))
 panels.append(entry)
out=dict(panels=panels,final_converged=converged,final_cells=sum(len(p['rows']) for p in panels),active_cells=count,unresolved=unresolved,correction_cells=correction_count,correction_cpu=correction['cpu_seconds'],correction_hidden_passes=correction['hidden_forward_sequences'],complete=count==14 and correction_count==3 and sum(len(p['rows']) for p in panels)==90,active_cpu=cpu,active_hidden_passes=passes,active_least_squares_solves=steps,maximum_full_matrix_mse_increase=maximum_increase)
(HERE/'final_summary.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k not in ('panels','unresolved')})
for p in panels:print(p['H'],p['seed'],[(r['family'],r['accuracy'],r['gap'],r['converged']) for r in p['rows'] if r['budget']==20])
