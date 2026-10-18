"""Independent saved-record audit; no fitting or author summary execution."""
import collections,hashlib,json,math,time
from pathlib import Path
start=time.process_time();here=Path(__file__).parent
src=Path('/workspace/work/recover-small_h2_56/algzoo/constant_budget_multifamily')
summary=json.loads((src/'final_summary.json').read_text())
reported={(p['H'],p['seed'],r['family'],r['budget']):r for p in summary['panels'] for r in p['rows']}
stages=collections.Counter();attempts=collections.Counter();cpu=collections.defaultdict(list);checked_hashes=0;max_ratio=0.;steps=0
for H in (16,32):
 for seed in range(3):
  data={}
  for stage,prefix in [('original','results'),('qr','qr'),('active','active'),('correction','correction')]:
   f=src/f'{prefix}_H{H}_seed{seed}.json'
   if f.exists():
    d=json.loads(f.read_text());data[stage]=d
    cpu[stage].append(d['compute']['cpu_seconds'] if stage=='original' else d['cpu_seconds'])
    for name,sha in d.get('source_hashes',{}).items():
     assert hashlib.sha256((src/name).read_bytes()).hexdigest()==sha;checked_hashes+=1
  assert hashlib.sha256((src/f'results_H{H}_seed{seed}.json').read_bytes()).hexdigest()==data['qr']['original_sha256'];checked_hashes+=1
  indices={s:{(r['family'],r['budget']):r for r in d['rows']} for s,d in data.items()}
  expected={(f,k) for f in ('fourier','position_polynomial','rank_values') for k in (1,3,5,10,20)}
  assert set(indices['original'])==expected
  failed=expected
  for stage in ('original','qr','active','correction'):
   if not failed:assert stage not in indices or not indices[stage];continue
   assert stage in indices and set(indices[stage])==failed,(H,seed,stage,failed)
   remaining=set()
   for key in sorted(failed):
    row=indices[stage][key];sol=row['solver'];base=indices['original'][key]['solver'];threshold=1e-7*max(1.,base['initial_gradient_inf'])
    assert sol['threshold']==threshold
    grad=sol['gradient_inf'] if stage=='original' else sol['original_gradient_inf']
    assert math.isfinite(grad) and sol['converged']==(grad<=threshold)
    coef=data[stage]['coefficients'][f'{key[0]}_{key[1]}'];assert len(coef)==key[1] and all(len(v)==H for v in coef)
    for neuron in sol.get('neurons',[]):
     for step in neuron['steps']:assert step['loss']<=step['old_loss'];steps+=1
    attempts[stage]+=1
    if not sol['converged']:remaining.add(key);continue
    stages[stage]+=1;max_ratio=max(max_ratio,grad/threshold)
    target=reported[(H,seed,*key)]
    assert target['stage']==stage and target['accuracy']==row['replacement_accuracy'] and target['converged']
    assert abs(target['gap']-(row['replacement_accuracy']-.1)/(data['original']['model_accuracy']-.1))<1e-14
   failed=remaining
  assert not failed
assert len(reported)==90 and sum(stages.values())==90
out=dict(final_cells=90,first_passing_stage=dict(stages),attempts=dict(attempts),source_hash_checks=checked_hashes,maximum_final_gradient_threshold_ratio=max_ratio,nonincreasing_saved_steps=steps,cpu_by_stage={s:math.fsum(v) for s,v in cpu.items()},review_cpu=time.process_time()-start,scope='Saved records and source hashes only; no independent fit/gradient reexecution or global optimality claim.')
out['recorded_total_cpu']=math.fsum(out['cpu_by_stage'].values())
(here/'multifamily_final_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
