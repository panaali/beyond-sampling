"""Frozen seed1 exact/recipe attribution with independent positive references."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/final3')
def collect(mode):
 folder=HERE/mode;capture=json.loads((folder/'error_capture.json').read_text());assert capture['complete'];parents=np.load(folder/'error_captured.npz')['parent_ids'];N=len(parents);rows=[];meta=[]
 for b in range(8):
  d=json.loads((folder/f'batch_{b}.json').read_text());assert d['complete'];meta.append(d);a=np.load(folder/f'batch_{b}.npz');idx=a['indices'];assert np.all((parents[idx]>=12*b)&(parents[idx]<12*(b+1)))
  for p,h in d['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
  rows.extend(zip(*(a[k].tolist() for k in ('indices','source','doubled','positive','bounds','orders','acute'))))
 rows.sort();assert [r[0] for r in rows]==list(range(N));old,new,positive,bound=[math.fsum(r[j] for r in rows) for j in (1,2,3,4)];oldops=sum(d['source_mass_named_ops'] for d in meta);newops=sum(d['doubled_mass_named_ops'] for d in meta);assert newops==2*oldops-156*N;geometry=capture['named_geometry_ops']-16*N-16+8*96
 fn=PRIOR/('exact_H8_T4.json' if mode=='exact' else 'res_H8_T4.json');prior=json.loads(fn.read_text())['zoo1'];priorvalue=prior['acc' if mode=='exact' else 'est'];assert abs(1-old-priorvalue)<2e-15;assert geometry+oldops==prior['ops']
 def error(j):
  e=[r[j]-r[3] for r in rows];return dict(signed=math.fsum(e),absolute_sum=math.fsum(abs(x) for x in e),max_absolute=max(abs(x) for x in e),negative_count=sum(r[j]<0 for r in rows),negative_sum=math.fsum(r[j] for r in rows if r[j]<0))
 return dict(mode=mode,simplices=N,leaves=capture['counts']['leaves'],all_indices_once=True,accuracy_original=1-old,accuracy_doubled=1-new,accuracy_positive=1-positive,conditional_bound=bound,original_numerical=error(1),doubled_numerical=error(2),bound_misses=sum(r[4]>1e-9/N for r in rows),order_counts={str(n):sum(r[5]==n for r in rows) for n in (20,32,48)},geometry_named_ops=geometry,original_mass_named_ops=oldops,doubled_mass_named_ops=newops,original_full_named_ops=geometry+oldops,doubled_full_named_ops=geometry+newops,original_value_and_counter_reproduced=True,controls_named_ops=sum(d['control_named_ops'] for d in meta),capture_cpu=capture['cpu_seconds_including_imports'],replay_cpu=sum(d['cpu'] for d in meta),prior_source_sha256=hashlib.sha256(fn.read_bytes()).hexdigest())
e=collect('exact');r=collect('recipe108');mc=json.loads((PRIOR/'res_H8_T4.json').read_text())['zoo1']['truth'];se=math.sqrt(mc*(1-mc)/4000000)
out=dict(complete=True,checkpoint_seed=1,exact=e,recipe=r,recipe_approximation_accuracy_delta=r['accuracy_positive']-e['accuracy_positive'],conditional_recipe_minus_exact_bound=e['conditional_bound']+r['conditional_bound'],doubled_recipe_total_accuracy_delta=r['accuracy_doubled']-e['accuracy_positive'],original_recipe_total_accuracy_delta=r['accuracy_original']-e['accuracy_positive'],mc_diagnostic=dict(samples=4000000,observed=mc,standard_error=se,positive_exact_minus_mc=e['accuracy_positive']-mc,z=(e['accuracy_positive']-mc)/se),scope='Fixed held-out checkpoint; conditional reference bounds exclude floatinggeometry/roundoff. Inherited counters partial. No tuning, fullpanel/rung or samplingwin claim.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
