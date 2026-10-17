"""Separate numerical error on recipe cells from captured recipe approximation."""
import hashlib,json,math
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'leaf_mass_review';meta=[];rows=[]
capture=json.loads((HERE/'error_capture.json').read_text());assert capture['complete'];parents=np.load(HERE/'error_captured.npz')['parent_ids'];N=len(parents)
for b in range(8):
 d=json.loads((HERE/f'batch_{b}.json').read_text());assert d['complete'];meta.append(d);a=np.load(HERE/f'batch_{b}.npz');idx=a['indices'];assert np.all((parents[idx]>=b*12)&(parents[idx]<12*(b+1)))
 for p,h in d['source_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 rows.extend(zip(*(a[k].tolist() for k in ('indices','source','doubled','positive','bounds','orders','acute'))))
rows.sort();assert [r[0] for r in rows]==list(range(N));sums=[math.fsum(r[j] for r in rows) for j in (1,2,3,4)];old,new,positive,bound=sums;exact=json.loads((SOURCE/'error_summary.json').read_text());reference=exact['positive_accuracy'];prior=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/final3/res_H8_T4.json');published=json.loads(prior.read_text())['zoo0'];assert abs(1-old-published['est'])<2e-15
oldops=sum(d['source_mass_named_ops'] for d in meta);newops=sum(d['doubled_mass_named_ops'] for d in meta);assert newops==2*oldops-156*N
geometry=capture['named_geometry_ops']-16*N-16+8*96;assert geometry+oldops==published['ops']
def numerical(j):
 errors=[r[j]-r[3] for r in rows];return dict(signed_error=math.fsum(errors),absolute_error_sum=math.fsum(abs(x) for x in errors),max_absolute_error=max(abs(x) for x in errors),negative_count=sum(r[j]<0 for r in rows),negative_sum=math.fsum(r[j] for r in rows if r[j]<0))
accuracies={'original':1-old,'doubled':1-new,'positive_recipe':1-positive,'positive_exact':reference};approx=1-positive-reference
out=dict(complete=True,simplices=N,recipe_leaves=capture['counts']['leaves'],all_indices_once=True,accuracies=accuracies,recipe_approximation_accuracy_delta=approx,original_numerical_accuracy_delta=positive-old,doubled_numerical_accuracy_delta=positive-new,doubled_total_accuracy_delta=1-new-reference,conditional_recipe_bound=bound,conditional_recipe_minus_exact_bound=bound+exact['conditional_integration_bound'],original_numerical_errors=numerical(1),doubled_numerical_errors=numerical(2),bound_misses=sum(r[4]>1e-9/N for r in rows),order_counts={str(n):sum(r[5]==n for r in rows) for n in (20,32,48)},source_reproduces_published=True,geometry_named_ops=geometry,original_mass_named_ops=oldops,doubled_mass_named_ops=newops,original_full_named_ops=geometry+oldops,doubled_full_named_ops=geometry+newops,control_named_ops=sum(d['control_named_ops'] for d in meta),capture_cpu=capture['cpu_seconds_including_imports'],replay_cpu=sum(d['cpu'] for d in meta),published_source_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),scope='One frozen recipe/checkpoint; both bounds exclude floating geometry/roundoff. Captured approximate cells need not match true-network errors pointwise. Inherited costs partial, no sampling-win/rung claim.')
(HERE/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
