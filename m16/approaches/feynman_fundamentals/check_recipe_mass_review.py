"""Independent exact sums of stored floats: recipe bias versus mass error."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,time,hashlib,math
import numpy as np
start=time.process_time(); here=Path(__file__).parent; root=Path('/workspace/work/recover-small_h2_910/m16/approaches'); src=root/'recipe_mass_separation'; ref=root/'leaf_mass_review'; summary=json.loads((src/'summary.json').read_text()); cap=json.loads((src/'error_capture.json').read_text()); arr=np.load(src/'error_captured.npz'); parents=arr['parent_ids']; N=len(parents); seen=set(); sums=[F(0)]*4; oldops=newops=0; rows=[]; cpu=0.
for b in range(8):
 d=json.loads((src/f'batch_{b}.json').read_text()); v=np.load(src/f'batch_{b}.npz'); ids=v['indices']; assert d['complete'] and not seen.intersection(ids.tolist());seen.update(ids.tolist()); assert np.all((parents[ids]>=12*b)&(parents[ids]<12*b+12))
 for path,sha in d['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
 fields=[v[k] for k in ('source','doubled','positive','bounds')]
 for j,field in enumerate(fields):sums[j]+=sum(map(lambda x:F(float(x)),field),F(0))
 rows.extend(zip(*fields)); oldops+=d['source_mass_named_ops'];newops+=d['doubled_mass_named_ops'];cpu+=d['cpu']
assert seen==set(range(N)) and N==28000; assert sum(p['leaves'] for p in cap['parents'])==6182
exact_mass=F(0); exact_bound=F(0)
for b in range(8):
 v=np.load(ref/f'error_batch_{b}.npz'); exact_mass+=sum((F(float(x)) for x in v['positive']),F(0));exact_bound+=sum((F(float(x)) for x in v['bounds']),F(0))
old,new,pos,bound=sums; approximation=exact_mass-pos; old_num=pos-old;new_num=pos-new
assert approximation+new_num==exact_mass-new
geometry=cap['named_geometry_ops']-16*(N+1)+8*96
assert newops==2*oldops-156*N and geometry+oldops==201163614 and geometry+newops==361560702
published=json.loads(Path('/workspace/work/admin-t121-rank-mass/m16/approaches/final3/res_H8_T4.json').read_text())['zoo0'];assert abs(float(1-old)-published['est'])<2e-15
for name,value in [('recipe_approximation_accuracy_delta',approximation),('original_numerical_accuracy_delta',old_num),('doubled_numerical_accuracy_delta',new_num),('conditional_recipe_minus_exact_bound',bound+exact_bound)]:assert abs(float(value)-summary[name])<2e-16
l1=sum((abs(F(float(r[1]))-F(float(r[2]))) for r in rows),F(0));neg=sum(bool(r[1]<0) for r in rows);assert neg==503 and abs(float(l1)-summary['doubled_numerical_errors']['absolute_error_sum'])<1e-18
out=dict(cones=N,leaves=6182,accuracies=dict(original=float(1-old),doubled=float(1-new),positive_recipe=float(1-pos),positive_exact=float(1-exact_mass)),recipe_approximation_accuracy_delta=float(approximation),original_numerical_accuracy_delta=float(old_num),doubled_numerical_accuracy_delta=float(new_num),doubled_total_accuracy_delta=float(exact_mass-new),combined_conditional_bound=float(bound+exact_bound),doubled_L1=float(l1),doubled_negative_terms=neg,source_mass_named_ops=oldops,doubled_mass_named_ops=newops,inherited_geometry_named_ops=geometry,recorded_replay_cpu=cpu,review_cpu=time.process_time()-start,scope='Exact stored-float sums, source provenance and partial-counter identity; no new integration, true-network membership of approximate cells, full floating certificate or noisy-model/cost pass.')
(here/'recipe_mass_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
