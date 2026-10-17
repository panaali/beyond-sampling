"""Independent all-array held-out exact/recipe attribution; no integration."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math
import numpy as np
start=time.process_time();here=Path(__file__).parent;root=Path('/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout');prior=Path('/workspace/work/admin-t121-rank-mass/m16/approaches/final3');out={};fractions={}
for mode in ('exact','recipe108'):
 src=root/mode;cap=json.loads((src/'error_capture.json').read_text());parents=np.load(src/'error_captured.npz')['parent_ids'];N=len(parents);seen=set();sums=[F(0)]*4;oldops=newops=0;cpu=0.;oldnegative=newnegative=0;misses=0;newl1=F(0)
 for b in range(8):
  meta=json.loads((src/f'batch_{b}.json').read_text());v=np.load(src/f'batch_{b}.npz');ids=v['indices'];assert meta['complete'];assert not seen.intersection(ids.tolist());seen.update(ids.tolist());assert np.all((parents[ids]>=12*b)&(parents[ids]<12*b+12))
  for path,sha in meta['source_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
  fields=[v[k] for k in ('source','doubled','positive','bounds')]
  for j,x in enumerate(fields):sums[j]+=sum((F(float(a)) for a in x),F(0))
  oldnegative+=int(np.sum(fields[0]<0));newnegative+=int(np.sum(fields[1]<0));newl1+=sum((abs(F(float(a))-F(float(b))) for a,b in zip(fields[1],fields[2])),F(0));misses+=int(np.sum(fields[3]>1e-9/N));oldops+=meta['source_mass_named_ops'];newops+=meta['doubled_mass_named_ops'];cpu+=meta['cpu']
 assert seen==set(range(N));assert newops==2*oldops-156*N;geometry=cap['named_geometry_ops']-16*(N+1)+8*96;original,doubled,pos,bound=sums;fractions[mode]=sums;published=json.loads((prior/('exact_H8_T4.json' if mode=='exact' else 'res_H8_T4.json')).read_text())['zoo1'];assert geometry+oldops==published['ops'];assert abs(float(1-original)-published['acc' if mode=='exact' else 'est'])<2e-15
 out[mode]=dict(simplices=N,leaves=cap['counts']['leaves'],accuracy_original=float(1-original),accuracy_doubled=float(1-doubled),accuracy_positive=float(1-pos),bound=float(bound),original_negative_count=oldnegative,doubled_negative_count=newnegative,doubled_numerical_accuracy_delta=float(pos-doubled),doubled_percone_L1=float(newl1),bound_misses=misses,old_full_partial_count=geometry+oldops,doubled_full_partial_count=geometry+newops,recorded_replay_cpu=cpu)
e=fractions['exact'];r=fractions['recipe108'];approx=e[2]-r[2];total=e[2]-r[1];assert approx+r[2]-r[1]==total;truth=json.loads((prior/'res_H8_T4.json').read_text())['zoo1']['truth'];se=math.sqrt(truth*(1-truth)/4000000)
result=dict(modes=out,recipe_approximation_accuracy_delta=float(approx),combined_conditional_bound=float(e[3]+r[3]),doubled_recipe_total_accuracy_delta=float(total),mc_observed=truth,mc_standard_error=se,positive_exact_minus_mc_SE=(float(1-e[2])-truth)/se,review_cpu=time.process_time()-start,scope='Frozen held-out checkpoint, exhaustive saved-array sums and source/counter parity. Bounds exclude floating geometry/roundoff; counters partial. No tuning, full nine-panel/rung or complete-cost sampling-win claim.')
if (root/'summary.json').exists():
 s=json.loads((root/'summary.json').read_text());assert abs(s['recipe_approximation_accuracy_delta']-float(approx))<2e-16;assert abs(s['doubled_recipe_total_accuracy_delta']-float(total))<2e-16
(here/'heldout_mass_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
