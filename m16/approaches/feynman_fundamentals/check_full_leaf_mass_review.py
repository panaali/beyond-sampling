"""Independent disjoint full replay aggregation and analytic pair totals."""
import json,math,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review');archive=np.load(src/'captured.npz');parents=archive['parent_ids'];g=json.loads((src/'capture.json').read_text());seen=set();rows=[];allsource=[];allpositive=[];allbounds=[];total_cpu=0.;reused=0
for batch in range(8):
 d=json.loads((src/f'full_batch_{batch}.json').read_text());v=np.load(src/f'full_batch_{batch}.npz');assert d['complete'];ids=v['indices'].tolist();expected=np.flatnonzero((parents>=12*batch)&(parents<12*(batch+1))).tolist();assert ids==expected and not seen.intersection(ids);seen.update(ids)
 for key,field in [('source','source_sum'),('positive','positive_sum'),('bounds','conditional_integration_bound')]:assert math.fsum(v[key])==d[field]
 assert min(v['positive'])>=0 and min(v['bounds'])>=0;assert sum(v['source']<0)==d['negative_source_count'];assert sum(v['bounds']>d['per_simplex_bound_allowance'])==d['bound_target_misses'];assert sum(v['reused'])==d['reused_selected_controls'];reused+=int(sum(v['reused']));total_cpu+=d['cumulative_cpu']
 truth=F(0);pair_records=[]
 for a in range(12*batch,12*(batch+1),2):
  p,q=g['parents'][a:a+2];assert p['perm']==q['perm'] and p['sigma']==q['sigma'];r=p['perm'].index(0)+1;positive=F(sum(math.comb(4,k) for k in range(r,5)),16);target=(positive if p['sigma']>0 else 1-positive)/24;truth+=target
  use=np.isin(parents[v['indices']],(a,a+1));value=math.fsum(v['positive'][use]);bound=math.fsum(v['bounds'][use]);assert abs(value-float(target))<=bound+2e-15
  pair_records.append(dict(parents=[a,a+1],truth=str(target),positive_error=value-float(target),bound=bound))
 assert abs(d['positive_sum']-float(truth))<=d['conditional_integration_bound']+2e-15
 rows.append(dict(batch=batch,simplices=len(ids),truth=str(truth),source_error=d['source_sum']-float(truth),positive_error=d['positive_sum']-float(truth),integration_bound=d['conditional_integration_bound'],negative_source_count=d['negative_source_count'],misses=d['bound_target_misses'],pairs=pair_records));allsource.extend(v['source']);allpositive.extend(v['positive']);allbounds.extend(v['bounds'])
assert seen==set(range(len(parents))) and len(seen)==43471
exact_source=sum(map(F,map(float,allsource)),F(0));exact_positive=sum(map(F,map(float,allpositive)),F(0));exact_bounds=sum(map(F,map(float,allbounds)),F(0))
assert float(exact_source)==math.fsum(allsource) and float(exact_positive)==math.fsum(allpositive) and float(exact_bounds)==math.fsum(allbounds)
out=dict(exact_binary_source_sum=str(exact_source),exact_binary_positive_sum=str(exact_positive),exact_binary_bound_sum=str(exact_bounds),rows=rows,covered_simplices=len(seen),source_mass=math.fsum(allsource),positive_mass=math.fsum(allpositive),integration_only_bound=math.fsum(allbounds),negative_source_count=int(sum(x<0 for x in allsource)),reused_selected_controls=reused,recorded_replay_cpu=total_cpu,review_cpu=time.process_time()-start,scope='Independent all-array aggregation and48 analytic chamber-union checks. Selected1762 certificates independently exact-audited separately; no all43471 exact certificate repetition, floating geometry certificate or error-region accuracy inference.')
(here/'full_leaf_mass_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
