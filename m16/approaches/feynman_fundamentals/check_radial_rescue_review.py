"""Independent exact Gram/KKT/determinant and bound/merge audit of32 rescues."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
from collections import Counter
import json,time,math
import numpy as np
import mpmath as mp
start=time.process_time();here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h2_56/m16/approaches/radial_bound_rescue');base=Path('/workspace/work/recover-small_h2_34/m16/approaches');rays=np.load(base/'class_region_t4/quadratic_exact_mass_calls.npz')['rays'];prior=np.load(base/'analytic_inner_mass/aggregate_values.npz');combined=np.load(src/'combined_values.npz');reference=np.load(base/'radial_t4_reference/combined_values.npz');rows=sum([json.loads((src/f'chunk_{i}.json').read_text())['rows'] for i in range(4)],[]);old=json.loads((base/'analytic_inner_mass/aggregate.json').read_text());assert [r['index'] for r in rows]==old['unresolved_indices'];ps=list(permutations(range(4)));methods=Counter();ledger=Counter();maxfactor=0.;mp.mp.dps=100
m=lambda f:mp.mpf(f.numerator)/f.denominator
for r in rows:
 i=r['index'];R=[[F(float(v)) for v in row] for row in rays[i]];G=[[sum(a*b for a,b in zip(x,y)) for y in R] for x in R];det=sum(((-1)**sum(p[a]>p[b] for a in range(4) for b in range(a+1,4))*math.prod(R[a][p[a]] for a in range(4)) for p in ps),F(0));stage=r['stages'][-1];c=stage['certificate'];a,b=F(c['a']),F(c['b']);assert det==F(c['det']) and b==max(G[j][j] for j in range(4));methods[stage['method']]+=1
 if stage['method']=='acute':assert all(G[j][k]>=0 for j in range(4) for k in range(j+1,4)) and a==min(G[j][j] for j in range(4))/4
 else:
  y=list(map(F,c['y']));assert sum(y)==1 and min(y)>=0;gy=[sum(g*t for g,t in zip(line,y)) for line in G];assert sum(t*g for t,g in zip(y,gy))==a;slack=[g-a for g in gy];assert min(slack)>=0 and slack==list(map(F,c['slack'])) and all(not yj or not sj for yj,sj in zip(y,slack))
 assert a>0 and det
 for trial in stage['trials']:
  n=trial['order'];alpha=(m(a)+m(b))/2;delta=mp.sqrt(m(a)*m(b));rho=(mp.sqrt(m(b))-mp.sqrt(m(a)))/(mp.sqrt(m(b))+mp.sqrt(m(a)));N=n-2;tail=2/delta**3*((alpha+delta*(N+1))*rho**(N+1)/(1-rho)+delta*rho**(N+2)/(1-rho)**2);quadrature=abs(m(det))*tail/(6*mp.pi**2);factor=abs(m(F(abs(r['old_determinant'])))-abs(m(det)))/(12*mp.pi**2*m(a)**2);total=quadrature+factor
  assert math.isclose(float(total),trial['total_bound'],rel_tol=1e-14,abs_tol=1e-300);assert (total<=mp.mpf(1)/(10**7*6347))==trial['qualifies'];maxfactor=max(maxfactor,float(factor))
 assert stage['trials'][-1]['qualifies'] and r['order']==stage['trials'][-1]['order'];assert r['value']==combined['values'][i] and r['conditional_bound']==combined['conditional_bounds'][i];assert total+mp.mpf(r['series_error_bound'])<=mp.mpf(1)/(10**7*6347);ledger.update(r['kernel_counts'])
mask=np.isfinite(prior['values']);assert int(mask.sum())==6315 and np.array_equal(combined['values'][mask],prior['values'][mask]);assert np.array_equal(combined['conditional_bounds'][mask],(prior['bounds']+prior['series_bounds'])[mask]);mass=sum((F(float(v)) for v in combined['values']),F(0));refmass=sum((F(float(v)) for v in reference['values']),F(0));summary=json.loads((src/'summary.json').read_text());assert abs(float(1-mass)-summary['accuracy'])<1e-16;assert sum(ledger.values())==2038604
out=dict(rescued=32,methods=dict(methods),unchanged_prior=6315,accuracy=float(1-mass),reference_accuracy=float(1-refmass),accuracy_delta=float(refmass-mass),conditional_bound_sum=math.fsum(combined['conditional_bounds']),new_kernel_named_total=sum(ledger.values()),maximum_determinant_factor_bound=maxfactor,review_cpu=time.process_time()-start,scope='Exact stored-ray Gram/KKT/determinant checks,100digit bound recomputation and bitwise merge. Inherited6315 and general floating kernel/rule/geometry errors excluded; rational/MP cost separate. Initial mass/accuracy label inversion corrected without changing numerical archive.')
(here/'radial_rescue_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
