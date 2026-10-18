"""Independent exact KKT certificate, permutation determinant and series-tail audit."""
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import json,time
import numpy as np
import mpmath as mp
start=time.process_time();p=Path('/workspace/work/recover-small_h2_34/m16/approaches');d=json.loads((p/'radial_t4_reference/results.json').read_text());R=np.load(p/'class_region_t4/quadratic_exact_mass_calls.npz')['rays'];mp.mp.dps=80
m=lambda q:mp.mpf(q.numerator)/q.denominator
maxrel=mp.mpf(0)
for row in d['rows']:
 A=[[F(float(v)) for v in ray] for ray in R[row['index']]];G=[[sum(a*b for a,b in zip(r,s)) for s in A] for r in A];c=row['certificate'];a,b=F(c['a']),F(c['b']);y=list(map(F,c['y']));slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)]
 assert min(y)>=0 and sum(y)==1 and slack==list(map(F,c['slack'])) and min(slack)>=0 and all(y[i]*slack[i]==0 for i in range(4))
 assert a==sum(y[i]*G[i][j]*y[j] for i in range(4) for j in range(4)) and b==max(G[i][i] for i in range(4)) and 0<a<=b
 det=sum((-1)**sum(t[i]>t[j] for i in range(4) for j in range(i+1,4))*__import__('math').prod(A[i][t[i]] for i in range(4)) for t in permutations(range(4)))
 assert det==F(c['det']) and det!=0
 alpha=(m(a)+m(b))/2;delta=mp.sqrt(m(a)*m(b));rho=(mp.sqrt(m(b))-mp.sqrt(m(a)))/(mp.sqrt(m(b))+mp.sqrt(m(a)));assert rho<mp.mpf('.5')
 for trial in row['trials']:
  N=trial['order']-2
  # Independent explicit summation of Chebyshev coefficient magnitudes.
  tail=mp.fsum(2*(alpha+k*delta)*rho**k/delta**3 for k in range(N+1,401));bound=abs(m(det))*tail/(6*mp.pi**2);rel=abs(bound-mp.mpf(trial['decimal']))/bound;maxrel=max(maxrel,rel);assert rel<mp.mpf('1e-47')
sets=[np.load(p/f'{name}/{file}') for name,file in [('positive_t4_corpus','positive_values.npz'),('acute_t4_reference','acute_values.npz'),('radial_t4_reference','radial_values.npz')]];masks=[np.isfinite(s['values']) for s in sets];assert np.all(sum(masks)==1)
sumf=lambda a:sum((F(float(x)) for x in a),F(0));mass=sum(sumf(s['values'][ok]) for s,ok in zip(sets,masks));bound=sum(sumf(s['bounds'][ok]) for s,ok in zip(sets,masks));combined=json.loads((p/'radial_t4_reference/combined.json').read_text());assert abs(float(1-mass)-combined['estimate'])<1e-16
out={'exact_KKT_certificates':len(d['rows']),'exact_permutation_determinants':len(d['rows']),'max_relative_explicit_series_tail_difference':str(maxrel),'disjoint_coverage':list(map(lambda a:int(sum(a)),masks)),'positive_accuracy':float(1-mass),'summed_integration_only_bound':float(bound),'cpu_seconds':time.process_time()-start,'scope':'all stored6347 cones covered; no floating geometry/quadrature-roundoff or complete cost certificate'}
Path(__file__).with_name('radial_certificate_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
