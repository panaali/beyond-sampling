"""Independent2x2 Schur conditional kernel at fixed original rule."""
from pathlib import Path
from fractions import Fraction as F
import json,time
import mpmath as m
here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h4_4/m16/approaches/plackett_precision_review');d=json.loads((src/'fixture.json').read_text());meta=d['rule_metadata'];assert meta['order']==12 and meta['power']==2 and meta['rmax']<.95 and meta['determinant']>1e-12
fixture=json.loads(Path('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review/negative_mass_fixture.json').read_text());start=time.process_time();m.mp.dps=100
def binary(x):q=F(float(x));return m.mpf(q.numerator)/q.denominator
R=m.matrix([[binary(x) for x in row] for row in fixture['rays']]);S=(R*R.T)**-1;C=m.matrix([[S[i,j]/m.sqrt(S[i,i]*S[j,j]) for j in range(4)] for i in range(4)])
rounded=m.matrix([[binary(x) for x in row] for row in d['rounded_C']]);rule=[tuple(binary(x) for x in row) for row in meta['rounded_rule']];x,w=m.gauss_quadrature(12,'legendre');ideal=[(1-(1-(x[k]+1)/2)**2,w[k]/2,2*(1-(x[k]+1)/2)) for k in range(12)]
def integrate(C,rule):
 pairs=[(i,j) for i in range(4) for j in range(i+1,4)];base=m.mpf(1)/16+m.fsum(m.asin(C[i,j]) for i,j in pairs)/(8*m.pi);terms=[]
 for t,wt,jac in rule:
  V=m.eye(4)+t*(C-m.eye(4))
  for i,j in pairs:
   k,l=[z for z in range(4) if z not in (i,j)];den=V[i,i]*V[j,j]-V[i,j]**2
   def conditional(a,b):return V[a,b]-(V[a,i]*V[b,i]*V[j,j]-(V[a,i]*V[b,j]+V[a,j]*V[b,i])*V[i,j]+V[a,j]*V[b,j]*V[i,i])/den
   rho=conditional(k,l)/m.sqrt(conditional(k,k)*conditional(l,l));assert abs(rho)<=1;terms.append(wt*jac*C[i,j]*m.asin(rho)/m.sqrt(1-(t*C[i,j])**2))
 return base+m.fsum(terms)/(4*m.pi**2)
rows=[]
for name,mat,nodes in [('rounded_C_rounded_rule',rounded,rule),('exact_ray_C_rounded_rule',C,rule),('exact_ray_C_ideal_rule',C,ideal)]:
 v=integrate(mat,nodes);claimed=m.mpf(d['rows'][1]['arms'][name]['mass']);delta=abs(v-claimed);assert delta<m.mpf('1e-80');assert v<0;rows.append(dict(arm=name,mass=m.nstr(v,30),independent_difference=m.nstr(delta,6),minus_positive=float(v-binary(d['positive_reference']))))
out=dict(arms=rows,rule_order=12,endpoint_power=2,review_cpu=time.process_time()-start,scope='Independent Schur formulation and Gram-inverse correlation reconstruction at100digits, same fixed rule. Numerical stability evidence, no MP interval or full-source attribution beyond fixture672.')
(here/'plackett_precision_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
