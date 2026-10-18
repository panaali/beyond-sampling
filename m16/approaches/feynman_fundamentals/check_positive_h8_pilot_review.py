"""Independent fixed64 selection, exact rescue certificate and ledger audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import permutations
import json,hashlib,time,math,numpy as np,mpmath as mp
start=time.process_time();H=Path(__file__).parent;P=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_h8_cost');D=json.loads((P/'pilot.json').read_text());S=json.loads((P/'summary.json').read_text());proto=json.loads((P/'protocol.json').read_text());rays=np.load('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review/error_captured.npz')['rays'];tol=1e-7/35098;mp.mp.dps=80;ledger=Counter();methods=Counter();errors=[];bounds=[];m=lambda f:mp.mpf(f.numerator)/f.denominator
for p,sha in proto['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha
assert D['complete'] and D['indices']==proto['indices'] and len(D['rows'])==64 and len(set(D['indices']))==64
for r in D['rows']:
 assert r['mode']=='returned' and r['bound']<=tol;methods[r['method']]+=1;ledger.update(r['center']['setup_selection_counts']);ledger.update(r.get('repeated_determinant_counts',{}));ledger.update(r.get('kernel_counts',{}));errors.append(F(r['value'])-F(r['reference']));bounds.append(F(r['bound']));assert abs(float(errors[-1]))<=r['bound']+r['reference_bound']+5e-14
 c=r['center']
 if r['method']=='center_interval':assert c['bound']<=tol and r['value']==c['value'] and r['bound']==c['bound']
 elif r['method']=='center_quadrature':
  assert c['trials'][-1]['order']==c['order'] and c['trials'][-1]['bound']<=tol and all(t['bound']>tol for t in c['trials'][:-1]);assert r['bound']==c['bound']+r['series_error_bound']
 else:
  assert c['mode']=='unresolved' and all(t['bound']>tol for t in c['trials']);z=r['rescue'];cert=z['stages'][-1]['certificate'];R=[[F(float(v)) for v in row] for row in rays[r['index']]];G=[[sum(a*b for a,b in zip(x,y)) for y in R] for x in R];a,b=F(cert['a']),F(cert['b']);y=list(map(F,cert['y']));assert min(y)>=0 and sum(y)==1;gy=[sum(g*t for g,t in zip(line,y)) for line in G];assert a==sum(t*g for t,g in zip(y,gy));slack=[v-a for v in gy];assert slack==list(map(F,cert['slack'])) and min(slack)>=0 and all(v*w==0 for v,w in zip(y,slack));assert b==max(G[i][i] for i in range(4))
  det=sum(((-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4))*math.prod(R[i][p[i]] for i in range(4)) for p in permutations(range(4))),F(0));assert det==F(cert['det']) and a>0
  for t in z['stages'][-1]['trials']:
   n=t['order'];alpha=(m(a)+m(b))/2;delta=mp.sqrt(m(a)*m(b));rho=(mp.sqrt(m(b))-mp.sqrt(m(a)))/(mp.sqrt(m(b))+mp.sqrt(m(a)));tail=2*rho**(n-1)/delta**3*(alpha/(1-rho)+delta*((n-1)/(1-rho)+rho/(1-rho)**2));bd=abs(m(det))*tail/(6*mp.pi**2)+abs(m(F(abs(z['old_determinant'])))-abs(m(det)))/(12*mp.pi**2*m(a)**2);assert math.isclose(float(bd),t['total_bound'],rel_tol=1e-14);assert bool(bd<=mp.mpf(1)/(10**7*35098))==t['qualifies']
  assert z['order']==12 and z['stages'][-1]['trials'][-1]['qualifies'] and r['bound']==z['conditional_bound']
ledger.update(D['cold_rule_counts']);ledger['universal_constant_arithmetic']+=6;assert dict(ledger)==D['known_float_named_counts'] and sum(ledger.values())==147062;assert dict(methods)==S['modes'];assert float(sum(map(abs,errors),F(0)))==S['absolute_error_sum'];assert float(sum(bounds,F(0)))==S['bound_sum']
out=dict(fixed_cases=64,methods=dict(methods),exact_KKT_rescue_index=31197,rescue_order=12,absolute_error_sum=float(sum(map(abs,errors),F(0))),max_absolute_error=float(max(map(abs,errors))),conditional_bound_sum=float(sum(bounds,F(0))),known_float_named_total=sum(ledger.values()),review_cpu=time.process_time()-start,scope='Fixed64 only. Existing center setup/inner formulas retained; exact one-rescue Gram/KKT/determinant independently verified. General floating setup/kernel errors and rational/MP/internal costs remain excluded; no full-corpus or cost extrapolation.')
(H/'positive_h8_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
