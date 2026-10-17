import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from decimal import Decimal as D,localcontext
import json,gzip,hashlib,math,time
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/thin_duffy_bound');start=time.process_time()
def read(p):return json.loads(p.read_text())
s=read(P/'summary.json');proto=read(P/'protocol.json');rays=np.load(proto['rays'])['rays'];prior=read(P/'prior_certificates.json');rows=[]
for e in s['shards']:
 p=P/e['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'];d=json.loads(gzip.decompress(p.read_bytes()));assert d['complete'];rows+=d['rows']
latest=Path('/workspace/work/recover-small_h2_56/m16/approaches/log_cofactor_review/latest');actual={r['index']:r for j in range(8) for r in read(latest/f'full_{j}.json')['rows'] if r['diagnostics']['branch']=='duffy10'};assert [r['index'] for r in rows]==sorted(actual)==proto['indices'];methods=Counter();exact=Counter();loose=[];mp.mp.dps=80
for r in rows:
 idx=r['index'];a0=actual[idx];assert r['mode']=='bounded' and r['source_value']==a0['value'] and r['reference']==a0['reference'] and r['reference_bound']==a0['reference_bound'];R=[[F(float(x)) for x in row] for row in rays[idx]];G=[[sum(x*y for x,y in zip(a,b)) for b in R] for a in R];b=max(G[i][i] for i in range(4));rs=list(map(sum,G));total=sum(rs);m=min(rs);candidates={}
 # Independent six-minor determinant identity.
 pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];u=[R[0][i]*R[1][j]-R[0][j]*R[1][i] for i,j in pairs];v=[R[2][i]*R[3][j]-R[2][j]*R[3][i] for i,j in pairs];det=u[0]*v[5]-u[1]*v[4]+u[2]*v[3]+u[3]*v[2]-u[4]*v[1]+u[5]*v[0]
 assert det==F(r['det']) and b==F(r['b']) and det
 if m>0 and total>0:candidates['center']=m*m/total
 acute=all(G[i][j]>=0 for i in range(4) for j in range(i+1,4));assert acute==r['acute']
 if acute:candidates['acute']=min(G[i][i] for i in range(4))/4
 if str(idx) in prior:
  c=prior[str(idx)]['certificate'];assert F(c['det'])==det and F(c['b'])==b
  if prior[str(idx)]['method']=='exact_kkt':
   a=F(c['a']);y=list(map(F,c['y']));slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)];assert sum(y)==1 and min(y)>=0 and min(slack)>=0 and all(x*t==0 for x,t in zip(y,slack));candidates['reused_kkt']=a
 assert {k:F(v) for k,v in r['lower_candidates'].items()}==candidates;a=max(candidates.values());assert F(r['a'])==a==candidates[r['lower_method']];methods[r['lower_method']]+=1;exact.update(r['exact_inventory']['operations'])
 af=mp.mpf(a.numerator)/a.denominator;bf=mp.mpf(b.numerator)/b.denominator;df=mp.mpf(abs(det.numerator))/det.denominator;alpha=(af+bf)/2;delta=mp.sqrt(af*bf);rho=(mp.sqrt(bf)-mp.sqrt(af))/(mp.sqrt(bf)+mp.sqrt(af));tail=2*rho**9/delta**3*(alpha/(1-rho)+delta*(9/(1-rho)+rho/(1-rho)**2));bound=df*tail/(6*mp.pi**2);q=r['quadrature'];assert q['order']==10 and q['polynomial_degree']==8 and math.isclose(float(bound),q['bound'],rel_tol=2e-13,abs_tol=1e-300);assert bool(bound<=mp.mpf(1)/350980000000)==q['qualifies']
 if not q['qualifies']:loose.append(idx)
assert loose==s['loose_indices'] and methods==Counter(s['lower_method_counts']) and exact==Counter(s['exact_operation_inventory']);assert math.fsum(r['quadrature']['bound'] for r in rows)==s['conditional_bound_sum'];assert math.fsum(abs(r['observed_difference']) for r in rows)==s['observed_thin_L1_error'];assert len(loose)==54 and s['aggregate_comparison_pass'] is False
out=dict(cases=len(rows),all_exact_ranges_and_determinants_checked=True,all_mp_tails_recomputed=True,methods=dict(methods),loose_indices=loose,conditional_bound_sum=s['conditional_bound_sum'],observed_L1=s['observed_thin_L1_error'],review_cpu=time.process_time()-start,scope='All stored dyadic Gram/range/determinant certificates independently reconstructed and fixed GL10 degree8 tails recomputed. Loose bounds are not measured errors. MP tail evaluation is not directed; source floating determinant/nodes/kernel/geometry and log branch remain outside.')
Path(__file__).with_name('thin_duffy_bound_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='loose_indices'}))
