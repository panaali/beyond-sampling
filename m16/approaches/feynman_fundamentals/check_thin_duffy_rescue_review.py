import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,gzip,hashlib,time,math
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/thin_duffy_bound');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'rescue/results.json');proto=read(P/'rescue/protocol.json');old={r['index']:r for p in P.glob('shard_*.json.gz') for r in json.loads(gzip.decompress(p.read_bytes()))['rows']};rays=np.load(read(P/'protocol.json')['rays'])['rays'];assert d['complete'] and [r['index'] for r in d['rows']]==read(P/'summary.json')['loose_indices']==proto['indices'];tot={i:F(r['quadrature']['bound_decimal']) for i,r in old.items()};construct=Counter();validate=Counter();mp.mp.dps=100
for r in d['rows']:
 idx=r['index'];c=r['certificate'];R=[[F(float(x)) for x in row] for row in rays[idx]];G=[[sum(x*y for x,y in zip(a,b)) for b in R] for a in R];a,b=F(c['a']),F(c['b']);y=list(map(F,c['y']));slack=[sum(G[i][j]*y[j] for j in range(4))-a for i in range(4)]
 assert sum(y)==1 and min(y)>=0 and min(slack)>=0 and all(v*w==0 for v,w in zip(y,slack));assert list(map(F,c['slack']))==slack and b==max(G[i][i] for i in range(4));assert a>=F(old[idx]['a']) and c['det']==old[idx]['det'];assert r['recheck']['a']==str(a) and r['recheck']['b']==str(b)
 m=lambda q:mp.mpf(q.numerator)/q.denominator;af,bf=m(a),m(b);delta=mp.sqrt(af*bf);rho=(mp.sqrt(bf)-mp.sqrt(af))/(mp.sqrt(bf)+mp.sqrt(af));tail=2*rho**9/delta**3*((af+bf)/(2*(1-rho))+delta*(9/(1-rho)+rho/(1-rho)**2));bound=abs(m(F(c['det'])))*tail/(6*mp.pi**2)
 for key in ['quadrature','precision100']:
  q=r[key];assert q['qualifies'] and bound<mp.mpf(1)/350980000000 and abs(bound-mp.mpf(q['bound_decimal']))/bound<mp.mpf('1e-70')
 tot[idx]=F(r['quadrature']['bound_decimal']);construct.update(r['construction_inventory']['operations']);validate.update(r['recheck']['exact_inventory']['operations'])
assert sum(tot.values(),F(0))==F(d['aggregate_bound_exact_decimal_sum']);assert float(sum(tot.values(),F(0)))==d['aggregate_bound_sum'];assert construct==Counter(d['construction_inventory']) and validate==Counter(d['validation_inventory']);assert d['unchanged_cones']==17901 and d['remaining_loose_indices']==[] and d['mass_values_unchanged'] and d['no_mass_integrations'] and not d['full_floating_certificate']
out=dict(exact_KKT_checked=54,unchanged_prior_bounds=17901,merged_bound_sum=d['aggregate_bound_sum'],prior_bound_sum=read(P/'summary.json')['conditional_bound_sum'],review_cpu=time.process_time()-start,scope='Independent exact KKT inequalities plus100digit fixed-tail and exact decimal merge checks. Same GL10 and masses; no integrations. MP evaluation still not directed and source floating determinant/nodes/geometry/kernel excluded.')
Path(__file__).with_name('thin_duffy_rescue_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
