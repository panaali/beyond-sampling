"""Saved child wedge references, directed tails and fixed-order aggregation."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,itertools,time
import mpmath as mp
mp.mp.dps=180;start=time.process_time();P=Path('/workspace/work/recover-small_h2_910/review/edge_split5')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def m(x):
 x=F(x);return mp.mpf(x.numerator)/x.denominator
d=read(P/'numerical.json');s=read(P/'summary.json');g=read(P/'geometry.json');protocol=read(P/'protocol.json')
assert d['complete'] and len(d['children'])==22 and len(d['rows'])==66 and len(d['groups'])==33
assert s['numerical_sha256']==sha(P/'numerical.json') and d['geometry_sha256']==sha(P/'geometry.json') and d['source_sha256']==sha(P/'numerical.py') and d['protocol_sha256']==sha(P/'protocol.json')
prior=read(Path(protocol['prior'])/'results.json');parents={r['name']:r for r in prior['fixtures']};geos={r['name']:r for r in g['rows']};children={};refs={};counts=Counter();dc=Counter();cc=Counter()
for r in d['children']:
 key=(r['fixture'],r['side']);assert key not in children;children[key]=r
 assert r['rays_hex']==geos[key[0]]['children_hex'][key[1]]
 A=[[F(float.fromhex(x)) for x in row] for row in r['rays_hex']]
 assert all(A[i][j]==F(i==j) for i in range(2,5) for j in range(5)) and all(A[i][j]==0 for i in range(2) for j in range(2,5))
 det=A[0][0]*A[1][1]-A[0][1]*A[1][0];dot=A[0][0]*A[1][0]+A[0][1]*A[1][1];assert det>0
 ref=mp.atan2(m(det),m(dot))/(16*mp.pi);refs[key]=ref
 assert abs(ref-mp.mpf(r['reference120']))<mp.mpf('1e-120') and abs(ref-mp.mpf(r['reference80']))<mp.mpf('1e-80')
 c=r['certificate'];G=[[sum((x*y for x,y in zip(u,v)),F()) for v in A] for u in A]
 assert G==[list(map(F,row)) for row in c['gram']] and det==F(c['det'])
 assert c['mode']=='acute' and all(x>=0 for row in G for x in row)
 assert F(c['a'])==min(G[i][i] for i in range(5))/5 and F(c['b'])==max(G[i][i] for i in range(5));counts.update(c['counts'])
for name in parents:assert abs(refs[name,0]+refs[name,1]-mp.mpf(parents[name]['reference120']))<mp.mpf('1e-120')
rows={}
for r in d['rows']:
 key=(r['fixture'],r['side'],r['order']);assert key not in rows;rows[key]=r
 assert 'exception' not in r and r['finite'] and r['probability_range'] and float.fromhex(r['returned_hex'])==r['value']
 ref=refs[key[:2]];err=m(r['value'])-ref
 assert abs(err-mp.mpf(r['error']))<mp.mpf('1e-70') and abs(abs(err)-mp.mpf(r['absolute_error']))<mp.mpf('1e-70')
 z=r['ideal_rule_bound'];c=children[key[:2]]['certificate'];a,b=F(z['a_expanded']),F(z['b_expanded']);assert 0<a<=F(c['a'])<=F(c['b'])<=b
 n=key[2];N=n-2;assert z['degree']==N
 mid=(m(a)+m(b))/2;t=(m(b)-m(a))/(m(b)+m(a));tail=m(a)**(-mp.mpf('2.5'))-mid**(-mp.mpf('2.5'))*sum(mp.rf(mp.mpf('2.5'),k)*t**k/mp.factorial(k) for k in range(N+1))
 assert m(z['raw_tail_interval'][0])<=tail<=m(z['raw_tail_interval'][1])
 ideal=abs(m(c['det']))*tail/(32*mp.pi**2);assert m(z['lower_formula_endpoint'])<=ideal<=m(z['upper_bound'])
 expected=dict(directed_divide=16+2*N,directed_multiply=16+4*N,directed_add=4+2*N,directed_subtract=4,correctly_rounded_sqrt=4,decimal_adjacent=4)
 assert z['counts']==expected and z['coefficient_counts']=={'rational_multiply':N};dc.update(z['counts']);cc.update(z['coefficient_counts'])
assert len(rows)==66
summary=[]
for n in (6,10,16):
 gs=[r for r in d['groups'] if r['order']==n];assert len(gs)==11
 for z in gs:
  name=z['fixture'];child=[rows[name,k,n] for k in (0,1)];total=sum((F(r['value']) for r in child),F());ref=refs[name,0]+refs[name,1];err=m(total)-ref;ub=sum((F(r['ideal_rule_bound']['upper_bound']) for r in child),F())
  assert abs(m(total)-mp.mpf(z['total_exact_float_sum']))<mp.mpf('1e-100') and abs(err-mp.mpf(z['error']))<mp.mpf('1e-70')
  assert F(z['bound_sum_fraction'])==ub and z['bound_qualifies']==(ub<=F(1,10**10))
  assert z['absolute_pass']==(abs(err)<=mp.mpf('1e-10')) and z['relative_pass']==(abs(err/ref)<=mp.mpf('1e-6'))
  assert z['source_proxy_ops']==sum(r['source_proxy_ops'] for r in child) and z['nodes']==2*n**4 and z['assembly_additions']==1
  old=next(r for r in prior['rows'] if r['fixture']==name and r['arm']==f'duffy{n}');assert z['original_unsplit_error']==old['error']
 reported=next(r for r in s['groups'] if r['order']==n)
 for field in ('absolute_pass','relative_pass'):assert reported[field]==sum(z[field] for z in gs)
 assert reported['bound_pass']==sum(z['bound_qualifies'] for z in gs) and reported['source_proxy_ops']==sum(z['source_proxy_ops'] for z in gs)
 summary.append(reported)
assert dict(counts)==s['radial_exact_counts'] and dict(dc)==s['directed_counts'] and dict(cc)==s['bound_coefficient_counts']
out=dict(complete=True,children=22,evaluations=66,groups=33,order_summaries=summary,numerical_sha256=sha(P/'numerical.json'),review_cpu=time.process_time()-start,new_mass_calls=0,scope='Exact binary child geometry inherited from separate independent review; actual wedge references, acute ranges, ideal directed tails and fixed-order raw aggregation independently checked. Order16 passes this finite11-fixture diagnostic, while only6/33 ideal bounds qualify. No trained/global/log/floating certificate or full native cost claim.')
Path(__file__).with_name('edge_split5_numerical_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
