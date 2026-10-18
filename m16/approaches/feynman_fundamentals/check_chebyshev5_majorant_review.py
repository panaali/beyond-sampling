"""Independent nu3 geometric-moment formula and fixed99 bound contraction."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from math import isqrt
import json,hashlib,time
import mpmath as mp
mp.mp.dps=180;start=time.process_time();P=Path('/workspace/work/recover-small_h2_910/review/chebyshev5_majorant')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def m(x):
 x=F(x);return mp.mpf(x.numerator)/x.denominator
def tail(ra,rb,N):
 if ra==rb:return ra*0
 ss=ra*rb;B=(ra*ra+rb*rb)/(2*ss);rho=(rb-ra)/(rb+ra);k=N+1
 T0=rho**k/(1-rho);T1=k*T0+rho**(k+1)/(1-rho)**2;T2=k*k*T0+2*k*rho**(k+1)/(1-rho)**2+rho**(k+1)*(1+rho)/(1-rho)**3
 return (T2+3*B*T1+(3*B*B-1)*T0)/ss**3
p=read(P/'protocol.json');c=read(P/'controls.json');d=read(P/'results.json')
for path,h in p['pins'].items():assert sha(path)==h
assert c['complete'] and d['complete'] and len(c['rows'])==9 and len(c['invalid'])==4 and len(d['rows'])==99 and len(d['groups'])==66
assert d['controls_sha256']==sha(P/'controls.json') and d['source_sha256']==sha(P/'run.py') and d['bound_sha256']==sha(P/'bound.py') and d['protocol_sha256']==sha(P/'protocol.json')
for r in c['rows']:
 a,b=F(r['a']),F(r['b']);ra=F(isqrt(a.numerator),isqrt(a.denominator));rb=F(isqrt(b.numerator),isqrt(b.denominator));assert ra*ra==a and rb*rb==b
 truth=tail(ra,rb,r['degree']);assert truth==F(r['exact']) and F(r['interval'][0])<=truth<=F(r['interval'][1])
assert all(z['status']=='rejected' for z in c['invalid']) and c['nu3_polynomial_equal']
orig=read(p['original']);child=read(p['children']);certs={('original',r['fixture'],None):r for r in orig['certificates']};certs.update({('child',r['fixture'],r['side']):r['certificate'] for r in child['children']})
seen=set();counts=Counter();qualified=Counter();tighter=Counter();rows={}
for r in d['rows']:
 key=(r['panel'],r['fixture'],r['side'],r['order']);assert key not in seen;seen.add(key);rows[key]=r;assert r['status']=='completed'
 cert=certs[key[:3]];assert r['certificate']=={k:cert[k] for k in ('a','b','det')}
 a,b=F(r['a_expanded']),F(r['b_expanded']);assert 0<a<=F(cert['a'])<=F(cert['b'])<=b
 N=r['order']-2;assert r['degree']==N and r['order'] in (6,10,16)
 ra,rb=mp.sqrt(m(a)),mp.sqrt(m(b));scale=(ra+rb)/2;t3=tail(ra,rb,N);t5=scale*t3;value=abs(m(cert['det']))*t5/(32*mp.pi**2)
 for field,val in [('sqrtK_interval',scale),('integer3_tail',t3),('majorant_tail',t5)]:assert m(r[field][0])<=val<=m(r[field][1])
 assert m(r['lower_formula_endpoint'])<=value<=m(r['upper_bound'])
 assert r['qualifies']==(F(r['upper_bound'])<=F('1e-10')) and r['strictly_tighter']==(F(r['upper_bound'])<F(r['previous_bound']))
 assert r['saved_error_below_ideal_bound']==(F(r['saved_floating_absolute_error'])<=F(r['upper_bound']))
 counts.update(r['counts'])
for g in d['groups']:
 matches=[r for k,r in rows.items() if k[0]==g['panel'] and k[1]==g['fixture'] and k[3]==g['order']]
 assert len(matches)==(1 if g['panel']=='original' else 2) and g['complete']
 new=sum((F(r['upper_bound']) for r in matches),F());old=sum((F(r['previous_bound']) for r in matches),F())
 assert F(g['bound_sum_fraction'])==new and F(g['previous_sum_fraction'])==old and g['qualifies']==(new<=F('1e-10')) and g['strictly_tighter']==(new<old)
 qualified[g['panel']]+=g['qualifies'];tighter[g['panel']]+=g['strictly_tighter']
 assert g['new_over_previous'] is None if not old else F(g['new_over_previous'])==new/old
assert qualified==Counter(original=7,child=7) and tighter['child']==33
assert len(d['reused_certificate_setup'])==d['reused_certificate_count']==33
for r in d['reused_certificate_setup']:
 cert=certs[r['panel'],r['fixture'],r['side']];assert r['counts']==cert['counts'] and r['max_bits']==cert['max_bits'] and r['prior_cpu']==cert['cpu']
out=dict(complete=True,exact_controls=9,invalid_controls=4,bounds=99,groups=66,qualifying_groups=dict(qualified),strictly_tighter_groups=dict(tighter),directed_categories=dict(counts),results_sha256=sha(P/'results.json'),review_cpu=time.process_time()-start,new_mass_calls=0,scope='Independent geometric-moment nu3 tails and coefficient-majorant factor verify all directed bound intervals/aggregates. Seven of33 groups qualify in each panel; all child groups improve but severe original bounds can worsen. Ideal truncation only, no floating error/native cost or adaptive order claim.')
Path(__file__).with_name('chebyshev5_majorant_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
