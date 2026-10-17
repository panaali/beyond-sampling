import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,Context,ROUND_CEILING
from collections import Counter
import json,gzip,hashlib,time,math
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/thin_duffy_bound/directed');start=time.process_time()
def read(p):return json.loads(p.read_text())
s=read(P/'summary.json');proto=read(P/'protocol.json');old={r['index']:r for p in P.parent.glob('shard_*.json.gz') for r in json.loads(gzip.decompress(p.read_bytes()))['rows']};rescues={r['index']:r for r in read(P.parent/'rescue/results.json')['rows']};rows=[]
for item in s['shards']:
 p=P/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];d=json.loads(gzip.decompress(p.read_bytes()));assert d['complete'];rows+=d['rows']
assert [r['index'] for r in rows]==proto['indices']==sorted(old);mp.mp.dps=100;counts=Counter(s['cold_once_counts']);upper=F(0);ctx=Context(prec=50,rounding=ROUND_CEILING);directed_sum=Decimal(0)
for r in rows:
 idx=r['index'];c=rescues[idx]['certificate'] if idx in rescues else old[idx];a,b,det=F(c['a']),F(c['b']),abs(F(c['det']));al,bu=F(r['radial_lower_used']),F(r['radial_upper_used']);assert 0<al<=a<=b<=bu;assert r['reused_kkt_rescue']==(idx in rescues);lo,hi=F(r['lower_formula_endpoint']),F(r['upper_bound']);assert 0<=lo<=hi<=F(1,350980000000) and r['qualifies'];upper+=hi;directed_sum=ctx.add(directed_sum,Decimal(r['upper_bound']));counts.update(r['counts'])
 m=lambda v:mp.mpf(v.numerator)/v.denominator;af,bf=m(al),m(bu);delta=mp.sqrt(af*bf);rho=(mp.sqrt(bf)-mp.sqrt(af))/(mp.sqrt(bf)+mp.sqrt(af));bound=m(det)*2*rho**9/delta**3*((af+bf)/(2*(1-rho))+delta*(9/(1-rho)+rho/(1-rho)**2))/(6*mp.pi**2);assert m(lo)<=bound<=m(hi)
counts['directed_add']+=2*len(rows);assert counts==Counter(s['known_directed_category_counts']);assert upper==F(s['exact_sum_of_upper_endpoints']) and str(directed_sum)==s['directed_aggregate_upper'] and F(directed_sum)>=upper
# Separate rational upper construction at32 fixed extrema/endpoints; no Decimal or MP rounding assumptions.
def sqrt_bounds(x):
 scale=1<<512;k=math.isqrt(x.numerator*scale*scale//x.denominator);return F(k,scale),F(k+1,scale)
def atan_bounds(q):
 v=sum((F((-1)**j,(2*j+1)*q**(2*j+1)) for j in range(120)),F(0));w=v+F(1,241*q**241);return v,w
l5,u5=atan_bounds(5);l239,u239=atan_bounds(239);pi_lower=16*l5-4*u239;assert pi_lower>3
chosen=sorted(rows,key=lambda r:F(r['upper_bound']),reverse=True)[:30]+[rows[0],rows[-1]]
for r in chosen:
 idx=r['index'];c=rescues[idx]['certificate'] if idx in rescues else old[idx];a,b,det=F(c['a']),F(c['b']),abs(F(c['det']));sa,_=sqrt_bounds(a);_,sb=sqrt_bounds(b);delta,_=sqrt_bounds(a*b);rho=(sb-sa)/(sb+sa);alpha=(a+b)/2;tail=2*rho**9/delta**3*(alpha/(1-rho)+delta*(9/(1-rho)+rho/(1-rho)**2));bound=det*tail/(6*pi_lower*pi_lower);assert bound<=F(r['upper_bound'])
out=dict(cases=len(rows),all_expanded_ranges_and_endpoints_checked=True,independent_rational_upper_checks=len(chosen),directed_aggregate_upper=str(directed_sum),exact_KKT_rescues_reused=len(rescues),review_cpu=time.process_time()-start,scope='All saved outward ranges,100digit expanded-range formula enclosures, exact endpoint sum and independent ceiling sum checked.32 rational sqrt/Machin-pi upper constructions also lie below saved upper endpoints. Certifies ideal GL10 truncation arithmetic under reviewed Decimal semantics only; source floating determinant/nodes/kernel/geometry and log branch excluded.')
Path(__file__).with_name('directed_thin_bound_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
