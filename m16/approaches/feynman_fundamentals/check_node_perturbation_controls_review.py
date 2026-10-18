"""Exact saved node/weight sensitivity primitive controls; no quadrature."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/gauss_legendre_enclosure/perturbation')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def frac(z):return F(int(z[0]),int(z[1]))
r=json.loads((P/'controls.json').read_text());p=json.loads((P/'run_protocol.json').read_text())
assert r['cases']==len(r['rows'])==16 and not r['failures']
for n,h in p['pins'].items():assert sha(Path(n))==h
expected={'zero':F(0),'node':F(46,24000),'weight':F(1,6000),'both':F(1,480),'source_sum2':F(5,8),'radial_quarter':F(5312,24000)}
sqrt_n=bound_n=status_n=0
for q in r['rows']:
 assert q['passed']
 if q['kind']=='sqrt':
  a=frac(q['a']);lo,hi=map(frac,q['interval']);k=math.isqrt((a.numerator<<320)//a.denominator)
  assert (lo,hi)==(F(k,1<<160),F(k+1,1<<160)) and lo*lo<=a<hi*hi;sqrt_n+=1
 elif q['kind']=='bound':
  assert q['name'] in expected and frac(q['expected'])==expected[q['name']]
  a=q['actual'];assert a['status']=='bounded' and frac(a['probability_upper'])==expected[q['name']]
  assert frac(a['integral_upper'])==frac(a['node_upper'])+frac(a['weight_upper'])
  assert frac(a['prefactor_upper'])==F(1,24)
  bound_n+=1
 else:
  assert q['kind']=='status' and q['actual']['status']==q['expected'];status_n+=1
assert (sqrt_n,bound_n,status_n)==(7,6,3)
out=dict(complete=True,controls=16,controls_sha256=sha(P/'controls.json'),runner_sha256=sha(P/'run.py'),protocol_sha256=sha(P/'run_protocol.json'),
 scope='Exact isqrt brackets, six algebraic values and three unknown/invalid statuses pass. Main source binds mapped nonnegative rule constants and reviewed exact radial certificates for original11x3 only; no floating kernel/map/determinant claim.',new_quadrature_calls=0)
Path(__file__).with_name('node_perturbation_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
