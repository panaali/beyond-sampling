"""Saved exact matrix-angle controls, including preserved serialization retry."""
import resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(20,20));sys.set_int_max_str_digits(0)
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
import hashlib,json
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/joint_cofactor_angle')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def det(a):
 n=len(a);v=F()
 for p in permutations(range(n)):
  t=F((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)))
  for i,j in enumerate(p):t*=a[i][j]
  v+=t
 return v
def interval(x):return F(x['lower']),F(x['upper'])
a=json.loads((P/'controls.json').read_text());b=json.loads((P/'retry_controls.json').read_text())
assert not a['complete'] and a['failures']==['positive','negative','drift'] and a['cases']==12
assert b['complete'] and b['cases']==3 and not b['failures']
for source in (a,b):
 for n,h in source['pins'].items():assert sha(P/n)==h
for r in a['rows']:
 if 'exception' in r:assert 'ValueError: Exceeds the limit (4300 digits)' in r['exception'] and 'serial' in r['exception']
rows=[r for r in a['rows'] if 'exception' not in r]+b['rows']
assert len(rows)==len({r['label'] for r in rows})==12
assert {r['label'] for r in b['rows']}==set(a['failures'])
bounded=unknown=0;mp.mp.dps=180
for r in rows:
 A=[[F(x) for x in row] for row in r['matrix']];g=r['result']
 assert g['status']==r['expected']
 if g['status']=='unknown_asymmetric':assert any(A[i][j]!=A[j][i] for i in range(4) for j in range(4));unknown+=1;continue
 leading=[det([row[:n] for row in A[:n]]) for n in range(1,5)]
 assert list(map(F,g['leading']))==leading[:len(g['leading'])]
 if g['status']=='unknown_not_spd':assert any(v<=0 for v in leading);unknown+=1;continue
 assert g['status']=='bounded' and all(v>0 for v in leading)
 i,j=r['pair'];k,l=[z for z in range(4) if z not in (i,j)]
 minor=[[A[x][y] for y in range(4) if y!=l] for x in range(4) if x!=k]
 c=(-1)**(k+l)*det(minor);d2=A[i][i]*A[j][j]-A[i][j]*A[j][i];rad=leading[-1]*d2
 assert F(g['cofactor'])==c and F(g['det2'])==d2 and F(g['radicand'])==F(g['jacobi'])==rad
 xl,xu=interval(g['denominator']);assert 0<xl<=xu and xl*xl<=rad<=xu*xu
 rl,ru=interval(g['ratio']);exact_ends=sorted([-c/xl,-c/xu]);assert rl<=exact_ends[0]<=exact_ends[1]<=ru
 lo,hi=interval(g['angle'])
 if r['label'].startswith('identity'):assert lo==hi==0
 else:
  ref=mp.pi/6 * (-1 if r['label']=='negative' else 1)
  assert mp.mpf(lo.numerator)/lo.denominator<=ref<=mp.mpf(hi.numerator)/hi.denominator
 bounded+=1
assert (bounded,unknown)==(9,3)
out=dict(complete=True,unique_cases=12,bounded=bounded,unknown=unknown,executed_attempts=15,
 controls_sha256=sha(P/'controls.json'),retry_sha256=sha(P/'retry_controls.json'),
 original_cpu=a['cpu_seconds'],retry_cpu=b['cpu_seconds'],new_matrix_or_native_calls=0,
 scope='Exact leading minors, signed cofactors/radicands, square-root and ratio containment checked independently. Nonzero analytic angles checked against pi/6 at180 digits; enclosure theorem derives from reviewed exact interval primitive. Original serialization failures and repeated work retained; no source floating or mass claim.')
Path(__file__).with_name('joint_angle_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
