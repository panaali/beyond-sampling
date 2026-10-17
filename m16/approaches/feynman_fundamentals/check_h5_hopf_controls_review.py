"""Saved H5 synthetic tracer controls and conditional closure accounting."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import hashlib,json
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ledger(l):
 c=Counter()
 for sites in l['sites'].values():c.update(sites)
 assert c==Counter(l['totals'])
 ints=sum(v for k,v in c.items() if k.startswith('integer_control_'))
 assert ints==l['integer_control_total'] and sum(c.values())-ints==l['named_numeric_total'] and sum(c.values())==l['inclusive_explicit_total']
 by=Counter()
 for r in l.get('source_closures',[]):
  if r['name']=='np.lexsort':assert r['numerical_allowance']==r['source_scalar_comparisons'] and r['source_comparator_calls']<=r['source_scalar_comparisons']<=3*r['source_comparator_calls']
  elif r['name']=='np.searchsorted':
   calls=r['keys']*(max((r['n']-1).bit_length(),1)+1) if r['n'] else 0
   assert r['source_comparator_calls']==calls and r['numerical_allowance']==4*calls
  else:assert r['name']=='np.argsort' and r['n']==4 and r['numerical_allowance']==157*r['rows']
  by[r['name']]+=r['numerical_allowance']
 assert sum(by.values())==l.get('source_closure_allowance',0)
 if 'source_closures' in l:assert len(l['source_closures'])==sum(z['count'] for z in l['opaque_calls'])
 return dict(explicit=sum(c.values()),closures=dict(by))
r=json.loads((P/'controls.json').read_text());scope=json.loads((P/'controls_scope.json').read_text())
assert r['complete'] and r['source_sha256']==sha(P/'controls.py') and len(r['rows'])==8
for p,h in scope['pins'].items():assert sha(Path(p))==h
checks=[]
for ix,row in enumerate(r['rows']):
 assert row['status']=='complete'
 counts=ledger(row['ledger'])
 if ix<6:
  assert row['baseline']==row['traced'] and row['baseline_legacy']==row['legacy'];ledger(row['startup'])
  if ix<5:
   assert row['case']=='circle' and row['model']==ix
   w=row['weights'];assert len(w[0])==len(w[1])==5 and all(len(x)==5 for x in w[1]+w[2]) and len(w[2])==4
  else:assert row['case']=='dense_grid' and (row['nu'],row['nd'])==(2,3)
 else:
  outputs=row['outputs'];assert row['case']=='einsum_shape' and outputs==(5 if ix==6 else 4)
  expected=[[[sum((F(i*5+k-3,8)*F(n*15+k*3+j-11,16) for k in range(5)),F()) for j in range(3)] for i in range(outputs)] for n in range(2)]
  assert expected==[[[F(x) for x in a] for a in batch] for batch in row['value']]
  assert counts['explicit']==2*outputs*3*9
  assert row['expected_multiplications']==2*outputs*3*5 and row['expected_additions']==2*outputs*3*4
 checks.append(counts)
assert r['rows'][0]['original_fraction']==.125
out=dict(complete=True,controls=8,controls_sha256=sha(P/'controls.json'),source_sha256=r['source_sha256'],counts=checks,research_cpu=r['cpu_seconds'],new_estimator_calls=0,
 scope='All saved baseline/traced bits/legacy, complete site and conditional closure sums, H5 shapes and exact dyadic einsum values verified. Padded original-mask defect remains .125. Single clean0 historical parity pilot remains a separate bounded test; no full native or panel transfer.')
Path(__file__).with_name('h5_hopf_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
