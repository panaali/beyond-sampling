import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/recipe85b_native_cost');start=time.process_time()
def read(p):return json.loads(p.read_text())
for path,digest in read(P/'inventory.json')['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
def check(l):
 c=Counter()
 for v in l['sites'].values():c.update(v)
 assert dict(c)==l['totals'];integer=sum(v for k,v in c.items() if k.startswith('integer_control'));assert integer==l['integer_control_total'];assert sum(c.values())-integer==l['named_numeric_total']
d=read(P/'controls.json');expected={'unequal_float_tuples':2,'equal_distinct_float_tuples':1,'shared_identity':0,'prefix':0,'scalar_keys':1,'shared_nan_identity':0,'distinct_nan':2,'reverse':2,'stable_equal':0}
assert {r['name'] for r in d['sort_cases']}==set(expected)
for r in d['sort_cases']:check(r['ledger']);assert r['numeric']==r['ledger']['named_numeric_total']==expected[r['name']]
for r in d['models']:
 a,b=r['baseline'],r['traced'];assert all(a[k]==b[k] for k in ['hex','legacy','stats','info']);check(b['ledger']);assert b['ledger']['sort_dispatch_calls']==0
branch=read(P/'branch_control.json');a,b=branch['baseline'],branch['traced'];assert all(a[k]==b[k] for k in ['hex','legacy','stats','info']);check(b['ledger']);assert b['ledger']['sort_dispatch_calls']==12 and branch['sort_coverage']
out=dict(sort_controls=9,initial_models=5,initial_model_sort_calls=0,rational_integrated_sort_calls=12,review_cpu=time.process_time()-start,scope='Pinned flat numeric tuple callsites sort() and sort(reverse=True); source identity/equality/order semantics and saved parity. No generic key=None adapter claim, trained execution or full-cost conclusion.')
Path(__file__).with_name('recipe85b_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
