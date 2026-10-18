import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,time
P=Path('/workspace/work/recover-small_h2_910/review/recipe112_native_cost');start=time.process_time();d=json.loads((P/'controls.json').read_text());assert len(d['operators'])==12 and len(d['models'])==5
expected=[5,9,6,20,9,6,5,3,2,1,2,3]
def ledger(r):
 c=Counter()
 for values in r['sites'].values():c.update(values)
 assert dict(c)==r['totals'];integer=sum(v for k,v in c.items() if k.startswith('integer_control'));assert r['integer_control_total']==integer and r['named_numeric_total']==sum(c.values())-integer
for r,n in zip(d['operators'],expected):ledger(r['ledger']);assert r['ledger']['named_numeric_total']==n==r['expected_numeric']
values={}
for r in d['models']:
 a,b=r['baseline'],r['traced'];assert all(a[k]==b[k] for k in ('hex','legacy','stats','info'));ledger(b['ledger']);values[r['name']]=dict(value=b['value'],named_numeric=b['ledger']['named_numeric_total'],legacy=b['legacy'])
out=dict(operator_controls=12,synthetic_models=5,models=values,review_cpu=time.process_time()-start,scope='Source theorem and independent saved site-ledger contractions; no model evaluation. Existing positive-volume tie failures retained, numeric/library counts named conventions, not full native cost.')
Path(__file__).with_name('native_recipe_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
