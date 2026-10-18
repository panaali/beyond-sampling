import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,math,time,numpy as np
start=time.process_time();P=Path('/workspace/work/recover-small_h2_34/m16/approaches/positive_h8_holdout');d=json.loads((P/'pilot.json').read_text());p=json.loads((P/'protocol.json').read_text());root=Path('/workspace/work/recover-small_h2_910/m16/approaches/frozen_recipe_holdout/exact');N=30726;tol=1e-7/N
for name,sha in p['pins'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
assert d['complete'] and d['indices']==p['indices'] and len(d['rows'])==64 and len(set(d['indices']))==64
assert d['N']==N and F(d['exact_tolerance'])==F(1,10**7*N);assert F(tol)<=F(d['exact_tolerance']) and abs(tol-float(F(d['exact_tolerance'])))<=math.ulp(tol)
ref={}
for i in range(8):
 b=np.load(root/f'batch_{i}.npz')
 for ix,v,bound in zip(b['indices'],b['positive'],b['bounds']):assert int(ix) not in ref;ref[int(ix)]=(float(v),float(bound))
assert len(ref)==N
ledger=Counter();methods=Counter();errors=[];bounds=[]
for r in d['rows']:
 assert (r['reference'],r['reference_bound'])==ref[r['index']];assert r['mode']=='returned' and r['bound']<=tol;methods[r['method']]+=1
 ledger.update(r['center']['setup_selection_counts']);ledger.update(r.get('repeated_determinant_counts',{}));ledger.update(r.get('kernel_counts',{}));c=r['center']
 if r['method']=='center_interval':assert r['bound']==c['bound'] and r['value']==c['value']
 else:
  assert r['method']=='center_quadrature';assert c['trials'][-1]['order']==c['order'] and c['trials'][-1]['bound']<=tol and all(t['bound']>tol for t in c['trials'][:-1]);assert r['bound']==c['bound']+r['series_error_bound']
 errors.append(F(r['value'])-F(r['reference']));bounds.append(F(r['bound']));assert abs(float(errors[-1]))<=r['bound']+r['reference_bound']+5e-14
ledger.update(d['cold_rule_counts']);ledger['universal_constant_arithmetic']+=6
assert dict(ledger)==d['known_float_named_counts'];assert sum(ledger.values())==d['known_float_named_total']==532864;assert dict(methods)==d['methods'];assert float(sum(map(abs,errors),F(0)))==d['absolute_error_sum'];assert float(sum(bounds,F(0)))==d['conditional_bound_sum']
out=dict(cases=64,methods=dict(methods),known_units=sum(ledger.values()),absolute_error_sum=float(sum(map(abs,errors),F(0))),bound_sum=float(sum(bounds,F(0))),tolerance_conservatively_rounded=True,review_cpu=time.process_time()-start,scope='Fixed held-out pilot and same source only, not full corpus or full model cost. N requires charged capture/geometry/storage in a future model-facing adapter.')
Path(__file__).with_name('positive_holdout_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
