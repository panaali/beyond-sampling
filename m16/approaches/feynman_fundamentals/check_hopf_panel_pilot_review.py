import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,gzip,time
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost');R=Path('/workspace/work/recover-small_h2_78/m16/approaches/bulk_hopf_review');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mh(m):return hashlib.sha256(b''.join(w.astype('<f8').tobytes(order='C') for w in m)).hexdigest()
models=read(P/'panel_models.json');assert len(models['rows'])==25 and models['noise_source_sha256']==sha(P/'noise_snapshot.py') and models['checkpoint_identity_sha256']==sha(P/'panel_checkpoint_identity.json');assert {(r['seed'],r['copy']) for r in models['rows']}=={(s,c) for s in range(5) for c in range(5)}
for seed in range(5):
 rows=sorted((r for r in models['rows'] if r['seed']==seed),key=lambda r:r['copy']);base=tuple(np.array(w,dtype=float) for w in rows[0]['model']);rng=np.random.default_rng(50)
 for i,r in enumerate(rows):
  m=base if i==0 else tuple(w+.05*rng.standard_normal(w.shape) for w in base);assert all(np.array_equal(a,np.array(b)) for a,b in zip(m,r['model'])) and mh(m)==r['model_sha256']
with gzip.open(P/'panel_s0_c1.json.gz','rt') as f:d=json.load(f)
assert d['status']=='complete' and (d['seed'],d['copy'],d['nu'],d['nd'])==(0,1,64,128)
for name,h in d['pins'].items():assert sha(R/'frozen/ev_44.json' if name=='frozen_ev_44.json' else P/name)==h
saved=read(R/'frozen/ev_44.json')['0']['rows'][1];assert d['value']==d['baseline_value']==d['saved_value']==saved['h64'];assert d['legacy']==d['baseline_legacy']==d['saved_legacy']==saved['h64_ops'];assert d['model_sha256']==next(r['model_sha256'] for r in models['rows'] if (r['seed'],r['copy'])==(0,1))
for l in (d['ledger'],d['startup']):
 c=Counter()
 for row in l['sites'].values():c.update(row)
 assert c==Counter(l['totals']);ints=sum(v for k,v in c.items() if k.startswith('integer_control'));assert ints==l['integer_control_total'] and sum(c.values())-ints==l['named_numeric_total'] and sum(c.values())==l['inclusive_explicit_total']
l=d['ledger'];by=Counter()
for r in l['source_closures']:
 if r['name']=='np.lexsort':assert r['numerical_allowance']==r['source_scalar_comparisons'] and r['source_comparator_calls']<=r['source_scalar_comparisons']<=3*r['source_comparator_calls']
 elif r['name']=='np.searchsorted':assert r['source_comparator_calls']==(r['keys']*(max((r['n']-1).bit_length(),1)+1) if r['n'] else 0) and r['numerical_allowance']==4*r['source_comparator_calls']
 else:assert r['name']=='np.argsort' and r['n']==4 and r['numerical_allowance']==157*r['rows']
 by[r['name']]+=r['numerical_allowance']
assert sum(by.values())==l['source_closure_allowance'] and len(l['source_closures'])==sum(r['count'] for r in l['opaque_calls']);assert d['explicit_with_constant_setup']==l['inclusive_explicit_total']+d['startup']['inclusive_explicit_total']==146777020;assert d['conditional_source_with_constant_setup']==146777020+sum(by.values())==247259835
out=dict(reconstructed_saved_models=25,pilot=(0,1),value=d['value'],legacy=d['legacy'],explicit=146777020,conditional_allowance=247259835,closures=dict(by),review_cpu=time.process_time()-start,scope='Saved model/noise reconstruction and complete pilot site/allowance ledger/pins/parity audited independently. First frozen noisy row only, original source, no new model/reference/MC evaluation; h128 not truth, mixed categories and runtime/FP qualifications retained.')
Path(__file__).with_name('hopf_panel_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
