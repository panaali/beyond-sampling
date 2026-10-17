import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
import json,hashlib,time
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/constant_output_shortcut');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'controls.json');assert d['complete'] and d['primitive_sha256']==hashlib.sha256((P/'primitive.py').read_bytes()).hexdigest();cnt=Counter()
for r in d['rows']:
 wi,wh,wo=r['model_weights'];H=len(wi);T=len(wo);zero=all(F(x)==0 for x in wi);dom=all(F(wo[0][i])>=F(row[i]) for row in wo[1:] for i in range(H));status='certified' if zero or dom else 'fallback';assert status==r['status']==r['certificate']['status'];cnt[status]+=1
 size=H+H*H+T*H;expected=dict(metadata_predicate=15,finite_checks=size,boolean_reduction=size-1+H-1,scalar_comparison=H)
 if not zero:expected['boolean_reduction']+=(T-1)*H-1;expected['scalar_comparison']+=(T-1)*H
 if status=='certified':expected['scalar_division']=1;assert r['certificate']['prediction']==0 and r['certificate']['exact_accuracy']==[1,T] and r['certificate']['accuracy']==1/T and r['observed_labels']==[0]
 else:assert r['returned_original_fallback_object']
 assert r['certificate']['counts']==expected and r['model_unchanged'] and r['not_an_accuracy_sample']
assert cnt==dict(certified=10,fallback=5) and len(d['invalid'])==13 and all(r['result']['status']=='rejected' for r in d['invalid']);assert sum(r['validation_forward_inputs'] for r in d['rows'])==d['validation_inputs']==1029
assert d['prior_alloff_failure_preserved']['N']==72 and not d['prior_alloff_failure_preserved']['complete'] and d['no_geometry_or_mass_calls'] and d['estimator_random_inputs']==0
trained=read(P/'trained.json');assert trained['complete'] and trained['primitive_sha256']==d['primitive_sha256'];assert len(trained['rows'])==2
for r in trained['rows']:assert r['recognizer']['status']=='fallback' and r['recognizer']['counts']==dict(metadata_predicate=15,finite_checks=104,boolean_reduction=133,scalar_comparison=32) and r['original_model_forwarded'] and r['original_result_object_returned'] and r['no_geometry_mass_replay']
out=dict(valid=15,certified=10,fallback=5,invalid_rejected=13,alloff_numeric_units=31,trained_numeric_units=269,metadata_predicates=15,review_cpu=time.process_time()-start,scope='Independently rederived every stored valid exact-binary predicate and named count, checked invalid/identity-control results and preservation of previous72-cone failure. Proof requires zero-initial bias-free ReLU, first-index ties, exchangeability with almost-sure distinct inputs and real arithmetic; no finite-overflow/fullcost guarantee or trained geometry replay.')
Path(__file__).with_name('constant_shortcut_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
