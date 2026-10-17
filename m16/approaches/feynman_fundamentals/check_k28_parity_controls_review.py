"""Saved fifteen-case k28 parity and complete site ledger audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,time
P=Path('/workspace/work/recover-small_h4_4/review/margin_k28_native')
D=Path('/workspace/work/recover-small_h2_78/m16/approaches/margin_training_recovery/frozen')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();r=read(P/'controls.json');assert r['complete'] and len(r['rows'])==15
assert r['driver_sha256']==sha(P/'controls.py') and r['gate_sha256']==sha(P/'controls_gate.json')
assert r['reconstruction_sha256']==sha(P/'reconstruction.json')
gate=read(P/'controls_gate.json');assert gate['complete']
for p,h in gate['pins'].items():assert sha(Path(p))==h
recon=read(P/'reconstruction.json');margin=read(D/'margin_k28.json');rung=read(D/'rung_k28.json')
totals=Counter();cases=[]
expected=[(s,n,j) for s in range(5) for n,j in [(None,None),(20,0),(23,255)]]
assert [(q['seed'],q['noise'],q['copy']) for q in r['rows']]==expected
for q in r['rows']:
 seed,n,j=q['seed'],q['noise'],q['copy'];identity=recon['records'][seed]
 idx=-1 if n is None else (n-20)*256+j
 truth=rung['seeds'][str(seed)]['clean']['truth'] if idx<0 else rung['seeds'][str(seed)]['panel']['truths'][idx]
 modelhash=identity['base_sha256'] if idx<0 else identity['noises'][n-20]['hashes'][j]
 assert q['index']==idx and q['truth']==truth and q['model_sha256']==modelhash
 assert set(q['recipes'])=={'r85b','r112'}
 for recipe,rr in q['recipes'].items():
  assert rr['parity'] and rr['old_error_match']
  b,t=rr['baseline'],rr['traced']
  for k in ('value','hex','legacy','stats','info','recipe'):assert b[k]==t[k]
  assert float(b['value']).hex()==b['hex']
  old=margin['seeds'][str(seed)]
  err=old['clean'][recipe]['err'] if idx<0 else old['p20'][recipe]['errs'][idx]
  assert b['value']-truth==err==rr['saved_error']==rr['actual_error']
  if idx<0:assert b['legacy']==old['clean'][recipe]['ops'] and b['stats']['cells']==old['clean'][recipe]['cells']
  l=t['ledger'];c=Counter()
  for sites in l['sites'].values():c.update(sites)
  assert c==Counter(l['totals'])
  integers=sum(v for k,v in c.items() if k.startswith('integer_control'));numeric=sum(c.values())-integers
  assert (numeric,integers)==(l['named_numeric_total'],l['integer_control_total']) and not l['complete_implementation_cost']
  totals[recipe+'_numeric']+=numeric;totals[recipe+'_integer']+=integers
  cases.append(dict(seed=seed,noise=n,copy=j,recipe=recipe,numeric=numeric,integer=integers,stored_error=err))
assert len({q['model_sha256'] for q in r['rows']})==15
assert r['historical_identity_verified'] is False and (r['sampler_numeric'],r['sampler_legacy'])==(111,72)
out=dict(complete=True,models=15,recipe_outputs=30,baseline_and_traced_calls=60,controls_sha256=sha(P/'controls.json'),
 source_sha256=r['driver_sha256'],gate_sha256=r['gate_sha256'],totals=dict(totals),rows=cases,
 research_cpu=r['cpu'],review_cpu=time.process_time()-start,new_evaluator_calls=0,
 scope='All fixed saved outputs, subtraction errors, clean counters/cells, source/model pins and complete traced site sums pass. Current canonical models agree on these controls; missing historical model hashes and per-copy noisy costs remain unproved. No broad-panel or native-runtime claim.')
Path(__file__).with_name('k28_parity_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
