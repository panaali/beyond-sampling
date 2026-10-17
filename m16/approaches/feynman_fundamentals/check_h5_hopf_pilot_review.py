"""Saved fixed H5 pilot/sampler; all sites and source allowances, no replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import ast,gzip,hashlib,json,time
P=Path('/workspace/work/recover-small_h2_910/review/hopf_h5_native')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Reuse only our pure ledger checker, never the other script's top-level audit.
helper=Path(__file__).with_name('check_h5_hopf_controls_review.py')
tree=ast.parse(helper.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='ledger')
exec(compile(ast.Module(body=[fn],type_ignores=[]),str(helper),'exec'))
start=time.process_time();p=read(P/'pilot.json');assert p['complete'] and p['phase']=='complete'
assert (p['seed'],p['copy'],p['nu'],p['nd'])==(0,0,64,128)
assert p['source_sha256']==sha(P/'pilot.py')
for name in ('pilot_scope.json','sampler_scope.json'):
 for path,h in read(P/name)['pins'].items():assert sha(Path(path))==h
models=read(P/'models.json');m=next(r for r in models['models'] if (r['seed'],r['copy'])==(0,0))
assert p['model_sha256']==m['model_sha256']
assert p['value_hex']==p['baseline_hex']==p['saved_hex']==m['saved_h64_hex']==float(p['value']).hex()
assert p['value']==p['baseline']==p['saved_value']==m['saved_h64']
assert p['legacy']==p['baseline_legacy']==p['saved_legacy']==m['saved_legacy']
assert sha(P/'pilot_ledger.json.gz')==p['ledger_sha256']
with gzip.open(P/'pilot_ledger.json.gz','rt') as f:r=json.load(f)
counts=ledger(r['report']);startup=ledger(r['startup']);l=r['report']
assert p['explicit_with_constant_setup']==counts['explicit']+startup['explicit']==208961448
assert p['conditional_source_with_constant_setup']==counts['explicit']+startup['explicit']+sum(counts['closures'].values())==321406262
assert p['startup_explicit']==startup['explicit']==1
assert p['opaque_calls']==l['opaque_calls'] and p['source_closure_count']==len(l['source_closures'])==57344
s=read(P/'sampler_control.json');assert s['complete'] and s['source_sha256']==sha(P/'sampler_control.py')
assert s['baseline_hex']==s['traced_hex'] and s['draws_consumed']==[5,5] and s['random_draws']==0
sl=ledger(s['ledger']);assert sl['explicit']==s['expected_explicit']==281*5+8*2+2==1423
assert (s['dense_indicator'],s['zero_state_optimized_indicator'])==(280,230)
assert 4*(2*5*5+2*5)+4*(2*5-1)+3+1==280 and 280-2*5*5==230
out=dict(complete=True,pilot_sha256=sha(P/'pilot.json'),ledger_sha256=p['ledger_sha256'],sampler_sha256=sha(P/'sampler_control.json'),
 value=p['value'],legacy=p['legacy'],explicit=208961448,conditional_source_allowance_total=321406262,
 closures=counts['closures'],sampler_fixed_units=1423,research_cpu=p['cpu_seconds'],review_cpu=time.process_time()-start,new_evaluator_calls=0,
 scope='Single H5 clean0 original h64 output/counter parity, complete saved sites/closures and deterministic sampler convention pass. Current model identity only; historical-byte gap, original mask defect, incomplete native/runtime cost and old75-family10M uncertainty remain. No H4 multiplier or new truth transferred.')
Path(__file__).with_name('h5_hopf_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
