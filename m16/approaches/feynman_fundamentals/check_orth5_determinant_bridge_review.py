"""Saved determinant category bridge; no native execution."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/orth5_cost_review/determinant_layout')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'bridge.json').read_text());assert r['complete'] and r['native_mass_model_MC_calls']==0
assert sha(P/'bridge.py')==r['source_sha256']
for name,h in r['dependency_pins'].items():assert sha(P/name)==h
assert '9572a0be' in r['review_premises']
pilot=json.loads((P.parent/'pilot.json').read_text());assert pilot['complete']
checked=0
for row in pilot['rows']:
 if row['branch']!='log':continue
 counts=Counter()
 for event in row['trace']['kernel_events']:
  if event['op']=='np.linalg.det':
   shape=event['inputs'][0]['shape'];assert len(shape) in (2,3) and shape[-1]==shape[-2]
   counts[shape[-1]]+=shape[0] if len(shape)==3 else 1
 assert counts=={3:3840,4:665,5:1}
 sums=Counter()
 for site in row['trace']['sites']:
  if site['unit']=='output_elements' and site['category'] in ('numeric_expression','numeric_comparison'):sums[site['category']]+=site['count']
 assert sum(sums.values())==32262
 checked+=1
assert checked==r['saved_synthetic_log_cases_checked']==3
assert r['conditional_selected_numeric_category_upper']==3840*(24+63+4+6+3)+665*(56+40+58+5+8+4)==497715
assert r['conditional_log_call_upper']==3840*3+665*4==14180
assert r['conditional_exp_call_upper']==3840+665==4505
assert r['historical_combined_named_category_upper']==497715+14180+4505==516400
assert r['source_exposed_expression_units_kept_separate']==32262
out=dict(complete=True,bridge_sha256=sha(P/'bridge.json'),source_sha256=r['source_sha256'],log_cases=checked,
 selected_numeric_categories=497715,log_calls=14180,exp_calls=4505,exposed_source_units_separate=32262,
 new_numerical_calls=0,scope='Conditional inherited selected-category envelopes only. Source expressions and named libm calls remain separate. det5, other kernels, historical binding, control/memory and complete native/runtime cost remain unresolved.')
Path(__file__).with_name('orth5_determinant_bridge_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
