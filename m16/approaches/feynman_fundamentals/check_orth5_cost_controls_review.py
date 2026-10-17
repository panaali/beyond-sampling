"""Saved transparent-adapter controls; no numerical replay."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/orth5_cost_review')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'controls.json').read_text());assert r['complete'] and len(r['rows'])==8 and r['orth5_model_MC_calls']==0
for n,h in r['source_sha256'].items():assert sha(P/n)==h
expected={
'signed_scalar':{'numeric_expression:Add':2,'numeric_expression:Sub':1,'numeric_expression:Mult':1,'numeric_expression:Div':1,'numeric_expression:USub':1,'numeric_expression:UAdd':1,'opaque_power:Pow':1},
'integer_sign':{'integer_expression:Pow':1,'integer_expression:Add':1,'integer_comparison:NotEq':1},
'broadcast':{'numeric_expression:Add':6,'numeric_expression:Mult':6},
'dot':{'opaque_matmul:MatMult':1},'matrix':{'opaque_matmul:MatMult':4},
'sqrt_maximum':{'opaque_call:np.maximum':3,'opaque_call:np.sqrt':3},
'short_circuit':{'numeric_comparison:Lt':2,'numeric_comparison:Gt':2},
'builtin_libm':{'opaque_call:sum':1,'opaque_call:min':1,'opaque_call:abs':1,'opaque_call:math.atan2':1}}
for row in r['rows']:
 assert row['name'] in expected and row['baseline']==row['adapted']
 tr=row['trace'];sums=Counter();calls=Counter()
 for z in tr['sites']:
  key=z['category']+':'+z['op']
  (sums if z['unit']=='output_elements' else calls)[key]+=z['count']
 assert dict(sums)==expected[row['name']]==tr['category_output_elements']==row['expected_category_output_elements']
 opaque=Counter(k['category']+':'+k['op'] for k in tr['kernel_events'])
 assert opaque==Counter({k:v for k,v in calls.items() if k.startswith('opaque')})
 if row['name']=='short_circuit':assert row['side_effect_counts']==[2,2]
 if row['name']=='integer_sign':assert row['baseline'][-2:]==[{'type':'float','hex':'0x1.0000000000000p-5'},{'type':'int','value':9}]
out=dict(complete=True,controls=8,controls_sha256=sha(P/'controls.json'),scope='Saved output bits, hand-counted categories, site/kernel event conservation and short-circuit effects checked. Source review accepts compiler-preserved folded constants and structural restoration. Output-element census is not a native kernel price or wholemass cost.',new_numerical_calls=0)
Path(__file__).with_name('orth5_cost_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
