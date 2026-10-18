import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import json,hashlib,math
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/orth5_cost_review')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
r=read(P/'pilot.json');g=read(P/'pilot_gate.json');old=read(P/'wedge_results.json');expected=read(P/'expected_counts.json')
assert r['complete'] and r['synthetic_mass_calls']==10 and r['trained_model_MC_calls']==0 and r['dependency_gate']==g
for n,h in g['source_sha256'].items():assert sha(P/n)==h
assert r['pilot_source_sha256']==sha(P/'pilot.py')
assert [q['fixture'] for q in r['rows']]==g['fixtures']
source={q['fixture']:q for q in old['rows'] if q['arm']=='source'};fixtures={q['name']:q for q in old['fixtures']};tot=Counter();detcensus={}
for q in r['rows']:
 name=q['fixture'];assert q['saved_source_record']==source[name] and q['parity_and_hand_count_pass']
 a,b=q['arms']['baseline'],q['arms']['adapted'];branch=q['branch'];assert branch==('thin' if name.endswith('22') else 'log')
 for k in ('returned_hex','value','proxy_ops','simplices','warnings'):assert a[k]==b[k]
 assert a['returned_hex']==source[name]['returned_hex'] and a['proxy_ops']==source[name]['source_proxy_ops'] and a['simplices']==1
 tr=q['trace'];count=Counter();calls=Counter()
 for z in tr['sites']:
  key=z['category']+':'+z['op'];(count if z['unit']=='output_elements' else calls)[key]+=z['count']
 assert dict(count)==tr['category_output_elements']==expected[branch]
 events=Counter(k['category']+':'+k['op'] for k in tr['kernel_events']);assert events==Counter({k:v for k,v in calls.items() if k.startswith('opaque')})
 dets=Counter()
 for event in tr['kernel_events']:
  if event['op']=='np.linalg.det':
   sh=event['inputs'][0]['shape'];assert sh[-1]==sh[-2];dets[sh[-1]]+=math.prod(sh[:-2])
 if branch=='log':
  assert dets==Counter({5:1,4:665,3:3840})
  assert calls['opaque_call:np.linalg.norm']==25 and calls['opaque_call:np.arctan2']==30 and calls['opaque_call:np.sum']==10
  assert count['numeric_expression:Mult']==17342 and count['numeric_expression:Div']==1327 and count['numeric_expression:Add']==9662 and count['numeric_expression:Sub']==85 and count['numeric_expression:USub']==3840
 else:
  assert dets==Counter({5:1}) and calls['opaque_matmul:MatMult']==1 and calls['opaque_call:np.sum']==2
  assert count['numeric_expression:Mult']==3 and count['numeric_expression:Div']==1297
 detcensus[name]=dict(dets);tot[branch]+=1
assert tot==Counter(log=3,thin=2)
H,T=8,5;mul=T*H+(T-1)*H*H+T*H;add=(T-1)*H*(H-1)+(T-1)*H+T*(H-1);cmp=T*H+T-1+1
assert (mul,add,cmp,mul+add+cmp)==(336,291,45,672)
assert mul+add+cmp+2*H*H==800
out=dict(complete=True,fixtures=5,branch_counts=dict(tot),determinant_instances=detcensus,sampler_scalar_optimized=672,sampler_scalar_executed=800,numeric_only=[671,799],pilot_sha256=sha(P/'pilot.json'),scope='Saved bits/proxy/STATS/warnings, fullsite categories and opaque call shape census pass. Hand expression counts and scalar sampler conventions checked separately; no equivalence to installed native kernel costs, timings or trained performance. Thin numerical failures remain inherited.',new_mass_calls=0)
Path(__file__).with_name('orth5_cost_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
