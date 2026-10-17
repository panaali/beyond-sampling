import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import json,hashlib,math,time,ast
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/fibre_t4_recovery');start=time.process_time();d=json.loads((P/'observer_controls.json').read_text());sha=hashlib.sha256((P/'observer.py').read_bytes()).hexdigest();assert d['observer_sha256']==sha and len(d['rows'])==8
for r in d['rows']:
 old=json.loads((P/('guard_'+r['case']+'.json')).read_text());a=next(x for x in old['arms'] if x['tier']==r['tier']);assert all(r[k]==a[k] for k in ['estimate','legacy_ops','stats'])
p=json.loads((P/'noise_preflight.json').read_text());identity=json.loads((P/'noise_identity.json').read_text());assert p['observer_sha256']==sha and p['complete'];assert hashlib.sha256(b''.join(np.array(x,dtype='<f8').tobytes() for x in identity['model'])).hexdigest()==p['model_sha256']==identity['model_sha256'];assert p['family']==p['side']==0;assert p['raw_error_component']/(2*math.pi**2)==p['normalized_error_component']
assert p['observer']['trace']['calls']==p['stats']['traces']==2631;assert p['observer']['trace']['max_iterations']==11;assert p['observer']['_walk_down']['calls']==p['observer']['_walk_down']['iterations']==784;assert all(v.get('exhaustions',0)==0 and v.get('unsatisfied_depth_caps',0)==0 for v in p['observer'].values())
# Verify for-else observer semantics: final-iteration break must not count as exhaustion.
def probe(stop):
 calls=[]
 for i in range(64):
  calls.append(i)
  if i==stop:break
 else:return len(calls),True
 return len(calls),False
assert probe(63)==(64,False) and probe(-1)==(64,True) and probe(0)==(1,False)
result=dict(analytic_arms_bitwise=8,observed_noisy_side=(0,0),raw_error_component=p['raw_error_component'],normalized_error_component=p['normalized_error_component'],walk_calls=784,max_trace_iterations=11,max_subdivision_depth=p['observer']['subdivide']['max_depth'],review_cpu=time.process_time()-start,scope='Source observer semantics and saved parity/normalization only. One additive side is not full accuracy, convergence or cost; no cubature rerun.')
Path(__file__).with_name('fibre_observer_review_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
