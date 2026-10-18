import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
import hashlib,json,math,time
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/moment_h8_cost');start=time.process_time()
def read(p):return json.loads(p.read_text())
def count(r):
 c=Counter(r['center']['setup_selection_counts']);c.update(r.get('kernel_counts',{}));c.update(r.get('repeated_determinant_counts',{}));return c
d=read(P/'pilot.json');protocol=read(P/'protocol.json');baseline=read(P.parent/'positive_h8_cost/pilot.json');old={r['index']:r for r in baseline['rows']}
for name,key in [('pilot.py','source_sha256'),('variant.py','variant_sha256'),('protocol.json','protocol_sha256')]:assert hashlib.sha256((P/name).read_bytes()).hexdigest()==d[key]
for path,digest in protocol['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
assert d['complete'] and [r['index'] for r in d['rows']]==protocol['indices'] and len(d['rows'])==64
counts=Counter();methods=Counter();attempts=0
for r in d['rows']:
 b=old[r['index']];assert r['mode']=='returned' and r['reference']==b['reference'] and r['reference_bound']==b['reference_bound'];assert r['baseline_value']==b['value'];assert r['baseline_known_row_total']==sum(count(b).values());assert r['bound']<=1e-7/35098 and abs(r['value']-r['reference'])<=r['bound']+r['reference_bound']+5e-14;assert r['error']==r['value']-r['reference'];methods[r['method']]+=1;counts.update(count(r));c=r['center'];m=c.get('moment');attempts+=m is not None
 if r['method']=='moment_interval':
  assert m['mean']>0 and m['variance']>0;scale=c['det']/(12*math.pi**2);lower=scale/(m['mean']*m['mean']);a2=c['a']*c['a'];upper=lower+3*scale*m['variance']/(a2*a2);lo=max(c['interval'][0],lower);hi=min(c['interval'][1],upper);assert m['interval']==[lo,hi] and r['value']==(lo+hi)/2 and r['bound']==(hi-lo)/2
 else:assert r['value']==b['value'] and r['bound']==b['bound']
counts.update(d['cold_rule_counts']);counts['universal_constant_arithmetic']+=6;assert dict(counts)==d['known_counts'] and sum(counts.values())==109484==d['known_total'];assert dict(methods)==d['methods'] and attempts==35==d['moment_attempts'];assert math.fsum(r['error'] for r in d['rows'])==d['error_sum'];assert math.fsum(abs(r['error']) for r in d['rows'])==d['absolute_error_sum'];assert math.fsum(r['bound'] for r in d['rows'])==d['conditional_bound_sum'];assert 147062-109484==d['known_unit_saving']
out=dict(cases=64,methods=dict(methods),moment_attempts=attempts,unchanged_other_values=39,known_total=109484,baseline_known_total=147062,L1_error=d['absolute_error_sum'],conditional_bound_sum=d['conditional_bound_sum'],review_cpu=time.process_time()-start,scope='Saved arithmetic/source pins; no mass rerun. Original floating moment cancellation and conditional-only bounds remain. Includes failed guard costs; no full cost/certification or full corpus extrapolation.')
Path(__file__).with_name('moment_h8_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
