import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from collections import Counter
from decimal import Decimal as D,localcontext
import gzip,json,hashlib,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/stable_moment_full');start=time.process_time()
def read(p):return json.loads(p.read_text())
def gz(p):return json.loads(gzip.decompress(p.read_bytes()))
s=read(P/'summary.json');proto=read(P/'protocol.json');B=Path(proto['baseline_folder']);pilot=read(P.parent/'stable_moment_h8/pilot.json');prior={r['index']:r for r in pilot['rows']};baseline={}
for p in sorted(B.glob('shard_*.json.gz')):
 for r in gz(p)['rows']:baseline[r['index']]=r
for path,h in proto['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
rows=[]
for item in s['shards']:
 p=P/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];d=gz(p);assert d['complete'];assert d['protocol_sha256']==hashlib.sha256((P/'protocol.json').read_bytes()).hexdigest();rows+=d['rows']
assert [r['index'] for r in rows]==list(range(35098))
known=Counter();exact=Counter();methods=Counter();bymethod=Counter();attempts=reused=unchanged=rescues=0
for r in rows:
 idx=r['index'];b=baseline[idx];c=r['center'];assert r['mode']=='returned' and r['comparison_pass'] and r['bound_pass'];assert r['reference']==b['reference'] and r['reference_bound']==b['reference_bound'];assert r['error']==r['value']-r['reference'];assert r['bound']<=1e-7/35098 and abs(r['error'])<=r['bound']+r['reference_bound']+5e-14
 if r.get('reused_pilot'):
  reused+=1;assert all(r[k]==v for k,v in prior[idx].items())
 counts=Counter(c['setup_selection_counts']);counts.update(r.get('kernel_counts',{}));counts.update(r.get('repeated_determinant_counts',{}));known.update(counts);methods[r['method']]+=1;bymethod[r['method']]+=sum(counts.values())
 if r['method']!='moment_interval':
  unchanged+=1;assert r['unchanged_numerical_result'] and r['value']==b['value'] and r['bound']==b['bound']
 if 'moment' in c:
  attempts+=1;m=c['moment'];assert 'failure' not in m and m['variance']>0
  # Independently reconstruct the positive Jensen/Taylor interval and intersection.
  lo=c['det']/(12*math.pi**2*m['mean']**2);hi=lo+3*c['det']/(12*math.pi**2)*m['variance']/c['a']**4
  left=max(c['interval'][0],lo);right=min(c['interval'][1],hi)
  assert math.isclose(left,m['interval'][0],rel_tol=1e-14,abs_tol=1e-30) and math.isclose(right,m['interval'][1],rel_tol=1e-14,abs_tol=1e-30)
  if r['method']=='moment_interval':assert r['value']==sum(m['interval'])/2 and r['bound']==(m['interval'][1]-m['interval'][0])/2
 if 'rescue' in r:
  rescues+=1;z=r['rescue'];bz=b['rescue'];exact.update(z['exact_operation_inventory']['operations'])
  for k in ['order','value','series_error_bound','old_determinant']:
   if k in z:assert z[k]==bz[k],(idx,k)
  assert z['stages'][-1]['certificate']==bz['stages'][-1]['certificate']
rules={};staged=Counter();sessions=[read(p) for p in sorted(P.glob('session_*.json'))]
for session in [pilot]+sessions:
 for e in session.get('cold_rule_events',session.get('cold_rules',[])):
  if e['order'] in rules:assert rules[e['order']]==e['counts']
  rules[e['order']]=e['counts'];staged.update(e['counts'])
canonical=Counter()
for c in rules.values():canonical.update(c)
known.update(canonical);known['universal_constant_arithmetic']+=6
assert known==Counter(s['known_float_named_counts']) and sum(known.values())==121379672;assert exact==Counter(s['exact_operation_inventory']);assert methods==Counter(s['method_counts']) and bymethod==Counter(s['known_row_cost_by_method']);assert staged==Counter(s['actual_staged_rule_counts']) and canonical==Counter(s['canonical_cold_rule_counts']);assert (attempts,reused,unchanged,rescues)==(20329,64,22502,132)
with localcontext() as ctx:
 ctx.prec=65;mass=sum(D(r['value']) for r in rows);bounds=sum(D(r['bound']) for r in rows);errors=[D(r['value'])-D(r['reference']) for r in rows]
 assert float(mass)==s['error_mass'] and float(bounds)==s['covered_conditional_bound_sum'];assert abs(float(1-mass)-s['accuracy'])<2e-16;assert math.isclose(float(sum(abs(v) for v in errors)),s['covered_absolute_error_sum'],rel_tol=1e-14)
z=np.load(P/'values.npz');assert np.array_equal(z['values'],[r['value'] for r in rows]) and np.array_equal(z['bounds'],[r['bound'] for r in rows]);assert s['baseline_known_total']==read(B/'summary.json')['known_float_named_total']==142873286;assert s['known_unit_saving']==21493614 and s['known_cost_ratio']==121379672/142873286;assert not s['complete_implementation_cost'] and not s['full_floating_certificate']
out=dict(cases=len(rows),reused_pilot_once=reused,moment_attempts=attempts,methods=dict(methods),nonmoment_bitwise_unchanged=unchanged,exact_certificates_identical_to_independently_reviewed_baseline=rescues,known_counts=dict(known),known_total=sum(known.values()),saving=21493614,accuracy=s['accuracy'],conditional_bound_sum=s['covered_conditional_bound_sum'],review_cpu=time.process_time()-start,scope='All saved shards, intervals, counts, baseline/reference alignment and reuse checked without integration. Exact rescue certificates match previously independently reconstructed certificates. Floating Gram/interval errors, exact/MP/library/setup and full implementation costs remain qualified.')
Path(__file__).with_name('stable_moment_full_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
