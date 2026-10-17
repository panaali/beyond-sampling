import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_34/m16/approaches/stable_moment_h8');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'pilot.json');old=read(P.parent/'moment_h8_cost/pilot.json');prior={r['index']:r for r in old['rows']};protocol=read(P/'protocol.json')
for path,digest in protocol['pins'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
for name,key in [('pilot.py','source_sha256'),('variant.py','variant_sha256'),('protocol.json','protocol_sha256')]:assert hashlib.sha256((P/name).read_bytes()).hexdigest()==d[key]
rays=np.load('/workspace/work/recover-small_h2_910/m16/approaches/leaf_mass_review/error_captured.npz')['rays'];assert [r['index'] for r in d['rows']]==[r['index'] for r in old['rows']];tot=Counter();errs=[];improved=0;verified=0
for r in d['rows']:
 o=prior[r['index']];assert r['mode']=='returned' and r['reference']==o['reference'] and r['reference_bound']==o['reference_bound'];assert r['method']==o['method'];assert r['bound']<=1e-7/35098 and abs(r['value']-r['reference'])<=r['bound']+r['reference_bound']+5e-14
 c=Counter(r['center']['setup_selection_counts']);c.update(r.get('kernel_counts',{}));c.update(r.get('repeated_determinant_counts',{}));tot.update(c)
 if r['method']!='moment_interval':assert r['value']==o['value'] and r['bound']==o['bound']
 if 'exact_variance_validation' in r:
  v=r['exact_variance_validation'];R=[[F(float(x)) for x in row] for row in rays[r['index']]];G=[[sum(x*y for x,y in zip(a,b)) for b in R] for a in R];sums=[sum(row) for row in G];diag=[G[i][i] for i in range(4)];mu=(sum(sums)+sum(diag))/20;variance=(sum((sums[i]+diag[i]-5*mu)**2 for i in range(4))+sum((G[i][j]-mu)**2 for i in range(4) for j in range(i,4)))/210
  assert mu==F(v['exact_mean']) and variance==F(v['exact_variance']) and variance>0
  a=float((F(v['original_variance'])-variance)/variance);b=float((F(v['stable_variance'])-variance)/variance);assert a==v['original_relative_error'] and b==v['stable_relative_error'];improved+=abs(b)<abs(a);errs.append((abs(a),abs(b)));verified+=1
  assert r['center']['setup_selection_counts']['arithmetic']==o['center']['setup_selection_counts']['arithmetic']-2
 tot.update({})
tot.update(d['cold_rule_counts']);tot['universal_constant_arithmetic']+=6;assert dict(tot)==d['known_counts'] and sum(tot.values())==109414==d['known_total'];assert old['known_total']-d['known_total']==2*verified==70
assert math.fsum(r['error'] for r in d['rows'])==d['error_sum'] and math.fsum(r['bound'] for r in d['rows'])==d['conditional_bound_sum'];assert max(a for a,b in errs)==d['max_original_relative_variance_error'] and max(b for a,b in errs)==d['max_stable_relative_variance_error']
out=dict(cases=64,exact_variances_reconstructed=verified,variance_improvements=improved,one_worsening_retained=verified-improved,known_total=109414,known_saving_vs_original=70,max_old_relative_variance_error=max(a for a,b in errs),max_new_relative_variance_error=max(b for a,b in errs),review_cpu=time.process_time()-start,scope='Independent exact centered-square Gram calculation versus saved256-monomial references; all saved costs/finite outputs checked, no mass rerun. Floating Gram loss remains and no full numerical or complete-cost certificate follows.')
Path(__file__).with_name('stable_moment_pilot_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
