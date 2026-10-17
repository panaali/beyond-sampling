"""Independent tail-bound integral and saved constant-extension arithmetic."""
from pathlib import Path
import json,time,resource
import mpmath as m
resource.setrlimit(resource.RLIMIT_CPU,(20,20));here=Path(__file__).parent;src=Path('/workspace/work/recover-small_h4_4/m16/approaches/plackett_endpoint_review');d=json.loads((src/'fixture.json').read_text());prior=json.loads((src.parent/'plackett_precision_review/fixture.json').read_text());start=time.process_time();m.mp.dps=70;C=m.matrix([[m.mpf(x) for x in r] for r in prior['rows'][-1]['exact_ray_C']]);r=d['rows'][-1];rows=[]
for t in r['tails']:
 a=m.mpf(t['t']);bound=m.fsum(m.quad(lambda z,rr=abs(C[i,j]):rr/m.sqrt(1-(rr*z)**2),[a,1]) for i in range(4) for j in range(i+1,4))/(8*m.pi);assert abs(bound-m.mpf(t['bound']))<m.mpf('1e-60');D=bound+(1-a)*abs(m.mpf(t['derivative']));assert abs(D-m.mpf(t['constant_extension_change_bound']))<m.mpf('1e-60');error=m.mpf(r['reference_error']);assert t['extension_test_excludes_sole_endpoint']==(t['name']=='last_node' and D<error);rows.append(dict(cut=t['name'],tail_bound=float(bound),constant_extension_bound=float(D),error=float(error),test_excludes=t['extension_test_excludes_sole_endpoint']))
eigs=list(map(m.mpf,r['eigenvalues']));assert min(eigs)>0 and abs(sum(eigs)-sum(C[i,i] for i in range(4)))<m.mpf('1e-60');assert abs(m.fprod(eigs)-m.det(C))<m.mpf('1e-60');condition=max(eigs)/min(eigs);assert abs(condition/m.mpf(r['condition_number'])-1)<m.mpf('1e-60')
out=dict(rows=rows,minimum_eigenvalue=float(min(eigs)),condition_number=float(condition),review_cpu=time.process_time()-start,scope='Conditional real-arithmetic tail bound checked by independent integral; last-node constant-tail test inconclusive. Smaller tails do not measure quadrature-error shares; no full floating certificate.')
(here/'endpoint_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
