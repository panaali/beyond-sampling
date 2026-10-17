"""Exact all6144 stored determinant/angle/weighted ceiling audit; no native kernels."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,itertools,time
import numpy as np
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/cofactor_angle_bounds');start=time.process_time()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x):return None if x is None else F(*map(int,x))
p=read(P/'weighted_protocol.json');r=read(P/'weighted_results.json');s=read(P/'weighted_summary.json');old=read(P/'frozen/saved_minors_results.json');ctrl=read(P/'weighted_controls.json')
for path,h in p['pins'].items():assert sha(path)==h
assert sha(P/'weighted_controls.json')==p['completed_controls_sha256'] and ctrl['cases']==17 and not ctrl['failures']
for q in ctrl['rows']:
 t=f(q['input']);u=int(q['units']);assert u==-((-t.numerator*(1<<160))//t.denominator)
assert r['complete'] and r['cases']==6144 and r['new_guard_calls']==6080 and r['reused_guard_cases']==64
reuse={(z['label'],z['pair_position'],z['node']):z for z in old['rows']}
with np.load(p['nodes_path']) as a:
 raw={label:{key:a[f'{label}_{key}'] for key in ('Rt','detRt','atan2_arguments')} for label in p['labels']}
R={label:[[[F(float(x)) for x in line] for line in node] for node in v['Rt']] for label,v in raw.items()}
pairs=list(itertools.combinations(range(4),2));seen=set();units=Counter();sums=Counter();pc=Counter();ac=Counter();statuses=Counter();maxterm={};reusecount=0
for z in r['rows']:
 label,pair,node=z['label'],z['pair'],z['node'];key=(label,pair,node);assert key not in seen;seen.add(key)
 i,j=pairs[pair];k,l=[q for q in range(4) if q not in (i,j)];ck=[q for q in range(4) if q!=k];cl=[q for q in range(4) if q!=l];sign=(-1)**(k+l)
 A=[[R[label][node][u][v] for v in cl] for u in ck]
 det=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);truth=sign*det
 lo,hi=[F(float.fromhex(v)) for v in z['signed_cofactor_interval']];point=F(float.fromhex(z['candidate_signed_point']));assert lo<=truth<=hi and lo<=point<=hi and sign==z['cofactor_sign']
 assert z['reused']==(key in reuse)
 if key in reuse:
  q=reuse[key];assert z['signed_cofactor_interval']==q['interval'] and z['candidate_signed_point']==q['candidate_signed_point'] and z['primitive_counts']==q['primitive_counts'];reusecount+=1
 x=F(float.fromhex(z['x_hex']));assert x==F(float(raw[label]['atan2_arguments'][pair,1,node])) and x>0
 low,high,pnt=-hi,-lo,-point;distance=min(abs(low),abs(high)) if low*high>0 else F();radius=max(pnt-low,high-pnt);slope=x/(x*x+distance*distance);bound=min(F(4),slope*radius)
 a=z['angle'];assert a['status']=='bounded' and f(a['bound'])==bound and f(a['raw'])==slope*radius and f(a['radius'])==radius and f(a['distance'])==distance and f(a['slope'])==slope
 term=abs(F(float.fromhex(z['weight_hex'])))*bound;assert term==f(z['weighted_term']);u=-((-term.numerator*(1<<160))//term.denominator);assert u==int(z['ceiling_units'])
 assert term<=F(u,1<<160)<term+F(1,1<<160)
 units[label]+=u;sums[label]+=term;maxterm[label]=max(maxterm.get(label,F()),term);pc.update(z['primitive_counts']);ac.update(a['counts']);statuses[a['status']]+=1
assert seen==set(itertools.product(p['labels'],range(6),range(128))) and reusecount==64
assert dict(pc)==r['primitive_counts_once'] and dict(ac)==r['angle_counts'] and dict(statuses)==r['statuses']
for c in r['cones']:
 label=c['label'];upper=F(units[label],1<<160)
 assert not c['unknown'] and f(c['conditional_bound'])==f(c['known_terms_bound'])==upper and f(c['max_term'])==maxterm[label]
 assert f(c['rounding_slack_upper'])==F(768,1<<160) and 0<=upper-sums[label]<F(768,1<<160)
 assert float(upper)==s['conditional_bounds'][str(label)]
progress=[json.loads(line) for line in (P/'weighted_progress.jsonl').read_text().splitlines()]
parities=[q for q in progress if q.get('stage')=='coefficient_parity'];assert [q['label'] for q in parities]==p['labels'] and all(q['Rt'] and q['denominators'] for q in parities)
assert r['new_setup_3det_calls']==r['new_setup_norm4_calls']==128 and r['new_setup_dot4_calls']==32 and r['new_setup_matrix4_products']==8
assert r['wrapper_actual_named_counts']['numerator_sign_negations']==18432 and r['wrapper_actual_named_counts']['input_dyadic_conversions']==30720
out=dict(complete=True,exact_archived_determinants=6144,source_coefficient_parity_records=8,exact_weighted_contractions=6144,reused=64,new=6080,conditional_bounds={str(k):float(F(v,1<<160)) for k,v in units.items()},max_bound=max(float(F(v,1<<160)) for v in units.values()),results_sha256=sha(P/'weighted_results.json'),review_cpu=time.process_time()-start,new_native_or_mass_calls=0,scope='All6144 exact stored minors enclosed, signed fixed-denominator derivative bounds and exact selected-dyadic weighted upward sums independently checked. Coefficient Rt/x parity follows pinned source and preserved records; weights are selected binary inputs, not exact original-network coefficients. Denominator/coefficient/libm/sequential sum/quadrature and full mass errors excluded.')
Path(__file__).with_name('weighted_cofactor_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
