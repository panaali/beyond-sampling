"""Independent exact derivative formulas and archived numerator sign audit."""
import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import hashlib,json,time
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_56/m16/approaches/cofactor_angle_bounds');start=time.process_time();mp.mp.dps=150
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f(x):return None if x is None else F(int(x[0]),int(x[1]))
def m(x):return mp.mpf(x.numerator)/x.denominator
def check(values,result):
 x,l,h,p=values
 if x is None:status='unknown_nonrational'
 elif l>h:status='invalid_interval'
 elif not l<=p<=h:status='invalid_point_outside'
 elif x<0:status='unsupported_negative_x'
 elif x==0:status='bounded_constant' if l>0 or h<0 else 'unknown_origin_possible'
 else:status='bounded'
 assert status==result['status']
 if status=='bounded':
  distance=min(abs(l),abs(h)) if l*h>0 else F(0);radius=max(p-l,h-p);slope=x/(x*x+distance*distance);raw=radius*slope
  expected=dict(bound=min(F(4),raw),raw=raw,radius=radius,distance=distance,slope=slope,uniform=radius/(2*distance) if distance else None)
  for k,v in expected.items():assert f(result[k])==v
 elif status=='bounded_constant':assert f(result['bound'])==0
 else:assert result['bound'] is None
 return status
protocol=read(P/'protocol.json');sp=read(P/'saved_protocol.json')
assert sha(P/'bound.py')==protocol['bound_sha256'] and sha(P/'controls.py')==protocol['controls_sha256']
assert sha(P/'controls.json')==sp['completed_controls_sha256'] and sha(P/'saved.py')==sp['saved_driver_sha256']
controls=read(P/'controls.json');assert controls['cases']==21 and not controls['failures']
for r in controls['rows']:
 values=list(map(f,r['inputs']));status=check(values,r['result']);assert r['ok']
 if status.startswith('bounded'):
  x,l,h,p=values
  for q in r['references']:
   delta=abs(mp.atan2(m(f(q['y'])),m(x))-mp.atan2(m(p),m(x)))
   assert delta<=m(f(r['result']['bound']))+mp.mpf('1e-140')
old=read(P/'frozen/saved_minors_results.json');man=old['manifest']
for p,h in man['pins'].items():assert sha(Path(p))==h
archive=np.load(next(p for p in man['pins'] if p.endswith('panel_nodes.npz')))
saved=read(P/'saved_results.json');assert saved['cases']==64 and not saved['failures'];counts=Counter();comparisons=Counter();bounds=[];peak=[0,0]
assert len(old['rows'])==len(saved['rows'])==64
args={label:archive[str(label)+'_atan2_arguments'] for label in man['labels']}
for oldrow,r in zip(old['rows'],saved['rows']):
 assert (r['label'],r['node'],r['pair'])==(oldrow['label'],oldrow['node'],oldrow['pair_position'])
 arr=args[r['label']];x=F(float(arr[r['pair'],1,r['node']]));source=F(float(arr[r['pair'],0,r['node']]))
 clo,chi=[F(float.fromhex(v)) for v in oldrow['interval']];l,h=-chi,-clo;p=-F(float.fromhex(oldrow['candidate_signed_point']));truth=-f(oldrow['exact_signed'])
 assert x==F(float.fromhex(r['x_hex'])) and [l,h]==list(map(f,r['signed_y_interval'])) and p==f(r['candidate_y']) and truth==f(r['exact_y']) and l<=truth<=h
 assert check([x,l,h,p],r['bound_result'])=='bounded';B=f(r['bound_result']['bound']);bounds.append(B)
 e=abs(mp.atan2(m(p),m(x))-mp.atan2(m(truth),m(x)));es=abs(mp.atan2(m(source),m(x))-mp.atan2(m(truth),m(x)))
 assert e<=m(B)
 a=r['angle_reference'];assert abs(mp.mpf(a['candidate_angle_error'])-e)<mp.mpf('1e-110') and abs(mp.mpf(a['source_angle_error'])-es)<mp.mpf('1e-110')
 cmp='tie' if e==es else ('candidate_better' if e<es else 'source_better');assert cmp==a['comparison'];comparisons[cmp]+=1
 counts.update(r['bound_result']['counts']);peak=[max(peak[i],r['bound_result']['peak_bits'][i]) for i in range(2)]
 assert r['additional_signed_wrapper_negations']==3 and r['input_dyadic_conversions']==4
 assert r['inherited_primitive_counts']==oldrow['primitive_counts'] and r['inherited_cofactor_sign_negations']==oldrow['signed_wrapper_negations']
assert counts==Counter(saved['total_bound_categories']) and peak==saved['peak_bits'] and max(bounds)==f(saved['max_bound'])
assert saved['additional_signed_wrapper_negations']==192 and saved['input_dyadic_conversions']==256 and saved['reference_angle_calls']==192 and saved['statuses']=={'bounded':64}
out=dict(complete=True,analytic_controls=21,stored_minors=64,max_angle_bound=float(max(bounds)),comparisons=dict(comparisons),categories=dict(counts),saved_sha256=sha(P/'saved_results.json'),review_cpu=time.process_time()-start,new_determinant_mass_model_calls=0,scope='Exact rational Lipschitz formula and source sign/interval correspondence independently verified. Fixed stored denominator only; high precision angles are diagnostic, no denominator/libm/quadrature or full mass certificate.')
Path(__file__).with_name('cofactor_angle_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
