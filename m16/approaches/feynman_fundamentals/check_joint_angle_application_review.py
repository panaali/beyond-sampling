"""All64 exact stored-matrix angle records, no native determinant/mass replay."""
import resource,sys,time
resource.setrlimit(resource.RLIMIT_CPU,(20,20));sys.set_int_max_str_digits(0)
from pathlib import Path
from fractions import Fraction as F
from itertools import permutations
from collections import Counter
import hashlib,json
import numpy as np
import mpmath as mp
P=Path('/workspace/work/recover-small_h2_78/m16/approaches/joint_cofactor_angle')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iv(x):return F(x['lower']),F(x['upper'])
def det(a):
 n=len(a);out=F()
 for p in permutations(range(n)):
  t=F((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n)))
  for i,j in enumerate(p):t*=a[i][j]
  out+=t
 return out
def mpf(f):return mp.mpf(f.numerator)/f.denominator
def check_iv(x,value):
 lo,hi=iv(x);assert lo<=hi and mpf(lo)-mp.mpf('1e-130')<=value<=mpf(hi)+mp.mpf('1e-130')
def absolute(x):
 lo,hi=iv(x);return (F() if lo<=0<=hi else min(abs(lo),abs(hi)),max(abs(lo),abs(hi)))
start=time.process_time();mp.mp.dps=160
r=read(P/'application.json');s=read(P/'summary.json')
assert r['complete'] and r['cases']==len(r['rows'])==64 and not r['failures'] and r['statuses']=={'bounded':64}
for path,h in r['pins']['pins'].items():assert sha(Path(path))==h
for name,h in s['pins'].items():assert sha(P/name)==h
old=read(P/'frozen/saved_minors.json');man=old['manifest']
for path,h in man['pins'].items():assert sha(Path(path))==h
archive=next(Path(p) for p in man['pins'] if p.endswith('panel_nodes.npz'))
with np.load(archive) as a:arrays={lab:{k:a[f'{lab}_{k}'] for k in ('Rt','detRt','atan2_arguments')} for lab in man['labels']}
expected=[(lab,n,pos) for lab in man['labels'] for n in man['nodes'] for pos in man['pair_indices']]
assert [tuple(q['key']) for q in r['rows']]==expected
maxdrift=F();comparisons=Counter();maxima={};fields=['source_minus_target','denominator_only_minus_target','candidate_minus_target']
for q,prior in zip(r['rows'],old['rows']):
 lab,n,pos=q['key'];raw=arrays[lab];A=[[F(float(x)) for x in row] for row in raw['Rt'][n]]
 assert q['matrix_hex']==[[float(x).hex() for x in row] for row in raw['Rt'][n]]
 assert all(A[i][j]==A[j][i] for i in range(4) for j in range(4))
 leading=[det([row[:k] for row in A[:k]]) for k in range(1,5)];assert all(v>0 for v in leading)
 i,j=q['pair'];assert q['pair']==man['pair_order'][pos];k,l=[z for z in range(4) if z not in (i,j)]
 c=(-1)**(k+l)*det([[A[x][y] for y in range(4) if y!=l] for x in range(4) if x!=k])
 d2=A[i][i]*A[j][j]-A[i][j]*A[j][i];rad=leading[-1]*d2;g=q['target']
 assert list(map(F,g['leading']))==leading and F(g['cofactor'])==c==F(int(prior['exact_signed'][0]),int(prior['exact_signed'][1]))
 assert F(g['det2'])==d2 and F(g['radicand'])==F(g['jacobi'])==rad
 xl,xu=iv(g['denominator']);assert 0<xl<=xu and xl*xl<=rad<=xu*xu
 rl,ru=iv(g['ratio']);bounds=sorted([-c/xl,-c/xu]);assert rl<=bounds[0]<=bounds[1]<=ru
 target=mp.atan2(-mpf(c),mp.sqrt(mpf(rad)));check_iv(g['angle'],target)
 sx,sy=F(float.fromhex(q['source_x'])),F(float.fromhex(q['source_y']));assert sx>0
 assert q['source_x']==float(raw['atan2_arguments'][pos,1,n]).hex() and q['source_y']==float(raw['atan2_arguments'][pos,0,n]).hex()==prior['saved_atan2_y']
 assert q['native_det4']==float(raw['detRt'][n]).hex()
 source=mp.atan2(mpf(sy),mpf(sx));den=mp.atan2(-mpf(c),mpf(sx));cand=mp.atan2(-mpf(F(float.fromhex(prior['candidate_signed_point']))),mpf(sx))
 check_iv(q['source_exact_argument_angle'],source)
 for field,value in zip(fields,(source-target,den-target,cand-target)):
  check_iv(q[field],value);ab=absolute(q[field])
  if field not in maxima or ab[1]>maxima[field][1]:maxima[field]=(q['key'],ab[1],ab[0])
 sl,sh=iv(q['source_exact_argument_angle']);tl,th=iv(g['angle']);assert iv(q['source_minus_target'])==(sl-th,sh-tl)
 assert F(q['native_det_minus_exact'])==F(float.fromhex(q['native_det4']))-leading[-1]
 assert F(q['source_x_squared_minus_radicand'])==sx*sx-rad
 assert F(q['exact_complement_minus_unit_diagonal_surrogate'])==d2-(1-A[i][j]**2)
 drift=max(abs(A[z][z]-1) for z in range(4));assert F(q['max_diagonal_drift'])==drift;maxdrift=max(maxdrift,drift)
 sb=absolute(q['source_minus_target']);cb=absolute(q['candidate_minus_target'])
 comparisons['candidate_better' if cb[1]<sb[0] else 'source_better' if sb[1]<cb[0] else 'overlap_unresolved']+=1
assert dict(comparisons)==s['strict_interval_comparison']=={'candidate_better':38,'source_better':23,'overlap_unresolved':3}
assert maxdrift==F(s['max_diagonal_drift'])==F(1,1<<52)
for field,(key,upper,lower) in maxima.items():
 saved=s['extrema'][field];assert saved['key']==key and F(saved['absolute_upper'])==upper and F(saved['absolute_lower'])==lower
mat=r['accounting']['matrix_categories']
assert {k:mat[k] for k in ('determinant_1','determinant_2','determinant_3','determinant_4','permutation_terms','cofactor_sign','outward_dyadic_endpoints')}==dict(determinant_1=64,determinant_2=128,determinant_3=256,determinant_4=64,permutation_terms=3392,cofactor_sign=192,outward_dyadic_endpoints=512)
assert s['control_attempts']==15 and s['unique_controls']==12
out=dict(complete=True,cases=64,application_sha256=sha(P/'application.json'),summary_sha256=sha(P/'summary.json'),comparison=dict(comparisons),maxima={k:dict(key=v[0],upper_display=float(v[1])) for k,v in maxima.items()},review_cpu=time.process_time()-start,new_native_determinant_mass_model_calls=0,
 scope='All stored matrices/signs, exact determinants/radicands, sqrt and ratio brackets, differences, source-array bindings and exact comparison extrema audited;160-digit angle diagnostics are separate from the reviewed rational-enclosure proof. Stored-matrix argument angles only, not source covariance/coefficients/libm/quadrature/wholemass errors.')
Path(__file__).with_name('joint_angle_application_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
