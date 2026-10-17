import resource
resource.setrlimit(resource.RLIMIT_CPU,(20,20))
from pathlib import Path
import hashlib,itertools,json,math,time
import numpy as np
P=Path('/workspace/work/recover-small_h4_4/m16/approaches/hopf_native_cost');start=time.process_time()
def read(p):return json.loads(p.read_text())
d=read(P/'sort_controls.json');assert d['status']=='complete' and d['build']['returncode']==0;assert hashlib.sha256((P/'sort_shadow.cpp').read_bytes()).hexdigest()==d['cpp_sha256'];assert hashlib.sha256((P/'sort_controls.py').read_bytes()).hexdigest()==d['source_sha256'];cases=iter(d['cases']);checked=0
# Independent Python stable ordering, including explicit NaN-last semantics.
def check(a,name,dtype):
 global checked
 r=next(cases);assert r['name']==name and r['n']==len(a) and r['dtype']==dtype
 key=lambda j:(math.isnan(float(a[j])),0 if math.isnan(float(a[j])) else a[j])
 ids=np.array(sorted(range(len(a)),key=key),dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==r['permutation_sha256'];assert r['calls']<=r['scalar_comparisons']<=r['calls']*(3 if dtype=='float64' else 1);checked+=1
for n in range(7):
 for j,p in enumerate(itertools.permutations(range(n))):
  for dtype in ('float64','int64'):check(p,f'permutation_{n}_{j}',dtype)
for n in (15,16,31,32,63,64,65,127,128,257,1024):
 x=list(range(n))
 for kind,a in [('increasing',x),('reversed',x[::-1]),('duplicate',[i%5 for i in x]),('modular',[(37*i+11)%n for i in x])]:
  for dtype in ('float64','int64'):check(a,f'{kind}_{n}',dtype)
 r=next(cases);ids=np.array(sorted(x,key=lambda i:(i%7,(17*i+3)%13)),dtype=np.int64);assert r['name']==f'lexsort_{n}' and hashlib.sha256(ids.tobytes()).hexdigest()==r['permutation_sha256'];checked+=1
for i,a in enumerate(([0.,-0.,float('nan'),-float('inf'),float('inf'),1.,-1.],[float('nan')]*5,[0.,-0.,0.,-0.])):check(a,f'nonfinite_{i}','float64')
assert next(cases,None) is None and checked==1850
for r in d['search_cases']:
 n=r['n'];a=sorted((17*i)%31 for i in range(n));keys=[-math.inf,-1,0,.5,4,15,30,31,math.inf,math.nan];ids=[n if math.isnan(key) else sum(v<=key for v in a) for key in keys];assert ids==r['indices'];assert r['comparator_calls']==(10*(max((n-1).bit_length(),1)+1) if n else 0)
assert len(d['search_cases'])==40
out=dict(sort_permutations_checked=checked,search_cases_checked=40,review_cpu=time.process_time()-start,scope='Independent Python stable ordering reconstructs every saved permutation hash and direct rank counts reconstruct all search outputs. Saved comparator counts satisfy declared per-call ranges; no source shadow/native binary rerun or equivalence proof. Compiler child CPU remains unmeasured separately.')
Path(__file__).with_name('hopf_sort_saved_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
