"""Independent saved certificate and analytic-matrix checks; no author imports."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations, permutations
from collections import Counter
import hashlib,json,resource,time
import mpmath as mp
resource.setrlimit(resource.RLIMIT_CPU,(20,20)); start=time.process_time();mp.mp.dps=180
ROOT=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_pairs');OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def unpack(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(t,str) and t.lstrip('-').isdigit() for t in x):return F(int(x[0]),int(x[1]))
  return [unpack(t) for t in x]
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 return x
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def det(a):return dot(a[0],cross(a[1],a[2]))
def rank(a):
 if any(det(list(rs)) for rs in combinations(a,3)):return 3
 if any(any(cross(*rs)) for rs in combinations(a,2)):return 2
 return int(any(any(r) for r in a))
def canon(a):
 d=abs(next(x for x in a if x));return tuple(x/d for x in a)
def q(x):return mp.mpf(x.numerator)/x.denominator
P=json.loads((ROOT/'protocol.json').read_text());D=json.loads((ROOT/'results.json').read_text())
assert sha(ROOT/'protocol.json')==D['protocol_sha256']
for p,h in P['pins'].items():assert sha(Path(p))==h
assert D['complete'] and D['all_pass'] and len(D['rows'])==24
progress=[json.loads(s) for s in (ROOT/'progress.jsonl').read_text().splitlines()];assert len(progress)==48
counts=Counter();statuses=Counter();checks=[];maxerr=mp.mpf(0)
for idx,row in enumerate(D['rows']):
 assert row['passed'] and progress[2*idx]['name']==row['name'] and progress[2*idx+1]['row']==row
 if row['kind']=='invalid':assert row['exception']=='ValueError';statuses['invalid']+=1;continue
 c=unpack(row['certificate']);counts.update(c['counts']);statuses[c['status']]+=1
 normals=[]
 for n in c['source']:
  if any(n) and canon(n) not in normals:normals.append(canon(n))
 assert normals==[tuple(n) for n in c['normals']] and rank(normals)==c['normal_rank']
 if rank(normals)<3:assert c['status']=='unsupported_lineality';continue
 rays=set()
 for a,b in combinations(normals,2):
  r=cross(a,b)
  if any(r):
   for s in (1,-1):
    v=[s*x for x in r]
    if all(dot(n,v)>=0 for n in normals):rays.add(canon(v))
 assert sorted(rays)==[tuple(r) for r in c['rays']] and rank(list(rays))==c['ray_rank']
 if c['status']=='pointed_full':
  center=[sum(r[k] for r in rays) for k in range(3)];assert center==c['center']
  expectedfacets={i for i,n in enumerate(normals) if sum(dot(n,r)==0 for r in rays)==2}
  assert expectedfacets=={f['normal_index'] for f in c['facets']}
  for f in c['facets']:
   n=normals[f['normal_index']];r,s=[c['rays'][i] for i in f['ray_indices']]
   assert list(n)==f['normal'] and r==f['r'] and s==f['s'] and dot(n,r)==dot(n,s)==0
   assert dot(center,cross(r,s))==f['triangle']['det']>0
   for key,u,v in [('tangent_r',r,s),('tangent_s',s,r)]:
    t=[y-dot(u,v)/dot(u,u)*x for x,y in zip(u,v)];assert t==f[key] and dot(t,t)==f[key+'_norm2'] and dot(t,v)>0
   exact={'center_norm2':dot(center,center),'r_norm2':dot(r,r),'s_norm2':dot(s,s),'cr':dot(center,r),'rs':dot(r,s),'sc':dot(s,center)}
   assert all(f['triangle'][k]==v for k,v in exact.items()) and f['normal_norm2']==dot(n,n)
 M=mp.matrix(3,3);kind=row['kind'];rotation=mp.eye(3)
 if row['name'].startswith('rotated'):rotation=mp.matrix([[mp.mpf(3)/5,-mp.mpf(4)/5,0],[mp.mpf(4)/5,mp.mpf(3)/5,0],[0,0,1]])
 if row['name'].startswith('reflected'):rotation=mp.diag([-1,1,-1])
 if kind=='zero':p=mp.mpf(0)
 elif kind=='orthant':
  p=mp.mpf(1)/8
  for i in range(3):
   for j in range(3):M[i,j]=p if i==j else 1/(4*mp.pi)
 elif kind=='square':
  p=mp.mpf(1)/12;delta=(1-1/mp.sqrt(3))/(4*mp.pi)
  M[0,0]=M[1,1]=p+delta;M[2,2]=p-2*delta;M[0,1]=M[1,0]=1/(2*mp.pi*mp.sqrt(3))
 elif kind=='wedge':
  a=mp.atan2(q(c['source'][1][0]),-q(c['source'][1][1]));p=a/(4*mp.pi)
  M[0,0]=(a+mp.sin(2*a)/2)/(4*mp.pi);M[1,1]=(a-mp.sin(2*a)/2)/(4*mp.pi);M[2,2]=p
  M[0,1]=M[1,0]=mp.sin(a)**2/(4*mp.pi);M[0,2]=M[2,0]=mp.sin(a)/(4*mp.pi);M[1,2]=M[2,1]=(1-mp.cos(a))/(4*mp.pi)
 else:raise AssertionError(kind)
 M=rotation*M*rotation.T
 for saved in ('reference','value120','value80'):
  v=row[saved];tol=mp.mpf('1e-75') if saved=='value80' else mp.mpf('1e-110')
  err=max([abs(mp.mpf(v['mass'])-p)]+[abs(mp.mpf(v['matrix'][i][j])-M[i,j]) for i in range(3) for j in range(3)])
  assert err<tol,(row['name'],saved,str(err))
  if saved=='value120':maxerr=max(maxerr,err)
 v=row['value120'];A=mp.matrix(v['matrix']);mass=mp.mpf(v['mass'])
 assert abs(sum(A[i,i] for i in range(3))-3*mass)<mp.mpf('1e-110')
 if kind!='zero':
  assert abs(sum(map(mp.mpf,v['triangles']))-mass)<mp.mpf('1e-110')
  for i in range(3):
   for j in range(3):assert abs(mp.mpf(v['boundary'][i][j])+(mass if i==j else 0)-A[i,j])<mp.mpf('1e-110')
 checks.append({'name':row['name'],'status':c['status']})
assert counts==Counter(div=1095,mul=6081,sub=963,add=3017,neg=25)
assert statuses==Counter(pointed_full=16,measure_zero=2,unsupported_lineality=3,invalid=3)
out={'pass':True,'cases':24,'statuses':dict(statuses),'named_saved_counts':dict(counts),'checks':checks,'maximum_analytic_error':mp.nstr(maxerr,80),'author_cpu':D['cpu'],'review_cpu':time.process_time()-start,'files':{str(ROOT/f):sha(ROOT/f) for f in ('results.json','progress.jsonl','protocol.json','primitive.py','controls.py')},'scope':'Independent exact saved geometry and alternative analytic matrix contraction; named ledger sums only, not full operation trace reconstruction. No author candidate or quadrature execution. Finite MP checks, no outward certificate or network covariance claim.'}
(OUT/'pointed_second_moments_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
