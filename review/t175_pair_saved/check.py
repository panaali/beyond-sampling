"""Independent saved geometry and spherical-excess matrix contraction; no author imports."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from collections import Counter
import json,hashlib,time
import mpmath as mp
HERE=Path(__file__).resolve().parent
AUTHOR=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_pairs')

def unpack(x):
 if isinstance(x,list):
  if len(x)==2 and all(isinstance(v,str) and v.lstrip('-').isdigit() for v in x):return F(int(x[0]),int(x[1]))
  return [unpack(v) for v in x]
 if isinstance(x,dict):return {k:unpack(v) for k,v in x.items()}
 return x
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def det(a,b,c):return dot(a,cross(b,c))
def canon(a):return tuple(x/abs(next(v for v in a if v)) for x in a)
def rank(rows):
 if any(det(*v) for v in combinations(rows,3)):return 3
 if any(any(cross(*v)) for v in combinations(rows,2)):return 2
 return int(any(any(v) for v in rows))
def q(x):return mp.mpf(x.numerator)/x.denominator
def vec(v):return [q(x) for x in v]
def unit(v):
 a=vec(v);n=mp.sqrt(dot(a,a));return [x/n for x in a]
def close(a,b):
 d=abs(a-mp.mpf(b));assert d<=mp.mpf('1e-95')*(1+abs(a)),(str(d),str(a),b)
 return d

def geometry(source,c):
 assert c['source']==source
 ns=[];removed=[]
 for i,n in enumerate(source):
  if not any(n):removed.append({'index':i,'reason':'zero'});continue
  a=canon(n)
  if a in ns:removed.append({'index':i,'reason':'positive_duplicate'})
  else:ns.append(a)
 assert [tuple(n) for n in c['normals']]==ns and c['removed']==removed
 assert c['normal_rank']==rank(ns)
 if rank(ns)<3:assert c['status']=='unsupported_lineality';return
 rays=set()
 for a,b in combinations(ns,2):
  z=cross(a,b)
  if not any(z):continue
  for r in (z,tuple(-x for x in z)):
   if all(dot(n,r)>=0 for n in ns):rays.add(canon(r))
 rays=sorted(rays)
 assert [tuple(r) for r in c['rays']]==rays and c['ray_rank']==rank(rays)
 if rank(rays)<3:assert c['status']=='measure_zero';return
 assert c['status']=='pointed_full'
 center=tuple(sum(r[i] for r in rays) for i in range(3))
 assert tuple(c['center'])==center and all(dot(n,center)>0 for n in ns)
 expected={i for i,n in enumerate(ns) if sum(dot(n,r)==0 for r in rays)==2}
 assert len(c['facets'])==len(expected)==len(rays)
 assert {f['normal_index'] for f in c['facets']}==expected
 incidence=Counter()
 for f in c['facets']:
  k=f['normal_index'];n=ns[k];i,j=f['ray_indices'];r,s=rays[i],rays[j]
  assert set((i,j))=={a for a,v in enumerate(rays) if dot(n,v)==0}
  assert tuple(f['normal'])==n and tuple(f['r'])==r and tuple(f['s'])==s
  assert det(center,r,s)>0
  rr,ss,rs=dot(r,r),dot(s,s),dot(r,s)
  pr=tuple(y-rs*x/rr for x,y in zip(r,s));ps=tuple(x-rs*y/ss for x,y in zip(r,s))
  assert tuple(f['tangent_r'])==pr and tuple(f['tangent_s'])==ps
  assert f['normal_norm2']==dot(n,n) and f['tangent_r_norm2']==dot(pr,pr) and f['tangent_s_norm2']==dot(ps,ps)
  assert f['triangle']==dict(det=det(center,r,s),center_norm2=dot(center,center),r_norm2=rr,s_norm2=ss,cr=dot(center,r),rs=rs,sc=dot(s,center))
  incidence.update((i,j))
 assert all(incidence[i]==2 for i in range(len(rays)))

def alternate(c):
 if c['status']=='measure_zero':return mp.mpf(0),mp.zeros(3)
 assert c['status']=='pointed_full'
 mass=mp.mpf(0);boundary=mp.zeros(3)
 for f in c['facets']:
  u,r,s=map(unit,(c['center'],f['r'],f['s']))
  angles=[]
  for a,b,d in ((u,r,s),(r,s,u),(s,u,r)):
   ab=dot(a,b);ad=dot(a,d)
   x=[v-ab*w for v,w in zip(b,a)];y=[v-ad*w for v,w in zip(d,a)]
   cosine=dot(x,y)/mp.sqrt(dot(x,x)*dot(y,y))
   assert -1<cosine<1
   angles.append(mp.acos(cosine))
  # Independent spherical excess, rather than source solid-angle atan2.
  mass+=(sum(angles)-mp.pi)/(4*mp.pi)
  cosine=dot(r,s);assert -1<cosine<1
  scale=mp.sqrt((1-cosine)/(1+cosine))/(4*mp.pi)
  n=unit(f['normal'])
  for i in range(3):
   for j in range(3):boundary[i,j]+=n[i]*(r[j]+s[j])*scale
 return mass,boundary+mp.eye(3)*mass

def main():
 start=time.process_time();out=HERE/'results.json';assert not out.exists()
 pins=json.loads((HERE/'pins.json').read_text())
 for p,h in pins['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 P=json.loads((AUTHOR/'trained_pilot/protocol.json').read_text())
 g=unpack(json.loads(Path(P['geometry']).read_text())['certificate'])
 d=unpack(json.loads((AUTHOR/'trained_pilot/results.json').read_text()))
 assert d['complete'] and d['all_pass'] and [(r['i'],r['j']) for r in d['rows']]==[(0,0),(1,1),(15,15),(0,1),(0,15)]
 statuses=Counter();rows=[];maximum=mp.mpf(0)
 with mp.workdps(140):
  for row in d['rows']:
   assert [c['sector'] for c in row['cells']]==list(range(34))
   for cell in row['cells']:
    s=g['sectors'][cell['sector']];v=s['h3_rows'][row['i']];w=s['h3_rows'][row['j']];c=cell['certificate'];statuses[c['status']]+=1
    if not any(v) or not any(w):
     assert c['status']=='zero_product' and cell['value120']['value']=='0.0' and cell['value120']['matrix_result'] is None
     rows.append(dict(i=row['i'],j=row['j'],sector=cell['sector'],status=c['status']));continue
    geometry(s['normals']+[v,w],c)
    assert c['status']!='unsupported_lineality'
    mass,M=alternate(c);saved=cell['value120']['matrix_result'];err=close(mass,saved['mass'])
    for i in range(3):
     for j in range(3):err=max(err,close(M[i,j],saved['matrix'][i][j]))
    val=sum(q(v[i])*M[i,j]*q(w[j]) for i in range(3) for j in range(3));err=max(err,close(val,cell['value120']['value']));maximum=max(maximum,err)
    rows.append(dict(i=row['i'],j=row['j'],sector=cell['sector'],status=c['status'],difference=mp.nstr(err,110)))
  result=dict(complete=True,cells=len(rows),statuses=dict(statuses),maximum_difference=mp.nstr(maximum,110),rows=rows,cpu_seconds=time.process_time()-start,
   scope='Independent exact saved geometry and 140-digit spherical-excess/tangent-half-angle contraction only. No author imports, model evaluations or angular integration. T175 upstream sector certificate inherited; finite MP comparison is not outward certification. Aggregate moments/covariance/costs reviewed separately by T55.',pins_sha256=hashlib.sha256((HERE/'pins.json').read_bytes()).hexdigest())
 assert len(rows)==170
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
