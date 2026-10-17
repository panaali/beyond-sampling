"""Independent saved exact pair geometry, spherical-excess matrices and aggregates."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from collections import Counter
import ast,json,hashlib,resource,time
import mpmath as mp
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();mp.mp.dps=180
O=Path(__file__).resolve().parent;P=Path('/workspace/work/recover-small_h2_34/m16/approaches/third_step_pairs');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
# Only independent reviewer algebra helpers, no author modules or previous review execution.
tree=ast.parse((O/'check_pointed_second_moments_review.py').read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('unpack','dot','cross','det','rank','canon','q')];exec(compile(tree,'reviewer exact helpers','exec'))
g=read(P/'network_protocol.json');d=read(P/'network_results.json');assert d['complete'] and d['all_pass'] and d['protocol_sha256']==sha(P/'network_protocol.json')
for p,h in g['pins'].items():assert sha(p)==h,p
source=read(g['sector_source']);assert len(d['rows'])==len(source['rows'])==6
progress=[json.loads(x) for x in (P/'network_progress.jsonl').read_text().splitlines()];assert len(progress)==709
expected={'positive_z':mp.mpf(1)/4,'negative_z':mp.mpf(1)/4,'opposite':0,'constant_x':mp.mpf(1)/2,'x_z':1/(2*mp.pi),'negative_x':0,'bounded':1/mp.pi,'zero':0}
assert len(d['reference_controls'])==8
refcalls=0
for r in d['reference_controls']:
 assert r['passed'] and abs(mp.mpf(r['reference']['value'])-expected[r['name']])<mp.mpf('1e-50');refcalls+=r['reference']['evaluations']
counts=Counter();statuses=Counter();cells=0;angular=0;maxmatrix=mp.mpf(0);maxref=mp.mpf(0);candidate_cpu=reference_cpu=0.;outrows=[]
for row,prior in zip(d['rows'],source['rows'],strict=True):
 assert row['name']==prior['name'] and row['complete'] and row['all_pass'];cert=unpack(prior['certificate']);H=cert['H'];assert H==4
 expectedkeys={(k,i,j) for k in range(len(cert['sectors'])) for i in range(H) for j in range(i,H)};seen=set();raw=mp.matrix(H,H);refraw=mp.matrix(H,H)
 for r in row['cells']:
  key=(r['sector'],r['i'],r['j']);assert key in expectedkeys and key not in seen and r['passed'];seen.add(key);k,i,j=key;sector=cert['sectors'][k];v,w=sector['h3_rows'][i],sector['h3_rows'][j];c=unpack(r['certificate']);statuses[c['status']]+=1;counts.update(c['counts']);cells+=1
  angular+=r['reference']['evaluations'];candidate_cpu+=r['candidate_cpu'];reference_cpu+=r['reference_cpu']
  value=mp.mpf(r['value120']['value']);ref=mp.mpf(r['reference']['value']);maxref=max(maxref,abs(value-ref));assert abs(value-ref)<mp.mpf('1e-50')+mp.mpf('1e-40')*abs(ref)
  assert abs(value-mp.mpf(r['value80']['value']))<mp.mpf('1e-60')*(1+abs(value))
  if not any(v) or not any(w):assert c['status']=='zero_product' and value==0
  else:
   assert c['source']==sector['normals']+[v,w];normals=[]
   for n in c['source']:
    if any(n) and canon(n) not in normals:normals.append(canon(n))
   assert normals==[tuple(n) for n in c['normals']] and rank(normals)==c['normal_rank']==3
   rays=set()
   for a,b in combinations(normals,2):
    z=cross(a,b)
    if any(z):
     for sign in (-1,1):
      z1=[sign*x for x in z]
      if all(dot(n,z1)>=0 for n in normals):rays.add(canon(z1))
   assert sorted(rays)==[tuple(x) for x in c['rays']] and rank(list(rays))==c['ray_rank']
   if c['status']=='measure_zero':assert c['ray_rank']<3 and value==0
   else:
    assert c['status']=='pointed_full';center=[sum(x[l] for x in rays) for l in range(3)];assert center==c['center'];mass=mp.mpf(0);B=mp.matrix(3,3)
    facets={a for a,n in enumerate(normals) if sum(dot(n,x)==0 for x in rays)==2};assert facets=={f['normal_index'] for f in c['facets']}
    for f in c['facets']:
     n=normals[f['normal_index']];a,b=[c['rays'][l] for l in f['ray_indices']];assert a==f['r'] and b==f['s'] and dot(n,a)==dot(n,b)==0 and dot(center,cross(a,b))>0
     tri=[center,a,b];angles=[]
     for l in range(3):
      x,y,z=tri[l],tri[(l+1)%3],tri[(l+2)%3];u=[yy-dot(x,y)/dot(x,x)*xx for xx,yy in zip(x,y)];vv=[zz-dot(x,z)/dot(x,x)*xx for xx,zz in zip(x,z)]
      co=q(dot(u,vv))/mp.sqrt(q(dot(u,u)*dot(vv,vv)));assert -1<=co<=1;angles.append(mp.acos(co))
     mass+=(sum(angles)-mp.pi)/(4*mp.pi)
     tangent=[]
     for name,x,y in [('tangent_r',a,b),('tangent_s',b,a)]:
      t=[yy-dot(x,y)/dot(x,x)*xx for xx,yy in zip(x,y)];assert t==f[name] and dot(t,t)==f[name+'_norm2'];tangent.append([q(z)/mp.sqrt(q(dot(t,t))) for z in t])
     un=[q(z)/mp.sqrt(q(dot(n,n))) for z in n]
     for l in range(3):
      for m in range(3):B[l,m]+=un[l]*(tangent[0][m]+tangent[1][m])/(4*mp.pi)
    M=B+mp.eye(3)*mass;saved=r['value120']['matrix_result'];assert abs(mass-mp.mpf(saved['mass']))<mp.mpf('1e-110')
    for l in range(3):
     for m in range(3):
      err=abs(M[l,m]-mp.mpf(saved['matrix'][l][m]));maxmatrix=max(maxmatrix,err);assert err<mp.mpf('1e-110')
    contracted=sum(q(v[l])*M[l,m]*q(w[m]) for l in range(3) for m in range(3));assert abs(contracted-value)<mp.mpf('1e-110')
  raw[i,j]+=value;refraw[i,j]+=ref
 assert seen==expectedkeys
 for i in range(H):
  for j in range(i,H):raw[j,i]=raw[i,j];refraw[j,i]=refraw[i,j]
 mu=mp.matrix([mp.mpf(x) for x in prior['mean']]);cov=raw-mu*mu.T
 for i in range(H):
  for j in range(H):
   assert abs(raw[i,j]-mp.mpf(row['raw_second'][i][j]))<mp.mpf('1e-110') and abs(refraw[i,j]-mp.mpf(row['reference_raw_second'][i][j]))<mp.mpf('1e-90')
   assert abs(cov[i,j]-mp.mpf(row['covariance'][i][j]))<mp.mpf('1e-110')
 for m in row['principal_minors']:
  ix=m['indices'];val=mp.det(mp.matrix([[cov[i,j] for j in ix] for i in ix]));assert abs(val-mp.mpf(m['value']))<mp.mpf('1e-90') and val>=-mp.mpf('1e-50')
 for a in row['analytic_controls']:
  i,j=a['i'],a['j'];name=row['name']
  e=0 if name=='zero' else max(q(cert['wi'][i]*cert['wi'][j]),0)/2 if name=='zero_recurrent' else q(cert['wi'][i]**2)*(mp.mpf(3)/2+(1+mp.sqrt(2))/(2*mp.pi))
  assert a['passed'] and abs(raw[i,j]-e)<mp.mpf('1e-60')*(1+abs(e))
 outrows.append(dict(name=row['name'],cells=len(row['cells']),analytic_checks=len(row['analytic_controls'])))
assert cells==340 and statuses==Counter(zero_product=20,pointed_full=266,measure_zero=54)
assert counts==Counter(div=19264,mul=108924,sub=19038,add=52462,neg=433) and angular==122736 and refcalls==2200
savedcells=[x['cell'] for x in progress if x['stage']=='pair_result'];assert savedcells==[c for r in d['rows'] for c in r['cells']]
out={'pass':True,'networks':6,'cells':340,'statuses':dict(statuses),'named_saved_counts':dict(counts),'angular_reference_evaluations':angular,'control_reference_evaluations':refcalls,'max_alternate_spherical_matrix_discrepancy':mp.nstr(maxmatrix,80),'max_saved_reference_error':mp.nstr(maxref,80),'rows':outrows,'candidate_cpu':candidate_cpu,'angular_reference_cpu':reference_cpu,'author_cpu':d['cpu'],'review_cpu':time.process_time()-start,'result_sha256':sha(P/'network_results.json'),'progress_sha256':sha(P/'network_progress.jsonl'),'scope':'Independent exact saved pair geometry, alternate spherical-excess mass and boundary contraction, means/raw/covariance aggregates and analytic controls. No candidate/quadrature/model replay. MP and PSD tolerances are finite diagnostics, no trained/whole-network/outward covariance claim.'}
(O/'network_pair_controls_review_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
