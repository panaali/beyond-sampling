import json,time,hashlib,itertools
from pathlib import Path
import numpy as np
from batch import Grouped,Engine,EDGE,SOURCE
from m16.approaches.small_ladder_review.portable_cost.accounting import Audit
from m16.approaches.small_ladder_review.portable_cost.t6.jacobi import outer_rule
HERE=Path(__file__).parent
start=time.process_time();cases=[];eye=np.eye(3,6);anchor=.2*np.ones(6)/6+.8*np.eye(6)[0]
for width in(1.,.1,.01,.001,.0001,.00001,.0000001,0.):
 b=np.ones((3,6))+width*eye;b/=b.sum(axis=1)[:,None];cases.extend(b[list(p)] for p in itertools.permutations(range(3)))
for width in(.1,.01,.001,.0001,.00001,.0000001,0.):
 b=np.stack([anchor,anchor+width*(eye[1]-eye[0]),anchor+width*(eye[2]-eye[0])]);cases.extend(b[list(p)] for p in itertools.permutations(range(3)))
cases.extend([eye,np.stack([eye[0],eye[0],eye[1]]),np.tile(anchor,(3,1))])
fixture=HERE.parent/'directed_chebyshev/t6_fixture.json';rays=np.array(json.loads(fixture.read_text())['rays'])
# Frozen endpoints; all fallback-only comparison avoids changing analytic screens.
for n in(4,8):
 y,rest,w=outer_rule(n,Audit())
 for ray in rays:
  prefix=y@ray[:,:3].T;cases.extend(prefix[:,None,:]+rest[:,None,None]*ray[:,3:].T[None,:,:])
cases=np.asarray(cases);rows=[]
for chunk in(16384,65536):
 a=Audit();b=Audit();scalar=Engine(a);group=Grouped(b,max_nodes=chunk);ts=time.process_time()
 old=[scalar.positive(v,6,1e-11,128) for v in cases];scalarcpu=time.process_time()-ts;ts=time.process_time();value,meta=group.many(cases,6,1e-11,128);batchcpu=time.process_time()-ts
 oldvalue=np.array([z[0] for z in old]);oldmeta=[z[1] for z in old];maxdiff=float(np.max(abs(value-oldvalue)))
 different=[i for i,(x,y) in enumerate(zip(oldmeta,meta)) if x!=y]
 row=dict(max_nodes=chunk,count=len(cases),max_absolute_difference=maxdiff,max_relative_difference=float(np.max(abs(value/oldvalue-1))),metadata_differences=different,orders=sorted(set(m['quadrature_order'] for m in meta)),scalar_cpu=scalarcpu,group_cpu=batchcpu,selection_cpu=group.selection_cpu,node_cpu=group.node_cpu,groups=group.groups,scalar_cost=a.report(),group_cost=b.report());rows.append(row)
 assert maxdiff<1e-11 and not different
 # Representative exact node arrays, independent per-triangle matmul versus batched.
 node_errors=[]
 for n in row['orders']:
  ids=[i for i,m in enumerate(meta) if m['quadrature_order']==n][:8];(bary,_),_=group.rule(n)
  batched=cases[ids].transpose(0,2,1)@bary
  node_errors.append(float(max(np.max(abs(cases[i].T@bary-batched[j])) for j,i in enumerate(ids))))
 row['node_max_difference']=max(node_errors)
# An individual rule cannot silently exceed the declared node cap.
try:Grouped(Audit(),max_nodes=1).many(cases[:1],6,1e-11,128)
except ValueError:pass
else:raise AssertionError('missing node-cap rejection')
# Empty callback contract.
g=Grouped(Audit());v,m=g.many(np.empty((0,3,6)),6,1e-11,128);assert len(v)==len(m)==0
out=dict(rows=rows,cpu=time.process_time()-start,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),empty_pass=True,node_cap_rejection_pass=True)
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
