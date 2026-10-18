"""Original T156 fixtures, exact saved outputs, and new native call-shape evidence."""
from pathlib import Path
import json,time,resource
import numpy as np
from observer import HERE,reset,report
PRIOR=HERE.parent/'rank_cost_review'
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();rows=[]
baseline=json.loads((PRIOR/'baseline.json').read_text())['rows']
for old in baseline:
 m=old['m'];k=min(m,4);D=np.zeros((m,4));D[np.arange(k),np.arange(k)]=np.array([1.,.5,.25,.125])[:k]
 name=old['name']
 if name=='zero':A=np.zeros((m,4))
 elif name=='rank_one':A=np.ones((m,4))
 elif name=='diagonal':A=D
 elif name=='large_diagonal':A=D*2.**500
 elif name=='small_diagonal':A=D*2.**-500
 elif name=='near_threshold':A=D.copy();A[np.arange(k),np.arange(k)]=np.array([1.,2.**-40,2.**-60,0.])[:k]
 else:raise ValueError(name)
 reset();s=np.linalg.svd(A,compute_uv=False);d=int(np.linalg.matrix_rank(A));f=int(np.linalg.matrix_rank(A,tol=1e-12));obs=report()
 row=dict(m=m,n=4,name=name,singular_values_hex=[float(x).hex() for x in s],rank_default=d,rank_tol1e12=f)
 assert row==old,(row,old);assert obs['gesdd'][0]==obs['gesdd'][1]==obs['gesdd'][8]==6
 assert all(x['failures']==0 for x in obs['kernels']);rows.append(dict(output=row,observer=obs))
dense=[]
for old in json.loads((PRIOR/'dense_baseline.json').read_text())['rows']:
 m=old['m'];k=min(m,4);A=np.zeros((m,4));A[:k,:k]=np.diag(np.full(k,2.))+np.diag(np.ones(k-1),1)+np.diag(np.ones(k-1),-1)
 reset();s=np.linalg.svd(A,compute_uv=False);d=int(np.linalg.matrix_rank(A));f=int(np.linalg.matrix_rank(A,tol=1e-12));obs=report()
 assert [float(x).hex() for x in s]==old['singular_values_hex'];assert d==old['rank_default'] and f==old['rank_tol']
 q5=sum(x['calls'] for x in obs['kernels'] if x['kernel']=='dlasq5');q6=sum(x['calls'] for x in obs['kernels'] if x['kernel']=='dlasq6')
 assert q5+q6==obs['nested'][8]>0;assert all(x['failures']==0 for x in obs['kernels']);dense.append(dict(output=old,observer=obs))
out=dict(rows=rows,dense=dense,cpu_seconds=time.process_time()-start)
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),dense=len(dense),cpu=out['cpu_seconds'])))
