"""Frozen analytic tridiagonals force a nontrivial values-only SVD path."""
import ctypes,hashlib,json,sys,time,resource
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
mode=sys.argv[1];assert mode in ('baseline','instrumented')
resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
env=json.loads((HERE/'environment.json').read_text())
for name,sha in env['binary_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
spy=ctypes.CDLL(str(HERE/'counted_spy.so')) if mode=='instrumented' else None
rows=[]
for m in [3,4,24]:
 k=min(m,4);A=np.zeros((m,4));A[:k,:k]=np.diag(np.full(k,2.))+np.diag(np.ones(k-1),1)+np.diag(np.ones(k-1),-1)
 if spy:spy.audit_reset();spy.nested_reset()
 s=np.linalg.svd(A,compute_uv=False);r0=int(np.linalg.matrix_rank(A));r1=int(np.linalg.matrix_rank(A,tol=1e-12))
 expected=2+2*np.cos(np.arange(1,k+1)*np.pi/(k+1));assert np.max(np.abs(s-expected))<1e-14
 assert r0==r1==k
 row=dict(m=m,singular_values_hex=[float(x).hex() for x in s],rank_default=r0,rank_tol=r1,max_analytic_error=float(np.max(np.abs(s-expected))))
 if spy:
  data=(ctypes.c_int64*12)();spy.nested_stats(data);row['nested']=list(map(int,data));assert data[11]==0
 rows.append(row)
if spy:
 old=json.loads((HERE/'dense_baseline.json').read_text())['rows'];assert [{k:v for k,v in x.items() if k!='nested'} for x in rows]==old
 assert sum(x['nested'][8] for x in rows)>0
out=dict(mode=mode,rows=rows,cpu_seconds=time.process_time()-start)
(HERE/('dense_'+mode+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
