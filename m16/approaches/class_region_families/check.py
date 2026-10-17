import itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from families import quadratic,nearest,owner,engine,run,Ops
from m43 import Wi0,Wh0,Wo0,noisy_models
from exact import exact_acc
start=time.process_time();rows=[]
# All exact vertex matrix identities and constructive interior points T3..10.
for T in range(3,11):
  op=Ops();hs=quadratic(T,op);fail=0;det=[]
  for h in hs:
    N=h['cone'].N.astype(int);B=np.rint(np.linalg.inv(N)).astype(int)
    fail+=int(not np.array_equal(N@B,np.eye(T,dtype=int)))
    # Integer interior via positive edge gaps, varying each coordinate exactly.
    x=B@np.arange(1,T+1);a,b=h['pair']
    fail+=int(owner(x)!=(a,b,int(h['sigma'])))
    det.append(int(round(np.linalg.det(N))))
  rows.append(dict(T=T,families=len(hs),failures=fail,determinants=sorted(set(det))))
assert all(r['failures']==0 for r in rows)
# Exhaustive distinct integer permutations at3..6: one interior chart family.
parts=[]
for T in range(3,7):
 hs=quadratic(T,Ops());bad=0;n=0
 for p in itertools.permutations(range(1,T+1)):
  for sg in(-1,1):
   x=sg*np.array(p);matches=[h for h in hs if np.all(h['cone'].N@x>0)]
   bad+=int(len(matches)!=1 or matches[0]['truth']!=owner(x)[1]);n+=1
 parts.append(dict(T=T,points=n,failures=bad));assert bad==0
masses=[]
for kind,fn in [('quadratic',quadratic),('nearest',nearest)]:
 op=Ops();hs=fn(3,op);byclass=[sum(engine.mass(h['cone'],op) for h in hs if h['truth']==j) for j in range(3)]
 masses.append(dict(kind=kind,families=len(hs),byclass=byclass,total=sum(byclass)))
 assert max(abs(v-1/3) for v in byclass)<1e-14
models=[(Wi0,Wh0,Wo0)]+noisy_models(k=4,sigma=.05,seed=80)
trained=[]
for i,m in enumerate(models):
 refops=Ops();ts=time.process_time();ref=exact_acc(m,refops);rr=[]
 for mode in('exact','recipe'):
  for kind in('old','quadratic','nearest'):
   t=time.process_time();r=run(m,kind,mode);r.update(kind=kind,mode=mode,cpu=time.process_time()-t,error=r['value']-ref);rr.append(r)
 trained.append(dict(model=i,reference=ref,reference_ops_proxy=refops.n,rows=rr,cpu=time.process_time()-ts))
out=dict(matrix_controls=rows,partition_controls=parts,mass_controls=masses,trained=trained,cpu=time.process_time()-start,snapshot_sha256=hashlib.sha256(Path(__file__).with_name('estT_snapshot.py').read_bytes()).hexdigest())
Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
