import itertools,json,time
from pathlib import Path
import numpy as np
from families import owner,run,Ops
from m43 import Wi0,Wh0,Wo0
start=time.process_time();fail=0;n=0
for T in range(3,7):
 for x in itertools.product((-1,0,1),repeat=T):
  a,b,s=owner(x); p=sorted(range(T),key=lambda i:(-x[i],i));n+=1
  fail+=int((a,b)!=tuple(p[:2]) or x[a]<x[b] or any(x[b]<x[k] for k in range(T) if k not in(a,b)))
rows=[]
for name,wo in [('zero_readout',np.zeros_like(Wo0)),('identical_readout',np.tile(Wo0[0],(3,1)))]:
 for kind in('old','quadratic','nearest'):
  r=run((Wi0,Wh0,wo),kind,'exact');r.update(name=name,kind=kind,reference=1/3,error=r['value']-1/3);rows.append(r)
out=dict(owner_points=n,owner_failures=fail,rows=rows,cpu=time.process_time()-start)
Path(__file__).with_name('tie_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
