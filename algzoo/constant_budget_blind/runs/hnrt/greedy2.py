import numpy as np, probe, sys
from util import *; from dict2 import *
rng=np.random.default_rng(9)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
D,o,s=base(x)
def quick(F,h,it=8):
    w=np.linalg.lstsq(F,h,rcond=None)[0]
    for _ in range(it):
        m=(h>0)|(F@w>0); w=np.linalg.lstsq(F[m],h[m],rcond=None)[0]
    return w, r2(np.maximum(F@w,0),h)
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; DD=dict(D); DD.update(groups_for(h,x,o,s))
    names=list(DD); A=np.stack([DD[n] for n in names],1)
    sel=[]; out=[]
    for step in range(5):
        best=None
        for k in range(len(names)):
            if k in sel: continue
            w,r=quick(A[:,sel+[k]],h,4)
            if best is None or r>best[1]: best=(k,r)
        sel.append(best[0]); w,r=quick(A[:,sel],h,20)
        out.append((names[best[0]],round(r,3)))
    print(i,out,np.round(w,2),flush=True)
