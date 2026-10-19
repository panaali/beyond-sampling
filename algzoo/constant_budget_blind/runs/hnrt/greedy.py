import numpy as np, probe, sys
from util import *; from dict import build
rng=np.random.default_rng(4)
x=rng.standard_normal((30000,10)); H=probe.hidden(x)
D=build(x); names=list(D); A=np.stack([D[n] for n in names],1)
def quick(F,h,it=8):
    w=np.linalg.lstsq(F,h,rcond=None)[0]
    for _ in range(it):
        m=(h>0)|(F@w>0); w=np.linalg.lstsq(F[m],h[m],rcond=None)[0]
    return w, r2(np.maximum(F@w,0),h)
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; sel=[]; out=[]
    for step in range(5):
        best=None
        for k,n in enumerate(names):
            if k in sel: continue
            w,r=quick(A[:,sel+[k]],h,5)
            if best is None or r>best[1]: best=(k,r)
        sel.append(best[0]); w,r=quick(A[:,sel],h,20)
        out.append((names[best[0]],round(r,3)))
    print(i,out,np.round(w,2),flush=True)
