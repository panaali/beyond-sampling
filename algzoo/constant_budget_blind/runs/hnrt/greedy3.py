import numpy as np, sys
from util import *; from dict2 import base
x=np.load('x200.npy')[:25000]; H=np.load('H200.npy')[:25000]
D,o,s=base(x)
names=list(D); cols=[D[n] for n in names]
for S in range(1,1024):
    idx=[j for j in range(10) if S>>j&1]
    if len(idx)<2: continue
    names.append('max{'+','.join(map(str,idx))+'}'); cols.append(x[:,idx].max(1))
A=np.stack(cols,1).astype(np.float64)
def quick(F,h,it):
    w=np.linalg.lstsq(F,h,rcond=None)[0]
    for _ in range(it):
        m=(h>0)|(F@w>0); w=np.linalg.lstsq(F[m],h[m],rcond=None)[0]
    return w, r2(np.maximum(F@w,0),h)
for i in [int(a) for a in sys.argv[1:]]:
    h=H[:,i]; sel=[]; out=[]
    for step in range(5):
        best=None
        for k in range(len(names)):
            if k in sel: continue
            w,r=quick(A[:,sel+[k]],h,3)
            if best is None or r>best[1]: best=(k,r)
        sel.append(best[0]); w,r=quick(A[:,sel],h,20)
        out.append((names[best[0]],round(r,3)))
    print(i,out,np.round(w,2).tolist(),flush=True)
