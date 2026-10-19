import numpy as np, probe
from lib import library; from feat import r2; from fit import fitrelu
rng=np.random.default_rng(8); N=30000
x=rng.standard_normal((N,10)); h=probe.hidden(x); L=library(x); keys=list(L); one=np.ones(N)
for i in range(8):
    y=h[:,i]; sel=[]
    for k in range(4):
        best=None
        for key in keys:
            if key in sel: continue
            A=np.c_[[L[q] for q in sel+[key]]].T; A=np.c_[A,one]
            c=np.linalg.lstsq(A,y,rcond=None)[0]; r=r2(y,A@c)
            if best is None or r>best[0]: best=(r,key)
        sel.append(best[1])
    A=np.c_[np.c_[[L[q] for q in sel]].T,one]; c,rr=fitrelu(A,y)
    print(i,sel,'lin',round(best[0],3),'relu',round(rr,3),np.round(c,2))
