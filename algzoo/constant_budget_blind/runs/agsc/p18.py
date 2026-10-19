import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib2 import lib2
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(17); N=30000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((N,10)); ht=probe.hidden(xt)
L=lib2(x); Lt=lib2(xt); keys=list(L); one=np.ones(N)
for i in range(8):
    y=h[:,i]; sel=[]; hist=[]
    for k in range(4):
        best=None
        for key in keys:
            if key in sel: continue
            A=np.c_[np.array([L[q] for q in sel+[key]]).T,one]
            c=np.linalg.lstsq(A,y,rcond=None)[0]; r=r2(y,A@c)
            if best is None or r>best[0]: best=(r,key)
        sel.append(best[1]); hist.append(round(best[0],3))
    A=np.c_[np.array([L[q] for q in sel]).T,one]; c,rr=fitrelu(A,y)
    At=np.c_[np.array([Lt[q] for q in sel]).T,one]
    print(i,sel,hist,'relu test',round(r2(ht[:,i],np.maximum(At@c,0)),3),np.round(c,2))
