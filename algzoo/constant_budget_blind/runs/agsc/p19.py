import numpy as np, probe
from feat import r2; from fit import fitrelu
from lib2 import lib3
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(18); N=20000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
xt=rng.standard_normal((N,10)); ht=probe.hidden(xt)
L=lib3(x); Lt=lib3(xt); keys=list(L); one=np.ones(N)
Xall=np.array([L[k] for k in keys]).T
def score(sel,y):
    A=np.c_[Xall[:,sel],one]; c=np.linalg.lstsq(A,y,rcond=None)[0]; return r2(y,A@c)
for i in range(8):
    y=h[:,i]; beam=[((),0)]
    for k in range(4):
        cand={}
        for sel,_ in beam:
            for j in range(len(keys)):
                if j in sel: continue
                s=tuple(sorted(sel+(j,)))
                if s not in cand: cand[s]=score(list(s),y)
        beam=sorted(cand.items(),key=lambda z:-z[1])[:6]
    outs=[]
    for sel,_ in beam[:3]:
        A=np.c_[Xall[:,list(sel)],one]; c,rr=fitrelu(A,y)
        At=np.c_[np.array([Lt[keys[j]] for j in sel]).T,one]
        outs.append((round(r2(ht[:,i],np.maximum(At@c,0)),3),[keys[j] for j in sel],np.round(c,2)))
    for o in sorted(outs,key=lambda z:-z[0])[:2]: print(i,o)
