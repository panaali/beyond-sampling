import numpy as np, probe, sys
from gate import fastfit, lmfit
np.set_printoptions(precision=2,suppress=True,linewidth=200)
rng=np.random.default_rng(13)
X0=rng.standard_normal((30000,10)); H=probe.hidden(X0)
for a in sys.argv[1:]:
    i,L=map(int,a.split(':')); x=X0[:,:L]
    s=np.argsort(-x,1); srt=-np.sort(-x,1); E=np.eye(L)
    X=np.c_[srt[:,0:1]*E[s[:,0]], srt[:,1:2]*E[s[:,1]], srt[:,2:3]*E[s[:,2]], x.sum(1)]
    h=H[:,i]; c,r=fastfit(h,X); c=lmfit(h,X,c); pr=np.maximum(X@c,0); r=1-((pr-h)**2).mean()/h.var()
    print(i,"L",L,"R2 top3+sum %.3f"%r, "sum coef %.2f"%c[-1])
    for j,nm in enumerate(['T1@am','T2@a2','T3@a3']): print("  ",nm,c[L*j:L*j+L])
