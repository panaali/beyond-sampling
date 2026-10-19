import numpy as np, probe
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
F=pf(x);Ft=pf(xt)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
for i in range(16):
    G,H,u=unconstrained(F,h[:,i])
    line=[]
    for v in ['V1','V2','V3','V4','V5']:
        th,cg,ch=fit(v,F,h[:,i],G,H)
        line.append('%s %.3f'%(v,r2(model(v,Ft,th,cg,ch),ht[:,i])))
    print(i,' '.join(line))
