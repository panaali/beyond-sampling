import numpy as np, probe, itertools
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
F=pf(x);Ft=pf(xt)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
cands={2:[[2],[2,4],[0,2,4],[0,1,2,3,4]],9:[[1],[0,1],[1,4,7]],13:[[1,3],[0,1,3],[0,1,2,3]],1:[[0],[0,7,4]],5:[[5,4],[0,2]],8:[[2],[2,5]],11:[[7]]}
for i,As in cands.items():
    G,H,u=unconstrained(F,h[:,i]);out=[]
    for A in As:
      for tr in ('lin','relu'):
        for excl in (True,False):
          for mode in ('g3','g2h2'):
            th,cg,ch=fitB(x,F,h[:,i],G,H,A,tr,excl,mode)
            out.append((r2(mB(xt,Ft,th,cg,ch,A,tr,excl,mode),ht[:,i]),A,tr,excl,mode,np.round(th,1),cg,ch))
    out.sort(key=lambda t:-t[0])
    for o in out[:2]: print(i,'%.3f'%o[0],o[1:])
