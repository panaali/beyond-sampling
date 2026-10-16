import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=3600
fc=faces23(n); Z=fc['Z']
F,R,C=F_faces(M,fc,60,radial=True); F=Z*F
k=np.arange(n)%360
cm=np.array([F[k==j].mean() for j in range(360)])
print('total sd',F.std(),'within-combo sd',np.sqrt(np.mean([(F[k==j]).var() for j in range(360)])),'between sd',cm.std())
q=np.quantile(F,[.1,.25,.5,.75,.9,.99]); print('quantiles',q, 'max',F.max())
# contribution of top 10% of faces
s=np.sort(F)[::-1]; print('share top1%',s[:n//100].sum()/s.sum(),'top10%',s[:n//10].sum()/s.sum())
# QMC actual error: random shifts
ests=[]
for r in range(8):
    sh=np.random.default_rng(r).random(9)
    f=faces23(n,shift=sh); G,_,_=F_faces(M,f,60,radial=True); ests.append(Z*G.mean())
print('shifted QMC ests sd',np.std(ests),'iid-equivalent sd',F.std()/np.sqrt(n), ests)
