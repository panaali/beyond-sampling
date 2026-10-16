import sys, time; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0, forward
M=(Wi0,Wh0,Wo0)
n=360*20
f23=faces23(n); f12=faces12(n)
m23=f23['Z']*np.mean(2*np.where(f23['hi']>f23['lo'],W(f23['lo'],f23['hi']),0))
m12=f12['Z']*np.mean(2*np.where(f12['hi']>f12['lo'],W(f12['lo'],f12['hi']),0))
print('mass23',m23,'mass12',m12,'blind',P_BLIND,'sum',m23+m12+P_BLIND)
x=np.random.default_rng(1).standard_normal((10**6,10)); v=-np.sort(-x,1)
nb=v[:,1]>0; r23=(v[:,1]-v[:,2]<v[:,0]-v[:,1])&nb
print('MC mass23',r23.mean(),'mass12',(nb&~r23).mean())
xb=blind_points(11*200); ok=forward(M,xb).argmax(1)==np.argsort(xb,1)[:,-2]
print('blind err cond',1-ok.mean())
for K in [1,2,3,50]:
    t=time.time()
    F,R,C=F_faces(M,f23,K); G,R2,C2=F_faces(M,f12,K)
    e23=f23['Z']*F.mean(); e12=f12['Z']*G.mean()
    print(K,'e23',e23,'unwalked',f23['Z']*R.mean(),'cells',C.mean(),'| e12',e12,'unwalked',f12['Z']*R2.mean(),'cells',C2.mean(),'acc',1-e23-e12-P_BLIND*(1-ok.mean()),time.time()-t)
