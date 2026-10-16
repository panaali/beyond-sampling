import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube')
import numpy as np
from tube import *
n=2**16
for fn,name in [(faces23,'23'),(faces12,'12')]:
    ms=[];mr=[]
    for r in range(4):
        f=fn(n,shift=np.random.default_rng(r).random(11))
        ms.append(f['Z']*np.mean(2*np.where(f['hi']>f['lo'],W(f['lo'],f['hi']),0)))
        rho=np.linalg.norm(f['x0'],axis=1); lo,hi=f['lo']/rho,f['hi']/rho
        mr.append(f['Z']*np.mean(2*np.where(hi>lo,Wrad(lo,hi),0)))
    print(name,np.mean(ms),np.std(ms),np.mean(mr),np.std(mr))
x=np.random.default_rng(1).standard_normal((4*10**6,10)); v=-np.sort(-x,1)
nb=v[:,1]>0; r23=(v[:,1]-v[:,2]<v[:,0]-v[:,1])&nb
print('MC',r23.mean(),(nb&~r23).mean(), 'sum check', 1-11/1024)
