import numpy as np
from lib import relufit
x=np.load('x0.npy');h=np.load('h0.npy')
o=np.argsort(-x,1); s=-np.sort(-x,1)
pair=o[:,0]*10+o[:,1]; E=np.eye(100)[pair]
for i in [3,13,15,12]:
    A=np.c_[E*s[:,0:1],E*s[:,1:2],E,s[:,2],s[:,3]]
    tr=slice(0,80000); te=slice(80000,100000)
    c,r=relufit(A[tr],h[tr,i])
    rt=1-((np.maximum(A[te]@c,0)-h[te,i])**2).mean()/h[te,i].var()
    print(i,'pair-table R2 test',round(rt,3))
    # print table of mean h by pair
    T=np.array([[h[pair==10*a+b,i].mean() if a!=b else np.nan for b in range(10)] for a in range(10)])
    print(np.round(T).astype(int) if False else np.round(T,0))
