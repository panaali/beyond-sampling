import numpy as np, probe
from lib import relufit
from forms import design
x=np.load('x0.npy');h=np.load('h0.npy')
xt=np.random.default_rng(7).standard_normal((50000,10)); ht=probe.hidden(xt)
C={}
for i in range(16):
    A=np.c_[tuple(design(i,x))]; c,r=relufit(A,h[:,i],it=15)
    pt=np.maximum(np.c_[tuple(design(i,xt))]@c,0); rt=1-((pt-ht[:,i])**2).mean()/ht[:,i].var()
    C[i]=[round(float(v),3) for v in c]; print(i,C[i],'train',round(r,3),'fresh',round(rt,3))
print(C)
