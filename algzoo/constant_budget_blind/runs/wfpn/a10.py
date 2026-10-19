import numpy as np
np.set_printoptions(suppress=True,linewidth=200,precision=2)
x=np.load('x.npy'); h=np.load('h.npy'); N=len(x)
def r2(B,y):
  B=np.c_[B]; c,*_=np.linalg.lstsq(B,y,rcond=None); return round(1-((B@c-y)**2).mean()/y.var(),3), c
M=np.maximum.accumulate(x,1)
def gaps(kind):
  G=np.zeros((N,10))
  for t in range(1,10):
    p=x[:,:t]; s=-np.sort(-p,1)
    if kind=='max': L=s[:,0]
    if kind=='top2': L=s[:,:2].mean(1)
    if kind=='top3': L=s[:,:min(3,t)].mean(1)
    if kind=='sec': L=s[:,min(1,t-1)]
    if kind=='relupos': L=np.maximum(p,0).sum(1)*0.4
    G[:,t]=np.maximum(L-x[:,t],0)
  return G
for kind in ['max','top2','top3','sec','relupos']:
  G=gaps(kind)
  B=np.c_[G[:,1:],x]
  print(kind,[r2(B,h[:,i])[0] for i in range(8)])
