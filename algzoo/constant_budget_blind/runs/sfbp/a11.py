import numpy as np
from relufit import relufit
d=np.load('d1.npz'); x,h=d['x'][:20000],d['h'][:20000]
p=x[:,:9]; Q=-np.sort(-p,1); r=lambda a:np.maximum(a,0)
G={'sumr':r(p).sum(1),'sumabs':np.abs(p).sum(1),'sum':p.sum(1),'norm':np.sqrt((p**2).sum(1)),'Q1':Q[:,0],'Q2':Q[:,1],'Q3':Q[:,2],'Q4':Q[:,3],'Q9':Q[:,8],'Q5':Q[:,4],'x9':x[:,9],
   'sumrneg':r(-p).sum(1)}
names=list(G); A=np.stack([G[n] for n in names],1)
for i in [8,4,20,24,30,3,9,10,14,28,25,31,5]:
    co,R=relufit(A,h[:,i])
    # also with positional x's
    co2,R2=relufit(np.c_[A,p],h[:,i])
    print(i,'R2 global=%.3f +x=%.3f'%(R,R2),dict(zip(names,np.round(co,2))))
