import numpy as np
from util import *; from dict2 import km
x=np.load('x200.npy')[:100000]; H=np.load('H200.npy')[:100000]
xt=np.load('x200.npy')[100000:]; Ht=np.load('H200.npy')[100000:]
def mk(x,kind):
    if kind=='s':
        o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    else:
        o=np.argsort(-x[:,:9],1); s=np.take_along_axis(x[:,:9],o,1)
    N=len(x); oh1=np.zeros((N,10)); oh1[np.arange(N),o[:,0]]=s[:,0]
    oh2=np.zeros((N,10)); oh2[np.arange(N),o[:,1]]=s[:,1]
    return oh1,oh2,s
for i in range(16):
    res=[]
    for kind in ['s','a']:
        oh1,oh2,s=mk(x,kind); t1,t2,st=mk(xt,kind)
        F=np.hstack([oh1,oh2,s[:,2:3],x[:,9:10]]); w=relufit(F,H[:,i],15)
        A=km(w[:10],2)[1]; B=km(w[10:20],2)[1]
        for third in ['s3','x9']:
            def fam(o1,o2,ss,xx): return np.stack([o1[:,A].sum(1),o1.sum(1),o2[:,B].sum(1),o2.sum(1),ss[:,2] if third=='s3' else xx[:,9]],1)
            w2=relufit(fam(oh1,oh2,s,x),H[:,i],30)
            r=r2(np.maximum(fam(t1,t2,st,xt)@w2,0),Ht[:,i])
            res.append((kind,third,round(r,3),A,B,np.round(w2,1).tolist()))
    best=max(res,key=lambda t:t[2]); print(i,[(a,b,c) for a,b,c,*_ in res],'BEST',best,flush=True)
