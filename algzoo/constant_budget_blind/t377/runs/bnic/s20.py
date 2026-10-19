import numpy as np
import labels_fit as L
x=np.load('x1.npy')[:50000]; h=np.load('h1.npy')[:50000]
srt=-np.sort(-x,1); s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
m=srt[:,0]; s1=srt[:,1]; I2=np.eye(10)[a2]; I1=np.eye(10)[am]
M=np.maximum.accumulate(x,1); T=m[:,None]-x; G9=np.maximum(x[:,9]-M[:,8],0)
def fit(B,y):
    c,*_=np.linalg.lstsq(B,y,rcond=None); return 1-((y-B@c)**2).mean()/y.var(),c
for i in range(16):
    y=h[:,i]; B=np.stack(L._terms(i,x),1)
    base=np.c_[T,G9,m[:,None]*I1]
    r0,_=fit(base,y); r1,_=fit(np.c_[base,(m-s1)[:,None]*I2],y); r2,_=fit(np.c_[base,(m-s1)[:,None]*I2,(m-s1)[:,None]*I1, srt[:,2:]],y)
    # x_j relative to the previous running max: P_j
    P=np.maximum(np.c_[np.zeros(len(x)),M[:,:-1]]-x,0)[:,1:]
    r3,_=fit(np.c_[base,P],y)
    print(i,'T+G9+amI %.3f +gap2*a2 %.3f +more %.3f +P %.3f'%(r0,r1,r2,r3))
