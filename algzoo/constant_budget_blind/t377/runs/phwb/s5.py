import numpy as np
x=np.load('x0.npy');h=np.load('h0.npy')
o=np.argsort(-x,1); s=-np.sort(-x,1)
p1,p2,p3=o[:,0],o[:,1],o[:,2]; m1,m2,m3=s[:,0],s[:,1],s[:,2]
E=np.eye(10)
def design(k):
    cols=[E[p1]*m1[:,None],E[p2]*m2[:,None],E[p1],E[p2]]
    if k>=1: cols.append(E[p3]*m3[:,None])
    if k>=2: cols.append(np.c_[x[:,9],x.mean(1)])
    return np.hstack(cols)
for i in range(16):
    g=h[:,i]>0.3
    r=[]
    for k in range(3):
        A=design(k)[g]; c,_,_,_=np.linalg.lstsq(A,h[g,i],rcond=None)
        pred=np.maximum(design(k)@c,0)
        r.append(1-((pred-h[:,i])**2).mean()/h[:,i].var())
    print(i,np.round(r,3))
