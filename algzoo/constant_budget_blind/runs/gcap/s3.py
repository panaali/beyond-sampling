import numpy as np
x=np.load('x.npy');H=np.load('H.npy')
M=np.maximum.accumulate(x,1)  # M[:,t]=max x0..xt
# running second max
S=np.zeros_like(x); 
s=np.full(len(x),-np.inf); m=x[:,0].copy(); S[:,0]=-10
for t in range(1,10):
    s=np.maximum(s,np.minimum(m,x[:,t])); m=np.maximum(m,x[:,t]); S[:,t]=s
def fit(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); r=y-F@c; return 1-r.var()/y.var(),c
for i in range(8):
    y=H[:,i]
    r1,c1=fit(np.c_[x,M],y); r2,c2=fit(np.c_[x,M,S[:,1:]],y)
    print(i,'x+M',round(r1,3),'x+M+S',round(r2,3))
    print('  x',c1[:10].round(2),'\n  M',c1[10:20].round(2))
