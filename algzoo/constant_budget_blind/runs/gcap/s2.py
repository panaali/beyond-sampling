import numpy as np
x=np.load('x.npy');H=np.load('H.npy')
sx=-np.sort(-x,1)
def fit(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); r=y-F@c; return 1-r.var()/y.var(),c
for i in range(8):
    y=H[:,i]
    r1,c1=fit(x,y); r2,c2=fit(sx,y); r3,c3=fit(np.c_[x,sx],y)
    print(i,'lin x',round(r1,3),'sorted',round(r2,3),'both',round(r3,3))
    print('  xcoef',c1[:10].round(2)); print('  sortcoef',c2[:10].round(2))
