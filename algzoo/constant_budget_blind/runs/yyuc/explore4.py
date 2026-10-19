import numpy as np
x=np.load('x1.npy'); H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; sm=o[:,1]
s=-np.sort(-x,1); M=np.maximum.accumulate(x,1)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for i in [0,3,6]:
  y=H[:,i]
  for k in range(10):
    m=(am==k)&(y>1e-6)
    if m.sum()<100: print(i,k,'n',m.sum()); continue
    F=np.stack([s[m,0],s[m,1],x[m,8],x[m,9],x[m,7]],1)
    c,*_=np.linalg.lstsq(F,y[m],rcond=None)
    r2=1-((F@c-y[m])**2).mean()/y[m].var()
    print(i,k,'n',m.sum(),'R2 %.3f'%r2,c)
