import numpy as np
x=np.load('x1.npy');H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
o9=np.argsort(-x[:,:9],1); b1=o9[:,0]; b2=o9[:,1]
def tabr2(key,y):
    k=key; p=np.zeros_like(y)
    for v in np.unique(k): m=k==v; p[m]=y[m].mean()
    return 1-((p-y)**2).mean()/y.var()
for i in range(32):
    y=H[:,i]
    print(i,'all10: am %.2f a2 %.2f pair %.2f | x0..8: am %.2f a2 %.2f pair %.2f'%(tabr2(am,y),tabr2(a2,y),tabr2(am*10+a2,y),tabr2(b1,y),tabr2(b2,y),tabr2(b1*10+b2,y)))
