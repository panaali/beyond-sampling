import numpy as np
x=np.load('x1.npy');h=np.load('h1.npy')
o=np.argsort(-x,1);s=-np.sort(-x,1)
np.set_printoptions(precision=2,suppress=True,linewidth=150)
def r2grp(keys,X):
    pred=np.zeros_like(h)
    for k in np.unique(keys):
        m=keys==k
        c,*_=np.linalg.lstsq(X[m],h[m],rcond=None);pred[m]=X[m]@c
    return 1-((h-pred)**2).mean(0)/h.var(0)
print('M|amax    ',r2grp(o[:,0],s[:,:1]))
print('M,S|amax    ',r2grp(o[:,0],s[:,:2]))
print('M,S|amax,asec',r2grp(o[:,0]*10+o[:,1],s[:,:2]))
print('M,S,x9|amax,asec',r2grp(o[:,0]*10+o[:,1],np.c_[s[:,:2],x[:,9]]))
print('M,S,T,x9|amax,asec',r2grp(o[:,0]*10+o[:,1],np.c_[s[:,:3],x[:,9]]))
