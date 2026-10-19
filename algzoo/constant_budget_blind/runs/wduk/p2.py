import numpy as np
x=np.load('x1.npy');h=np.load('h1.npy')
o=np.argsort(-x,1);s=-np.sort(-x,1)
def r2grp(keys,X):
    out=np.zeros(16);pred=np.zeros_like(h)
    for k in np.unique(keys):
        m=keys==k
        if m.sum()<30: pred[m]=h[m].mean(0);continue
        c,*_=np.linalg.lstsq(X[m],h[m],rcond=None);pred[m]=X[m]@c
    return 1-((h-pred)**2).mean(0)/h.var(0)
print('lin x      ',np.round(r2grp(np.zeros(len(x)),x),2))
print('lin x|amax ',np.round(r2grp(o[:,0],x),2))
print('lin x|amax,asec',np.round(r2grp(o[:,0]*10+o[:,1],x),2))
X2=np.c_[x,s[:,:3]]
print('lin x,s|amax',np.round(r2grp(o[:,0],X2),2))
