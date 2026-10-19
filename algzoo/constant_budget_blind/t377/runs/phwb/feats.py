import numpy as np
def feats(x):
    N=x.shape[0]; F={}
    s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
    F['m1']=s[:,0];F['m2']=s[:,1];F['m3']=s[:,2]
    F['a1']=o[:,0];F['a2']=o[:,1]
    for k in range(10): F[f'x{k}']=x[:,k]
    return F
