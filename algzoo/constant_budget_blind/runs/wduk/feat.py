import numpy as np
def feats(x):
    N=x.shape[0]
    F={}
    for j in range(10): F[f'x{j}']=x[:,j]
    s=np.sort(x,1)[:,::-1]
    F['max']=s[:,0];F['sec']=s[:,1];F['third']=s[:,2]
    o=np.argsort(-x,1)
    F['amax']=o[:,0].astype(float);F['asec']=o[:,1].astype(float)
    F['x9-max8']=x[:,9]-x[:,:9].max(1)
    return F,s,o
