import numpy as np
def feats(x):
    N=x.shape[0]; F={}
    s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
    F['max']=s[:,0];F['sec']=s[:,1];F['third']=s[:,2]
    F['amax']=o[:,0].astype(float);F['asec']=o[:,1].astype(float)
    F['gap12']=s[:,0]-s[:,1]; F['gap23']=s[:,1]-s[:,2]
    for t in range(10): F[f'x{t}']=x[:,t]
    F['x9-max']=x[:,9]-s[:,0]
    F['x9-sec']=x[:,9]-s[:,1]
    rm=np.maximum.accumulate(x,1)
    F['rm8']=rm[:,8]
    F['x9-rm8']=x[:,9]-rm[:,8]
    return F
def r2(y,p):
    return 1-np.var(y-p)/max(np.var(y),1e-12)
