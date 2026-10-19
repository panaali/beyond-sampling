import numpy as np
def feats(x):
    N=x.shape[0]
    F={}
    s=np.sort(x,1)[:,::-1]
    am=np.argmax(x,1); o=np.argsort(-x,1); a2=o[:,1]
    F['max']=s[:,0];F['sec']=s[:,1];F['third']=s[:,2]
    F['am']=am;F['a2']=a2
    for k in range(10): F[f'x{k}']=x[:,k]
    rm=np.maximum.accumulate(x,1)
    F['last']=x[:,9]
    F['m-s']=s[:,0]-s[:,1]
    return F
