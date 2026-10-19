import numpy as np
def feats(x):
    N=x.shape[0]
    s=np.sort(x,1)[:,::-1]
    am=np.argmax(x,1)
    o=np.argsort(-x,1); a2=o[:,1]
    F={}
    for t in range(10): F['x%d'%t]=x[:,t]
    F['M']=s[:,0];F['S']=s[:,1];F['T']=s[:,2]
    for t in range(10): F['am%d'%t]=(am==t)*1.
    for t in range(10): F['a2%d'%t]=(a2==t)*1.
    pm=np.maximum.accumulate(x,1)
    F['M9']=pm[:,8]
    F['relu(x9-M9)']=np.maximum(x[:,9]-pm[:,8],0)
    return F
