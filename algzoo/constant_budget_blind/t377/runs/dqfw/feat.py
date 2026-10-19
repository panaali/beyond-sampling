import numpy as np
def feats(x):
    N=x.shape[0]
    o=np.argsort(-x,1); s=-np.sort(-x,1)
    F={}
    F['am']=o[:,0]; F['a2']=o[:,1]; F['m1']=s[:,0]; F['m2']=s[:,1]; F['m3']=s[:,2]
    rm=np.maximum.accumulate(x,1); F['rm']=rm
    return F
