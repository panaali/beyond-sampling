import numpy as np
T=10
GB=np.array([0.005,0.01,0.02,0.035,0.05,0.075,0.1,0.15,0.2,0.3,0.5])
def feats(x):
    o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    v1,v2,v3,v4=s[:,0],s[:,1],s[:,2],s[:,3]
    g12=(v1-v2)/np.abs(v1); g23=(v2-v3)/np.maximum(np.abs(v2),1e-12)
    return dict(p1=o[:,0],p2=o[:,1],p3=o[:,2],p4=o[:,3],blind=v2<=0,g12=g12,g23=g23,
                b12=np.searchsorted(GB,g12),b23=np.searchsorted(GB,g23),v1=v1,v2=v2,v3=v3,v4=v4)
