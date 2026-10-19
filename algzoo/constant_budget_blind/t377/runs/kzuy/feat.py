import numpy as np
def feats(x):
    N=x.shape[0]; F={}
    s=np.sort(x,1)[:,::-1]
    am=np.argmax(x,1); o=np.argsort(-x,1); a2=o[:,1]
    F['max']=s[:,0];F['m2']=s[:,1];F['m3']=s[:,2];F['gap12']=s[:,0]-s[:,1];F['gap23']=s[:,1]-s[:,2]
    for t in range(10): F[f'x{t}']=x[:,t]
    for t in range(10): F[f'am=={t}']=(am==t)*1.
    for t in range(10): F[f'a2=={t}']=(a2==t)*1.
    F['am']=am*1.;F['a2']=a2*1.
    F['x9-max08']=x[:,9]-x[:,:9].max(1)
    F['x9-m2']=x[:,9]-s[:,1]
    return F
