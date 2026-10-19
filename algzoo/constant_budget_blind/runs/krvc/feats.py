import numpy as np
def feats(x):
    N=len(x); F={}
    s=-np.sort(-x,1); o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
    rank=np.argsort(o,1)
    for j in range(10): F['x%d'%j]=x[:,j]
    F['M']=s[:,0];F['S']=s[:,1];F['T']=s[:,2];F['min']=s[:,9]
    for k in range(2,10): F['P%d'%k]=x[:,:k].max(1)   # max of x0..x_{k-1}
    for k in range(1,9): F['Q%d'%k]=x[:,k:].max(1)    # max of x_k..x9
    for k in range(2,10):
        ss=-np.sort(-x[:,:k],1); F['P2_%d'%k]=ss[:,1]  # second of prefix x0..x_{k-1}
        if k>=3: F['P3_%d'%k]=ss[:,2]
    for j in range(10):
        F['am%d'%j]=(am==j)*1.; F['a2%d'%j]=(a2==j)*1.; F['top2_%d'%j]=(rank[:,j]<=1)*1.
        oth=np.delete(x,j,1); so=-np.sort(-oth,1)
        F['xmo%d'%j]=x[:,j]-so[:,0]; F['xso%d'%j]=x[:,j]-so[:,1]
        F['rxso%d'%j]=np.maximum(x[:,j]-so[:,1],0); F['rxmo%d'%j]=np.maximum(x[:,j]-so[:,0],0)
        if j<9: F['later%d'%j]=x[:,j]-x[:,j+1:].max(1)
        if j>0: F['earl%d'%j]=x[:,j]-x[:,:j].max(1); F['rearl%d'%j]=np.maximum(F['earl%d'%j],0)
    F['am']=am*1.;F['a2']=a2*1.
    return F
