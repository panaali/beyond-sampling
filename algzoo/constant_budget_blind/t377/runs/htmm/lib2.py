import numpy as np
def library2(x):
    F={}
    p=x[:,:9]; o=np.argsort(-p,1); s=np.sort(p,1)[:,::-1]
    for t in range(10): F['x%d'%t]=x[:,t]
    F['P8']=s[:,0];F['Q8']=s[:,1];F['R8']=s[:,2]
    F['P7']=x[:,:8].max(1)
    for k in range(9):
        F['am%d'%k]=(o[:,0]==k)*1.; F['a2%d'%k]=(o[:,1]==k)*1.; F['t2%d'%k]=((o[:,0]==k)|(o[:,1]==k))*1.
    for k in range(9):
        F['x%d-Q8'%k]=x[:,k]-s[:,1]
        F['x%d-R8'%k]=x[:,k]-s[:,2]
    return F
