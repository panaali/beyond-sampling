import numpy as np
def build(x):
    N=len(x); o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    D={}
    for j in range(10): D['x%d'%j]=x[:,j]
    for k in range(3): D['s%d'%(k+1)]=s[:,k]
    for t in range(1,10): D['P%d'%t]=x[:,:t+1].max(1)
    for t in range(1,9): D['Q%d'%t]=x[:,t:].max(1)
    for k in range(2):
        for j in range(10): D['s%d@%d'%(k+1,j)]=s[:,k]*(o[:,k]==j)
        for c in range(3): D['s%d@m%d'%(k+1,c)]=s[:,k]*(o[:,k]%3==c)
    for c in range(3): D['M%d'%c]=x[:,c::3].max(1)
    for j in range(1,10): D['new%d'%j]=np.maximum(x[:,j]-x[:,:j].max(1),0)
    return D
