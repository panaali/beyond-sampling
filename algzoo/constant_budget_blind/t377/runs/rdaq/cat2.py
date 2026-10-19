import numpy as np
from feats import *
def catalog2(x):
    C=catalog(x)
    o9=np.argsort(-x[:,:9],1); o10=np.argsort(-x,1)
    for j in range(10):
        if j<9:
            C['I1_%d'%j]=(o9[:,0]==j)*1.; C['I2_%d'%j]=(o9[:,1]==j)*1.
            o=np.delete(x[:,:9],j,1); s=-np.sort(-o,1)
            C['Mo9_%d'%j]=s[:,0]; C['So9_%d'%j]=s[:,1]
            C['rSo9_%d'%j]=np.maximum(x[:,j]-s[:,1],0); C['rMo9_%d'%j]=np.maximum(x[:,j]-s[:,0],0)
        C['J1_%d'%j]=(o10[:,0]==j)*1.; C['J2_%d'%j]=(o10[:,1]==j)*1.
    return C
