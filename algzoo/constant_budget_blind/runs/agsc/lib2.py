import numpy as np
from lib import library
def lib2(x):
    L=library(x); N=len(x)
    rm=np.maximum.accumulate(x,1)
    for t in range(1,10):
        L[f'DR{t}']=np.maximum(rm[:,t-1]-x[:,t],0)
        L[f'NR{t}']=np.maximum(x[:,t]-rm[:,t-1],0)   # amount by which x_t sets a new record
        L[f'CAP{t}']=np.minimum(x[:,t],rm[:,t-1])
    return L
def lib3(x):
    L=lib2(x); N=len(x)
    for t in range(2,10):
        s=np.sort(x[:,:t],1)
        L[f'DS{t}']=np.maximum(s[:,-2]-x[:,t],0)
        L[f'DA{t}']=np.maximum(0.5*(s[:,-1]+s[:,-2])-x[:,t],0)
    return L
