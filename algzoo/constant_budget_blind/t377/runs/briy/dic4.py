import numpy as np
from dic3 import dict3, pref
def J(x,K):
    M=np.maximum.accumulate(x,1); d=np.diff(np.c_[np.zeros(len(x)),M],axis=1)  # d0=x0
    od=sum(d[:,k] for k in range(1,K+1,2)); ev=sum(d[:,k] for k in range(0,K+1,2))
    return od,ev
def dict4(x):
    D=dict3(x)
    for K in (5,6,7,8,9):
        od,ev=J(x,K); D[f'Jodd{K}']=od; D[f'Jeven{K}']=ev
    M=np.maximum.accumulate(x,1)
    for k in range(2,10): D[f'relu(x{k}-S{k-1})']=np.maximum(x[:,k]-pref(x,k-1)[1],0)
    return D
