import numpy as np
def pm(x,t):  # max of x0..x_{t-1}
    return x[:,:t].max(1)
def ps(x,t):  # second max of x0..x_{t-1}
    return np.sort(x[:,:t],1)[:,-2]
def feats(x):
    F={}
    for k in range(10): F['x%d'%k]=x[:,k]
    for t in range(2,11): F['M%d'%t]=pm(x,t)
    for t in range(3,11): F['S%d'%t]=ps(x,t)
    return F
