import numpy as np
def feats(x):
    N=len(x); F={}
    for j in range(10): F["x%d"%j]=x[:,j]
    for t in range(1,10): F["pm%d"%t]=x[:,:t+1].max(1)   # max x0..xt
    for t in range(1,9): F["sm%d"%t]=x[:,t:].max(1)      # max xt..x9
    for k in range(10): F["mex%d"%k]=np.delete(x,k,1).max(1)
    for t in range(2,10):
        s=np.sort(x[:,:t+1],1); F["p2nd%d"%t]=s[:,-2]
    S=-np.sort(-x,1)
    for j in range(1,3): F["S%d"%j]=S[:,j]
    return F
