import numpy as np
def base(x):
    F={}
    S=-np.sort(-x,1); N=len(x)
    for j in range(10): F["x%d"%j]=x[:,j]
    for j in range(3): F["S%d"%j]=S[:,j]
    for t in range(1,10): F["pm%d"%t]=x[:,:t+1].max(1)
    for t in range(1,9): F["sm%d"%t]=x[:,t:].max(1)
    for t in range(2,10): F["p2nd%d"%t]=np.sort(x[:,:t+1],1)[:,-2]
    for k in range(10):
        o=np.sort(np.delete(x,k,1),1)
        F["mex%d"%k]=o[:,-1]; F["s2ex%d"%k]=o[:,-2]
        F["isMax%d"%k]=np.maximum(x[:,k]-o[:,-1],0)        # margin by which x_k is the max
        F["inTop2_%d"%k]=np.maximum(x[:,k]-o[:,-2],0)      # margin by which x_k beats the 3rd-of-rest
        F["below%d"%k]=np.maximum(o[:,-1]-x[:,k],0)        # how far x_k is below max of others
        if k>0: F["rec%d"%k]=np.maximum(x[:,k]-x[:,:k].max(1),0)
    return F
