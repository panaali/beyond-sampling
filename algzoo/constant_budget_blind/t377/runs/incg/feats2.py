import numpy as np
P5={3:[1,4,7],9:[0,1,4,5,6,7,8],13:[3,5,6,8],14:[0,1,2,3,4,5],17:[0,1,2,3],18:[0,5,7,8],20:[0,1,3,4],22:[1,3,6,7,8],23:[0,5,7,8,9],24:[0,1,2,3,4,5,6],28:None,30:None,12:[2,3,4,6,7]}
def feats(x):
    F={}
    S=-np.sort(-x,1)
    for j in range(10): F["x%d"%j]=x[:,j]
    for t in range(1,10): F["pm%d"%t]=x[:,:t+1].max(1)
    for t in range(1,9): F["sm%d"%t]=x[:,t:].max(1)
    for k in range(10):
        o=np.sort(np.delete(x,k,1),1); F["mex%d"%k]=o[:,-1]; F["s2ex%d"%k]=o[:,-2]
        F["rec%d"%k]=np.maximum(x[:,k]-(x[:,:k].max(1) if k>0 else -np.inf),0) if k>0 else np.zeros(len(x))
        F["topgap%d"%k]=np.maximum(x[:,k]-o[:,-1],0)
    for t in range(2,10): F["p2nd%d"%t]=np.sort(x[:,:t+1],1)[:,-2]
    for j in range(3): F["S%d"%j]=S[:,j]
    for g in range(9):  # max over j<9 excluding... group maxima of pairs/triples skipped
        pass
    return F
