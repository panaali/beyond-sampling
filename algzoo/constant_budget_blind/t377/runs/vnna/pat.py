import numpy as np, sys
from collections import Counter
np.set_printoptions(precision=2,suppress=True,linewidth=200)
G=np.load('G.npy'); x=np.load('Gx.npy')
i=int(sys.argv[1])
g=np.round(G[:,:,i],1)
keys=[tuple(r) for r in g]
C=Counter(keys)
s=np.argsort(-x,1)
for k,n in C.most_common(int(sys.argv[2]) if len(sys.argv)>2 else 12):
    m=np.array([kk==k for kk in keys])
    ranks=np.argsort(np.argsort(-x[m],1),1)  # rank of each position
    print(n, np.array(k), "mean rank by pos", ranks.mean(0).round(1))
