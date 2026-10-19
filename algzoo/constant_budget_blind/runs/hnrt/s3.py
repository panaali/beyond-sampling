import numpy as np
d=np.load('grad.npz'); x,H,G=d['x'],d['H'],d['G']
N=len(x)
rank=np.argsort(np.argsort(-x,1),1)  # 0 = largest
np.set_printoptions(precision=1,suppress=True,linewidth=200)
for i in range(16):
    T=np.full((4,10),np.nan)
    for r in range(4):
        for j in range(10):
            m=rank[:,j]==r
            T[r,j]=np.median(G[m,j,i])
    rest=np.zeros(10)
    for j in range(10):
        m=rank[:,j]>=4; rest[j]=np.median(G[m,j,i])
    print("neuron",i); print(np.vstack([T,rest]))
