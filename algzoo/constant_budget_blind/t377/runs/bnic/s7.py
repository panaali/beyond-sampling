import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
M=np.maximum.accumulate(x,1); T=M[:,9:]-x
bins=[0,1e-9,0.25,0.5,0.75,1,1.5,2,2.5,3,4,6]
for i,j in [(8,7),(14,4),(10,3),(10,7),(15,8),(2,8),(0,5),(0,7),(9,6),(4,5),(7,1),(13,2)]:
    t=T[:,j]; d=np.digitize(t,bins)
    print(i,j,' '.join(f'{h[d==k,i].mean():.0f}±{h[d==k,i].std():.0f}' for k in range(1,len(bins))))
