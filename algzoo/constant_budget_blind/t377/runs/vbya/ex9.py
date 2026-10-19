import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy'); H=np.load('H1.npy')
rk=np.argsort(np.argsort(-x,1),1)  # rank of each position, 0=largest
bins=np.array([-9,-1,0,0.5,1,1.5,2,9])
for i,t in [(29,0),(21,1),(1,2),(11,3),(2,4),(23,6),(26,7),(14,8),(24,7),(12,5),(31,8),(27,8),(10,8),(8,9),(13,9)]:
    h=H[:,i]; r=rk[:,t]
    print(i,'t',t,'mean h by rank of x_t:',np.array([h[r==k].mean() for k in range(10)]))
    print('     frac active by rank', np.array([(h[r==k]>0).mean() for k in range(10)]))
