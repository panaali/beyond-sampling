import numpy as np
x=np.load('x1.npy');H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
for i in [2,15,19,22,14,26,27,11,10,4,3,8,20,24,30,28,25]:
    y=H[:,i]
    tab=np.zeros((10,3))
    for j in range(10):
        tab[j,0]=y[am==j].mean(); tab[j,1]=y[a2==j].mean(); tab[j,2]=y[(am!=j)&(a2!=j)].mean()
    print(i,'mean|am=j',np.round(tab[:,0],2),'\n   mean|a2=j',np.round(tab[:,1],2))
