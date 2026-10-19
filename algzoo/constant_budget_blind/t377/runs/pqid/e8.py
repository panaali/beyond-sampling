import numpy as np
x=np.load('x2.npy');h=np.load('h2.npy')
o=np.argsort(-x,1); am,a2=o[:,0],o[:,1]
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [3,13,15]:
    t=np.full((10,10),np.nan)
    for a in range(10):
        for b in range(10):
            g=(am==a)&(a2==b)
            if g.sum()>50: t[a,b]=h[g,i].mean()
    print('neuron',i,'rows am, cols a2'); print(t)
