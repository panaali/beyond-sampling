import numpy as np
x=np.load('x1.npy');H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
W=x[:,:9]; o9=np.argsort(-W,1); am9=o9[:,0]; a29=o9[:,1]
for i in [3,4,8,9,10,5,20,24,30,28,25,1,31,17]:
    y=H[:,i]
    T=np.zeros((10,10)); 
    for a in range(10):
        for b in range(10):
            m=(am==a)&(a2==b)
            T[a,b]=y[m].mean() if m.sum()>20 else np.nan
    print(i,'rows=argmax(all 10), cols=argsecond'); print(np.round(T,1))
