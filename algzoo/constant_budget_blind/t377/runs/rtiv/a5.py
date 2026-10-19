import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
o=np.argsort(-x,1);am=o[:,0];a2=o[:,1]
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in [0,1,2,4,7]:
    print("neuron",i)
    for k in range(10):
        m=(am==k);y=H[m,i];X=np.c_[x[m]]
        c,*_=np.linalg.lstsq(X,y,rcond=None);r=1-((y-X@c)**2).mean()/y.var()
        print(k,round(r,3),c)
