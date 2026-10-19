import numpy as np
from feats import feats
x=np.load('x0.npy');h=np.load('h0.npy');F=feats(x)
np.set_printoptions(linewidth=200)
for i in [0,10,12,14,15,3,13,2,7]:
    print('neuron',i)
    for p in [1,3,5,8]:
        g=(F['a1']==p)&(h[:,i]>0.5)
        xx=x[g].copy(); m=xx[:,p].copy(); xx[:,p]=0
        A=np.c_[m,xx,np.ones(g.sum())]
        c,_,_,_=np.linalg.lstsq(A,h[g,i],rcond=None)
        r2=1-((A@c-h[g,i])**2).mean()/h[g,i].var()
        print(f"  p={p} m1 {c[0]:6.2f} xk",np.round(c[1:11],1),f"b {c[11]:.2f} r2 {r2:.2f}")
