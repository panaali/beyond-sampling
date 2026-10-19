import numpy as np
from feats import feats
x=np.load('x0.npy');h=np.load('h0.npy');F=feats(x)
P8=x[:,:9].max(1)
for i in [1,6,7,8,0,10,11,12,14,15,2,3,13]:
    print('neuron',i)
    for p in range(10):
        g=(F['a1']==p)&(h[:,i]>0.5)
        A=np.c_[F['m1'][g],F['m2'][g],x[g,9],np.ones(g.sum())]
        c,res,_,_=np.linalg.lstsq(A,h[g,i],rcond=None)
        r2=1-((A@c-h[g,i])**2).mean()/h[g,i].var()
        print(f"  a1={p} n={g.sum():5d} frac_act={g.sum()/(F['a1']==p).sum():.2f} m1 {c[0]:6.2f} m2 {c[1]:6.2f} x9 {c[2]:6.2f} b {c[3]:6.2f} r2 {r2:.2f}")
