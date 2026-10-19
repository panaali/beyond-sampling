import numpy as np, probe
from fitlib import *
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
xt=np.random.default_rng(7).standard_normal((30000,10));ht=probe.hidden(xt)
F=pf(x);Ft=pf(xt)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
for i in [1,2,4,5,7,8,9,11,12,13,14]:
    W=rankw(x,h[:,i]);w=W[1:].mean(0);G=W[0]
    dev=w-np.median(w)
    out=[]
    for sign in (1,-1):
        order=np.argsort(-sign*dev)
        for k in (1,2,3):
            A=list(order[:k])
            for excl in (True,False):
                th,cg=fit9(x,F,h[:,i],G,A,excl)
                out.append((r2(m9(xt,Ft,th,cg,A,excl),ht[:,i]),A,excl,np.round(th,1),cg))
    out.sort(key=lambda t:-t[0])
    print(i,'w',w);
    for o in out[:3]: print('   %.3f'%o[0],o[1:])
