import numpy as np, feats
import labels_fit as L
x=np.load('x1.npy')[:40000]; h=np.load('h1.npy')[:40000]
F=feats.pool3(x); names=list(F); A=np.stack([F[k] for k in names],1)
am=np.argmax(x,1)
for i in [14]:
    r=h[:,i]-L.f(i,x)
    cs=[(abs(np.corrcoef(A[:,k],r)[0,1]) if A[:,k].std()>0 else 0,names[k]) for k in range(len(names))]
    print(sorted(cs)[-12:])
    print('resid mean by am',[round(r[am==k].mean(),2) for k in range(10)])
    print('h14 mean by am', [round(h[am==k,i].mean(),1) for k in range(10)])
    # by second position
    a2=np.argsort(-x,1)[:,1]
    print('resid by a2',[round(r[a2==k].mean(),2) for k in range(10)])
    print('frac h14==0 by am',[round((h[am==k,i]<=0).mean(),2) for k in range(10)])
