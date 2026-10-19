import numpy as np, sys
import labels as L
from feats import feats
d=np.load('d1.npz');x=d['x'][:50000];H=d['H'][:50000]
F=feats(x);o=np.argsort(-x,1);am=o[:,0];a2=o[:,1];M=x.max(1);S=-np.sort(-x,1)[:,1]
for k in range(10):
    F[f'M@am{k}']=M*(am==k);F[f'S@a2_{k}']=S*(a2==k);F[f'1@am{k}']=(am==k)*1.;F[f'1@a2_{k}']=(a2==k)*1.
names=list(F);A=np.stack([F[n] for n in names],1);A=(A-A.mean(0))/(A.std(0)+1e-9)
for i in [int(a) for a in sys.argv[1:]]:
    r=H[:,i]-L.f(i,x); r=r-r.mean()
    c=A.T@r/len(r)/r.std()
    idx=np.argsort(-np.abs(c))[:10]
    print(i,"resid std %.2f"%r.std(),[(names[j],round(c[j],2)) for j in idx])
