import numpy as np, probe, itertools
from feat import r2; from fit import fitrelu
from lib import library
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(11); N=30000
x=rng.standard_normal((N,10)); h=probe.hidden(x); L=library(x); one=np.ones(N)
D={}; Dn={}
for p in range(1,10):
    sp=np.sort(x[:,:p],1)[:,::-1]; m1=sp[:,0]; m2=sp[:,1] if p>1 else sp[:,0]
    D[p]=np.maximum(0.4*m1+0.6*m2-x[:,p],0)
    Dn[p]=np.maximum(x[:,p]-m1,0)  # new record
for i in range(8):
    y=h[:,i]; res=[]
    for p,q in itertools.combinations(range(1,10),2):
        for base in ['MSlag','M+S']:
            A=np.c_[L[base],D[p],D[q],one]; c=np.linalg.lstsq(A,y,rcond=None)[0]; res.append((r2(y,A@c),base,p,q,c))
    res.sort(key=lambda z:-z[0]); print(i,[(round(a,3),b,p,q,np.round(c,2)) for a,b,p,q,c in res[:2]])
