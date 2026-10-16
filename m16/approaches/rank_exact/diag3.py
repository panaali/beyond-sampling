"""Diagnostic: classes = (ordering, K=#positives). Canonical point = E[z | K]."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from diag2 import gates, linmap, fwd, T
rng=np.random.default_rng(7)
ref=np.sort(np.random.default_rng(0).standard_normal((1000000,T)),1)[:,::-1]
Kref=(ref>0).sum(1)
EOSK=np.array([ref[Kref==k].mean(0) for k in range(T+1)])
bank=np.sort(rng.standard_normal((20000,T)),1)[:,::-1]; Kb=(bank>0).sum(1)
res=[]
for it in range(300):
    perm=rng.permutation(T); tgt=perm[1]
    X=np.empty_like(bank); X[:,perm]=bank
    ok=(fwd(X).argmax(1)==tgt)
    s=np.zeros(len(X),bool)
    for k in range(T+1):
        idx=Kb==k
        if not idx.any(): continue
        xc=np.empty(T); xc[perm]=EOSK[k]
        A=linmap(gates(xc)); s[idx]=((X[idx]@A.T).argmax(1)==tgt)
    res.append((ok.mean(),s.mean(),(ok==s).mean(), ok[Kb>=2].mean(), s[Kb>=2].mean()))
r=np.array(res); print('truth, surrogate, agreement, truth K>=2, surr K>=2:',r.mean(0))
print('mean abs err per ordering',np.abs(r[:,1]-r[:,0]).mean())
