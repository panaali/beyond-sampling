import numpy as np, probe, labels
from feat import r2; from fit import fitrelu
rng=np.random.default_rng(34); x=rng.standard_normal((40000,10)); h=probe.hidden(x)
gcol={0:3,1:3,2:3,3:0,6:0,7:0}
for i in [0,1,2,3,6,7]:
    A=labels._design(i,x); j=gcol[i]; res=[]
    for k in range(0,6):
        for nm,G in [('abs',np.abs(x[:,:k+1]).sum(1)),('relu',np.maximum(x[:,:k+1],0).sum(1)),('max',x[:,:k+1].max(1)),('sum',x[:,:k+1].sum(1))]:
            B=A.copy(); B[:,j]=G; c,rr=fitrelu(B,h[:,i]); res.append((round(rr,3),nm,f'x0..x{k}'))
    res.sort(reverse=True); print(i,res[:4])
