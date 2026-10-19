import numpy as np, probe, labels
from feat import r2; from fit import fitrelu
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(33); x=rng.standard_normal((40000,10)); h=probe.hidden(x)
gcol={0:3,1:2,2:3,3:0,6:0,7:0}
for i in [0,1,2,3,6,7]:
    A=labels._design(i,x); j=gcol[i]; res=[]
    _,r0=fitrelu(A,h[:,i]); 
    for k in range(1,10):
        s=np.sort(x[:,:k+1],1)
        for nm,G in [('m1',s[:,-1]),('m1+m2',s[:,-1]+s[:,-2]),('relusum',np.maximum(x[:,:k+1],0).sum(1))]:
            B=A.copy(); B[:,j]=G; c,rr=fitrelu(B,h[:,i]); res.append((round(rr,3),nm,f'x0..x{k}'))
    res.sort(reverse=True); print(i,'base',round(r0,3),res[:4])
