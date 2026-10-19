import numpy as np, probe
from feat import *
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10)); h=probe.hidden(x)
F=feats(x)
print('mean',h.mean(0).round(3)); print('std',h.std(0).round(3)); print('frac>0',(h>1e-6).mean(0).round(3))
for i in range(8):
    cs=sorted(((abs(np.corrcoef(h[:,i],v)[0,1]),k) for k,v in F.items()),reverse=True)[:6]
    print(i,[(k,round(c,3)) for c,k in cs])
# linear R2 on x and sorted
A=np.c_[x,np.ones(len(x))]
for i in range(8):
    c=np.linalg.lstsq(A,h[:,i],rcond=None)[0]; print(i,'linR2',round(r2(h[:,i],A@c),3),c.round(2))
