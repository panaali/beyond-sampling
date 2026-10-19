import numpy as np, probe, labels
from feat import r2; from fit import fitrelu
rng=np.random.default_rng(31); x=rng.standard_normal((100000,10)); h=probe.hidden(x)
xt=rng.standard_normal((100000,10)); ht=probe.hidden(xt)
out={}
for i in range(8):
    A=labels._design(i,x); c,rr=fitrelu(A,h[:,i]); out[i]=[round(float(v),4) for v in c]
    pt=np.maximum(labels._design(i,xt)@c,0); print(i,'train',round(rr,4),'test',round(r2(ht[:,i],pt),4),out[i])
print(out)
