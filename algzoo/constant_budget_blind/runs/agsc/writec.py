import numpy as np, probe, labels, re
from feat import r2; from fit import fitrelu
rng=np.random.default_rng(31); x=rng.standard_normal((100000,10)); h=probe.hidden(x)
out={}
for i in range(8):
    c,rr=fitrelu(labels._design(i,x),h[:,i]); out[i]=[round(float(v),4) for v in c]
s=open('labels.py').read()
s=s.replace("C = {i: None for i in range(8)}","C = "+repr(out))
open('labels.py','w').write(s); print(out)
