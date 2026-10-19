import numpy as np, probe
from util import *
rng=np.random.default_rng(3)
x=rng.standard_normal((100000,10)); H=probe.hidden(x)
o,s=feats(x)
N=len(x)
F=[]
for r in range(3):
    oh=np.zeros((N,10)); oh[np.arange(N),o[:,r]]=s[:,r]; F.append(oh)
F.append(x[:,9:10])
F=np.hstack(F)
np.set_printoptions(precision=1,suppress=True,linewidth=200)
for i in range(16):
    w=relufit(F,H[:,i]); p=np.maximum(F@w,0)
    print(i,"R2 relu",round(r2(p,H[:,i]),3)); print(w[:30].reshape(3,10)); print(' x9',w[30])
