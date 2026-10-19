import numpy as np, probe
from feat import *; from fit import *
np.set_printoptions(precision=3,suppress=True,linewidth=150)
rng=np.random.default_rng(3)
x=rng.standard_normal((50000,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; rm=np.maximum.accumulate(x,1)
srt8=np.sort(x[:,:9],1)[:,::-1]
one=np.ones(len(x))
for i in [5,4]:
    y=h[:,i]
    for name,A in [('x',np.c_[x,one]),('M,S,x9',np.c_[s[:,0],s[:,1],x[:,9],one]),
                   ('pre8 M,S,x8,x9',np.c_[srt8[:,0],srt8[:,1],x[:,8],x[:,9],one]),
                   ('pre8 sorted, x9',np.c_[srt8,x[:,9],one]),('pre8 sorted,x8,x9',np.c_[srt8,x[:,8],x[:,9],one])]:
        c,r=fitrelu(A,y); print(i,name,round(r,4),c)
