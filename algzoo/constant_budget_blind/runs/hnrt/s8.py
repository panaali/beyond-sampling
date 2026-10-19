import numpy as np, probe
from util import *
from dict import build
rng=np.random.default_rng(7)
x=rng.standard_normal((60000,10)); H=probe.hidden(x)
D=build(x)
F1=np.stack([D['s1@%d'%j] for j in range(10)],1)
F2=np.stack([D['x0']]+[D['P%d'%t] for t in range(1,10)],1)
F3=np.stack([D['s2@%d'%j] for j in range(10)],1)
Fq=np.stack([D['Q%d'%t] for t in range(1,9)],1)
ex=np.stack([D['s2'],D['s3'],D['x9']],1)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in range(16):
    h=H[:,i]; res=[]
    for nm,F in [('s1pos',np.hstack([F1,ex])),('prefix',np.hstack([F2,ex])),('pre+s2pos',np.hstack([F2,F3,ex])),('pre+suf',np.hstack([F2,Fq,ex])),('all',np.hstack([F1,F2,F3,Fq,ex]))]:
        w=relufit(F,h,15); res.append((nm,round(r2(np.maximum(F@w,0),h),3)))
        if nm=='prefix': wp=w
    print(i,res,'prefixcoef',wp)
