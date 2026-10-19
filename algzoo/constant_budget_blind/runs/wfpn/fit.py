import numpy as np, probe
from featlib import basics, design
np.set_printoptions(suppress=True,precision=4)
xa=np.random.default_rng(7).standard_normal((100000,10)); ha=probe.hidden(xa)
xb=np.random.default_rng(8).standard_normal((100000,10)); hb=probe.hidden(xb)
ba=basics(xa); bb=basics(xb); C={}
for i in range(8):
  B=np.stack(design(i,ba),1); y=ha[:,i]; m=y>0
  c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  for it in range(5):
    mm=(B@c>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  # polish with gradient steps on relu loss
  L=lambda c: ((np.maximum(B@c,0)-y)**2).mean()
  lr=0.01
  for it in range(300):
    pr=B@c; r=np.maximum(pr,0)-y; grad=2*B[pr>0].T@r[pr>0]/len(y)
    c2=c-lr*grad
    if L(c2)<L(c): c=c2; lr*=1.2
    else: lr*=0.5
  Bt=np.stack(design(i,bb),1); pr=np.maximum(Bt@c,0)
  C[i]=c.round(4).tolist(); print(i, repr(c.round(4).tolist()), "test R2", round(1-((pr-hb[:,i])**2).mean()/hb[:,i].var(),4))

open('consts.txt','w').write(repr(C))
