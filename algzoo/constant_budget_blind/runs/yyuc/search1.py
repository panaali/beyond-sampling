import numpy as np, itertools, sys
from fitlib import *
x=np.load('x1.npy')[:30000]; H=np.load('H1.npy')[:30000]
d=feats(x); pa=d['pa']
subsets=[S for r in range(1,9) for S in itertools.combinations(range(9),r)]
out={}
for i in range(16):
  if i in (10,15): continue
  y=H[:,i]; best=(-9,None)
  act=y>1e-6
  for S in subsets:
    g=np.isin(pa,S).astype(float)
    F=np.stack([d['P1'],d['P2'],d['P1']*g,d['P2']*g,d['x9']],1)
    c,*_=np.linalg.lstsq(F[act],y[act],rcond=None)
    pred=np.maximum(F@c,0); r2=1-((pred-y)**2).mean()/y.var()
    if r2>best[0]: best=(r2,S,c)
  r2,S,c=best
  g=np.isin(pa,S).astype(float)
  F=np.stack([d['P1'],d['P2'],d['P1']*g,d['P2']*g,d['x9']],1)
  c2,r22,_=relufit(F,y,c)
  print(i,'S',S,'lin %.3f relu %.3f'%(r2,r22),c2.round(2),flush=True)
