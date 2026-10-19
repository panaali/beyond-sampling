import numpy as np, itertools
from fitlib import *
x=np.load('x1.npy')[:30000]; H=np.load('H1.npy')[:30000]
d=feats(x)
def score(i,S,gate):
  R=x[:,list(S)].max(1)
  F=np.stack([R,x[:,9]],1)
  y=H[:,i]
  m = (x[:,8]<d['M7']) if gate else np.ones(len(y),bool)
  c,_,p=relufit(F[m],y[m])
  pred=np.zeros_like(y); pred[m]=p
  return 1-((pred-y)**2).mean()/y.var(),c
for i in [0,3,6]:
  best=None
  for r in range(1,9):
    for S in itertools.combinations(range(8),r):
      pass
  # exhaustive over subsets of 0..7 (255) cheap-ish with lstsq approx: use linear fit on active
  res=[]
  y=H[:,i]; m=(x[:,8]<d['M7'])
  for r in range(1,9):
    for S in itertools.combinations(range(8),r):
      R=x[:,list(S)].max(1)
      a=m&(y>0)
      F=np.stack([R[a],x[a,9]],1); c,*_=np.linalg.lstsq(F,y[a],rcond=None)
      pred=np.zeros_like(y); pred[m]=np.maximum(np.stack([R[m],x[m,9]],1)@c,0)
      res.append((1-((pred-y)**2).mean()/y.var(),S,c))
  res.sort(key=lambda t:-t[0])
  for t in res[:4]: print(i,'%.4f'%t[0],t[1],t[2])
