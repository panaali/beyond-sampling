import numpy as np
from feats2 import feats2 as feats
x=np.load('x.npy'); h=np.load('h.npy')
F=feats(x); names=list(F); A=np.stack([F[n] for n in names],1)
def ev(cols,i):
  m=h[:,i]>0
  B=A[:,cols]; c,*_=np.linalg.lstsq(B[m],h[m,i],rcond=None)
  for it in range(3):  # refit on predicted-active or true-active
    pr=B@c; mm=(pr>0)|m; c,*_=np.linalg.lstsq(B[mm],h[mm,i],rcond=None)
  pr=np.maximum(B@c,0); return 1-((pr-h[:,i])**2).mean()/h[:,i].var(), c
for i in range(8):
  cols=[]
  for step in range(5):
    best=max((ev(cols+[j],i)[0],j) for j in range(len(names)) if j not in cols)
    cols.append(best[1])
    r,c=ev(cols,i)
    print(i,step+1,round(r,3),[names[j] for j in cols],c.round(2))
