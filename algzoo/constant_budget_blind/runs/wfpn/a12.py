import numpy as np, sys
from pool import pool
x=np.load('x.npy')[:30000]; h=np.load('h.npy')[:30000]
F=pool(x); names=list(F); A=np.stack([F[n] for n in names],1)
def ev(cols,i):
  y=h[:,i]; m=y>0; B=A[:,cols]
  c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  pr=B@c; mm=(pr>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  pr=np.maximum(B@c,0); return 1-((pr-y)**2).mean()/y.var(), c
for i in [int(a) for a in sys.argv[1:]]:
  cols=[]
  for step in range(5):
    best=max((ev(cols+[j],i)[0],j) for j in range(len(names)) if j not in cols)
    cols.append(best[1]); r,c=ev(cols,i)
    print(i,step+1,round(r,3),[names[j] for j in cols],c.round(2),flush=True)
