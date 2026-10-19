import numpy as np, itertools, sys, pickle
from fitlib import *
x=np.load('x1.npy')[:20000]; H=np.load('H1.npy')[:20000]
d=feats(x); pa,pb=d['pa'],d['pb']
pool={}
for nm in ['P1','P2','P3','P4','x8','M7']: pool[nm]=d[nm]
subs=[S for r in range(1,9) for S in itertools.combinations(range(9),r)]
for S in subs:
  pool['max'+''.join(map(str,S))]=x[:,list(S)].max(1)
  pool['P1@'+''.join(map(str,S))]=d['P1']*np.isin(pa,S)
  pool['P2@'+''.join(map(str,S))]=d['P2']*np.isin(pb,S)
names=list(pool); A=np.stack([pool[n] for n in names],1)
neur=[int(a) for a in sys.argv[1:]]
res={}
for i in neur:
  y=H[:,i]; act=y>1e-6; sel=[]; cur=[d['x9']]
  for step in range(4):
    best=(-9,None)
    base=np.stack(cur,1)
    for j in range(A.shape[1]):
      F=np.column_stack([base,A[:,j]])
      c,*_=np.linalg.lstsq(F[act],y[act],rcond=None)
      pred=np.maximum(F@c,0); r2=1-((pred-y)**2).mean()/y.var()
      if r2>best[0]: best=(r2,j)
    sel.append(names[best[1]]); cur.append(A[:,best[1]])
    print(i,step,sel,'%.4f'%best[0],flush=True)
  res[i]=sel
pickle.dump(res,open('greedy_%s.pkl'%'_'.join(sys.argv[1:]),'wb'))
