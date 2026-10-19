import numpy as np, sys, pickle
from fitlib import *
X=np.load('x1.npy'); HH=np.load('H1.npy')
CIRC=[2,5,8,1,4,7,0,3,6]
arcs=[tuple(sorted(CIRC[(s+j)%9] for j in range(L))) for s in range(9) for L in range(1,9)]
arcs=sorted(set(arcs))
def build(x):
  d=feats(x); pa,pb=d['pa'],d['pb']; pool={}
  for nm in ['P1','P2','P3','x8','M7']: pool[nm]=d[nm]
  for S in arcs:
    k=''.join(map(str,S))
    pool['max'+k]=x[:,list(S)].max(1)
    pool['P1@'+k]=d['P1']*np.isin(pa,S)
    pool['P2@'+k]=d['P2']*np.isin(pb,S)
  return d,pool
d,pool=build(X[:20000]); names=list(pool); A=np.stack([pool[n] for n in names],1)
dt,poolt=build(X[50000:])
res={}
for i in [int(a) for a in sys.argv[1:]]:
  y=HH[:20000,i]; act=y>1e-6; sel=[]; cur=[d['x9']]
  for step in range(4):
    best=(-9,None)
    base=np.stack(cur,1)
    for j in range(A.shape[1]):
      F=np.column_stack([base,A[:,j]])
      c,*_=np.linalg.lstsq(F[act],y[act],rcond=None)
      r2=1-((np.maximum(F@c,0)-y)**2).mean()/y.var()
      if r2>best[0]: best=(r2,j)
    sel.append(names[best[1]]); cur.append(A[:,best[1]])
  out=[]
  for n in range(1,5):
    F=np.stack(cur[:n+1],1); c,r2,_=relufit(F,y)
    Ft=np.stack([dt['x9']]+[poolt[s] for s in sel[:n]],1); yt=HH[50000:,i]
    r2t=1-((np.maximum(Ft@c,0)-yt)**2).mean()/yt.var()
    out.append((n,sel[:n],c.round(3).tolist(),round(r2t,4)))
    print(i,n,sel[:n],c.round(2),'test R2 %.4f'%r2t,flush=True)
  res[i]=out
pickle.dump(res,open('g2_%s.pkl'%'_'.join(sys.argv[1:]),'wb'))
