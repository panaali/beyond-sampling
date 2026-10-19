import numpy as np, sys, pickle
from fitlib import *
X=np.load('x1.npy'); HH=np.load('H1.npy')
CIRC=[2,5,8,1,4,7,0,3,6]
arcs=sorted(set(tuple(sorted(CIRC[(s+j)%9] for j in range(L))) for s in range(9) for L in range(1,9)))
def build(x):
  d=feats(x); pa,pb=d['pa'],d['pb']; o=np.argsort(-x[:,:9],1); pc=o[:,2]; pool={}
  for nm in ['P1','P2','P3','P4','M7']: pool[nm]=d[nm]
  for t in range(10): pool['x%d'%t]=x[:,t]
  pool['P1*[pa<pb]']=d['P1']*(pa<pb); pool['P2*[pb<pa]']=d['P2']*(pb<pa); pool['P2*[pb>pa]']=d['P2']*(pb>pa)
  for S in arcs:
    k=''.join(map(str,S))
    xs=x[:,list(S)]
    pool['max'+k]=xs.max(1)
    if len(S)>1: pool['2nd'+k]=-np.sort(-xs,1)[:,1]
    pool['P1@'+k]=d['P1']*np.isin(pa,S)
    pool['P2@'+k]=d['P2']*np.isin(pb,S)
    pool['P3@'+k]=d['P3']*np.isin(pc,S)
  return d,pool
d,pool=build(X[:20000]); names=list(pool); A=np.stack([pool[n] for n in names],1)
dt,poolt=build(X[50000:])
for i in [int(a) for a in sys.argv[1:]]:
  y=HH[:20000,i]; act=y>1e-6; sel=[]; cur=[]
  for step in range(5):
    best=(-9,None)
    for j in range(A.shape[1]):
      F=np.column_stack(cur+[A[:,j]])
      c,*_=np.linalg.lstsq(F[act],y[act],rcond=None)
      r2=1-((np.maximum(F@c,0)-y)**2).mean()/y.var()
      if r2>best[0]: best=(r2,j)
    sel.append(names[best[1]]); cur.append(A[:,best[1]])
    F=np.stack(cur,1); c,r2,_=relufit(F,y)
    Ft=np.stack([poolt[s] for s in sel],1); yt=HH[50000:,i]
    r2t=1-((np.maximum(Ft@c,0)-yt)**2).mean()/yt.var()
    print(i,step+1,sel,c.round(2),'test R2 %.4f'%r2t,flush=True)
