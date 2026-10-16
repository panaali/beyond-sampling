import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo,run
from collections import Counter
rng=np.random.default_rng(1); N=200000
x=rng.standard_normal((N,10)); hs=run(x)
G=(hs[:,1:]>0).reshape(N,-1)
keys=np.packbits(G,1); keys=[k.tobytes() for k in keys]
c=Counter(keys); print('distinct activation regions',len(c))
cnt=np.array(sorted(c.values()))[::-1]; cs=np.cumsum(cnt)/N
for q in [0.5,0.9,0.99,0.999]: print(q, np.searchsorted(cs,q)+1)
# record pattern
def pattern(x):
    N=len(x); code=np.zeros(N,np.int64); m1=x[:,0].copy(); m2=np.full(N,-np.inf)
    for t in range(1,10):
        e=np.where(x[:,t]>m1,0,np.where(x[:,t]>m2,1,2))
        code=code*3+e
        m2=np.where(e==0,m1,np.where(e==1,x[:,t],m2)); m1=np.where(e==0,x[:,t],m1)
    return code
pc=pattern(x)
print('patterns seen',len(set(pc)))
# within pattern, regions
d={}
for k,p in zip(keys,pc): d.setdefault(p,Counter())[k]+=1
# fraction of mass in the modal region of its pattern
tot=sum(max(v.values()) for v in d.values()); print('modal region per pattern covers',tot/N)
nr=[len(v) for v in d.values()]; print('mean regions per pattern',np.mean(nr))
