"""Which gates matter? Per (ordering,K) canonical gates vs actual; greedy: which (neuron,time) gates, if taken
from the actual input, fix the surrogate the most."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from common import Wi,Wh,Wo
T=10
ref=np.sort(np.random.default_rng(0).standard_normal((1000000,T)),1)[:,::-1]; Kref=(ref>0).sum(1)
EOSK=np.array([ref[Kref==k].mean(0) for k in range(T+1)])
rng=np.random.default_rng(7)
N=60000; x=rng.standard_normal((N,T)); order=np.argsort(-x,1); K=(x>0).sum(1); tgt=order[:,1]
xc=np.empty_like(x); np.put_along_axis(xc,order,EOSK[K],1)
def gates(X):
    h=np.zeros((len(X),16)); G=np.zeros((len(X),T,16),bool)
    for t in range(T):
        u=h@Wh.T+X[:,t:t+1]*Wi; G[:,t]=u>0; h=u*G[:,t]
    return G
def surrogate(X,G):
    h=np.zeros((len(X),16))
    for t in range(T): h=(h@Wh.T+X[:,t:t+1]*Wi)*G[:,t]
    return (h@Wo.T).argmax(1)
Ga=gates(x); Gc=gates(xc)
print('true acc',(surrogate(x,Ga)==tgt).mean(),' canonical-gate surrogate acc',(surrogate(x,Gc)==tgt).mean(),
      'agree',(surrogate(x,Gc)==surrogate(x,Ga)).mean())
print('gate disagreement rate per neuron (avg over t):',np.round((Ga!=Gc).mean((0,1)),3))
print('per time:',np.round((Ga!=Gc).mean((0,2)),3))
# neuron-level fixes
ya=surrogate(x,Ga)
G=Gc.copy(); chosen=[]
for step in range(6):
    best=None
    for n in range(16):
        if n in chosen: continue
        G2=G.copy(); G2[:,:,n]=Ga[:,:,n]; a=(surrogate(x,G2)==ya).mean()
        if best is None or a>best[0]: best=(a,n)
    chosen.append(best[1]); G[:,:,best[1]]=Ga[:,:,best[1]]; print('fix neuron',best[1],'agreement',best[0], 'acc',(surrogate(x,G)==tgt).mean())
