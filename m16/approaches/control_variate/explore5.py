import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from feats import feats
from thr import thresholds, TRIPLES
rng=np.random.default_rng(4); n=1_000_000
x=rng.standard_normal((n,T)); F=feats(x)
tid=np.zeros((T,T,T),int); tid[TRIPLES[:,0],TRIPLES[:,1],TRIPLES[:,2]]=np.arange(len(TRIPLES))
k=tid[F['p1'],F['p2'],F['p3']]
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n05_0',noisy_models(6,0.005)[0]),('n025_0',noisy_models(6,0.0025)[0])]:
    Y=(forward(m,x).argmax(1)==F['p2']).astype(float)
    t23,t12,ev=thresholds(m,iters=10)
    S=(~F['blind'])&(F['g23']>t23[k])&(F['g12']>t12[k]); S=S.astype(float)
    C=np.cov(Y,S); beta=C[0,1]/C[1,1]; r=Y-beta*S
    print(name,f"acc {Y.mean():.4f} ES {S.mean():.4f} agree {(Y==S).mean():.4f} beta {beta:.2f} gain {Y.var()/r.var():.2f} evals {ev}")
