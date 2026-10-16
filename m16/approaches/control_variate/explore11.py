"""Does the weight-derived threshold surrogate add anything WITHIN the stratified design's strata?"""
import sys, numpy as np
sys.path.insert(0,"/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi,Wh,Wo
from feats import feats
from thr import thresholds, TRIPLES
import stratified as S
rng=np.random.default_rng(11); n=1_000_000
x=rng.standard_normal((n,T)); F=feats(x); st=S.stratum(x)
tid=np.zeros((T,T,T),int); tid[TRIPLES[:,0],TRIPLES[:,1],TRIPLES[:,2]]=np.arange(len(TRIPLES)); k=tid[F['p1'],F['p2'],F['p3']]
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n025_0',noisy_models(6,0.0025)[0])]:
    Y=(forward(m,x).argmax(1)==F['p2']).astype(float)
    t23,t12,ev=thresholds(m,iters=8)
    Sv=((~F['blind'])&(F['g23']>t23[k])&(F['g12']>t12[k])).astype(float)
    a=[];b=[]
    for h in range(S.N_STRATA):
        y=Y[st==h]; s=Sv[st==h]; P=(st==h).mean()
        C=np.cov(y,s); beta=C[0,1]/C[1,1] if C[1,1]>0 else 0
        a.append(P*y.std()); b.append(P*(y-beta*s).std())
    print(name, f"Neyman-stratified sd-sum ratio^2 (within-stratum CV gain): {(sum(a)/sum(b))**2:.3f}, probe cost {ev}")
