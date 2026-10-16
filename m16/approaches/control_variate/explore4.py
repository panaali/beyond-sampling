import sys, numpy as np
sys.path.insert(0, "/home/user/clean-neurons/m16")
from arc_eval import forward, noisy_models, T
from common import Wi, Wh, Wo
from feats import feats
rng=np.random.default_rng(3); n=1_000_000
x=rng.standard_normal((n,T)); F=feats(x)
rank=np.argsort(np.argsort(-x,1),1)
for name,m in [('base',(Wi,Wh,Wo)),('n05_1',noisy_models(6,0.005)[1]),('n05_0',noisy_models(6,0.005)[0])]:
    pr=forward(m,x).argmax(1); Y=pr==F['p2']
    pred_rank=rank[np.arange(n),pr]
    print(name,"pred rank dist on errors",np.bincount(pred_rank[~Y],minlength=10)/n)
    big=(F['g23']>0.2)&(F['g12']>0.2)&~F['blind']
    e=~Y&big
    print(" big-gap errors: pred rank",np.bincount(pred_rank[e],minlength=10)/n)
    print(" by p2:",np.round([ (e&(F['p2']==p)).sum()/max(1,(big&(F['p2']==p)).sum()) for p in range(10)],3))
    print(" by p1:",np.round([ (e&(F['p1']==p)).sum()/max(1,(big&(F['p1']==p)).sum()) for p in range(10)],3))
    print(" by v2 quint:",np.round([ e[big&(F['v2']>a)&(F['v2']<=b)].mean() for a,b in [(0,.3),(.3,.6),(.6,1),(1,1.5),(1.5,9)]],3))
