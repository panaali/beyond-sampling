import numpy as np, probe
from feat import feats
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
F=feats(x)
print('acc',(p==F['a2']).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print('mean',h.mean(0));print('std',h.std(0));print('frac>0',(h>0).mean(0))
keys=['max','sec','third','m-s']+[f'x{k}' for k in range(10)]
for i in range(16):
    c=[np.corrcoef(h[:,i],F[k])[0,1] for k in keys]
    # mean by am and a2
    ma=[h[F['am']==j,i].mean() for j in range(10)]
    m2=[h[F['a2']==j,i].mean() for j in range(10)]
    print(i,' '.join(f'{k}:{v:.2f}' for k,v in zip(keys,c)))
    print('   by am',np.array(ma)); print('   by a2',np.array(m2))
