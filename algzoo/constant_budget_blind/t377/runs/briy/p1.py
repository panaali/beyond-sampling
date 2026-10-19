import numpy as np, probe
from feat import feats
rng=np.random.default_rng(1); x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
o=np.argsort(-x,1); print('acc',(p==o[:,1]).mean())
np.save('x1.npy',x); np.save('h1.npy',h)
F=feats(x)
for i in range(16):
    hi=h[:,i]; print(i,'mean %.3f std %.3f frac0 %.3f'%(hi.mean(),hi.std(),(hi<1e-6).mean()))
    cs=[(abs(np.corrcoef(hi,v)[0,1]),k) for k,v in F.items() if v.std()>0]
    cs.sort(reverse=True); print('   ',[(k,round(c,2)) for c,k in cs[:5]])
    # mean by argmax and a2
    print('    by am',np.round([hi[F['am']==k].mean() for k in range(10)],2))
    print('    by a2',np.round([hi[F['a2']==k].mean() for k in range(10)],2))
