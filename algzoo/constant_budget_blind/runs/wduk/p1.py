import numpy as np, probe
from feat import feats
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
F,s,o=feats(x)
print('acc',(p==o[:,1]).mean())
np.save('x1.npy',x);np.save('h1.npy',h)
for i in range(16):
    hi=h[:,i]
    cs={k:np.corrcoef(hi,v)[0,1] for k,v in F.items()}
    top=sorted(cs.items(),key=lambda t:-abs(t[1]))[:5]
    print(i,'mean %.2f std %.2f frac0 %.2f'%(hi.mean(),hi.std(),(hi==0).mean()),' '.join(f'{k}:{v:.2f}' for k,v in top))
