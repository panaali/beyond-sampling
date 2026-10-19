import numpy as np, probe
from feat import feats
x=np.random.default_rng(1).standard_normal((100000,10))
h=probe.hidden(x); F=feats(x)
print('mean',h.mean(0).round(2)); print('frac>0',(h>0).mean(0).round(2)); print('std',h.std(0).round(2))
print('acc',(probe.predict(x)==np.argsort(-x,1)[:,1]).mean())
for i in range(16):
    c={k:np.corrcoef(v,h[:,i])[0,1] for k,v in F.items() if v.std()>0}
    top=sorted(c.items(),key=lambda t:-abs(t[1]))[:7]
    print(i,' '.join(f'{k}:{v:.2f}' for k,v in top))
