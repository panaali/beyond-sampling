import numpy as np, probe
from feat import feats
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); F=feats(x)
np.save('x1.npy',x); np.save('h1.npy',h)
acc=(probe.predict(x)==F['a2']).mean(); print('acc',acc)
for i in range(32):
    hi=h[:,i]
    c=[np.corrcoef(hi,x[:,j])[0,1] for j in range(10)]
    cm=[np.corrcoef(hi,F['am']==j)[0,1] for j in range(10)]
    c2=[np.corrcoef(hi,F['a2']==j)[0,1] for j in range(10)]
    print(i,'mean %.2f std %.2f frac0 %.2f'%(hi.mean(),hi.std(),(hi==0).mean()))
    print('  x ',' '.join('%+.2f'%v for v in c))
    print('  am',' '.join('%+.2f'%v for v in cm))
    print('  a2',' '.join('%+.2f'%v for v in c2))
    print('  m1 %+.2f m2 %+.2f m3 %+.2f'%tuple(np.corrcoef(hi,F[k])[0,1] for k in ['m1','m2','m3']))
