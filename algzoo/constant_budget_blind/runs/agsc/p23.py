import numpy as np, probe, labels
from feat import r2; from fit import fitrelu
np.set_printoptions(precision=2,suppress=True,linewidth=160)
rng=np.random.default_rng(32); x=rng.standard_normal((100000,10)); h=probe.hidden(x)
am=np.argmax(x,1); o=np.argsort(-x,1); asec=o[:,1]
for i,cut in [(0,5),(1,6),(2,5),(3,7),(6,4),(7,6)]:
    A=labels._design(i,x)
    for nm,m in [('amax<=cut',am<=cut),('amax>cut',am>cut)]:
        c,rr=fitrelu(A[m],h[m,i]); print(i,nm,m.mean().round(2),'R2',round(rr,3),c)
