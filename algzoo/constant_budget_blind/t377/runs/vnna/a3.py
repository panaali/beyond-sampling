import numpy as np, probe, sys
from lib import *
rng=np.random.default_rng(3)
x=rng.standard_normal((6000,10))
H=probe.hidden(x); F=lib(x); names=[n for n in F if F[n].min()>NEG]
for i in [int(a) for a in sys.argv[1:]]:
    out=greedy(H[:,i],F,names,steps=4)
    print(i," | ".join("%s %.3f"%(",".join(s),r) for s,r,c in out), "coef",np.round(out[-1][2],2),flush=True)
