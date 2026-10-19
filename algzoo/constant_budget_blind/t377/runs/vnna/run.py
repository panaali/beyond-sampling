import numpy as np, probe, sys
from fx import *
rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 21)
x=rng.standard_normal((20000,10)); H=probe.hidden(x); d=ns(x)
xt=np.random.default_rng(77).standard_normal((20000,10)); Ht=probe.hidden(xt); dt=ns(xt)
for line in open('cands.txt'):
    line=line.strip()
    if not line or line.startswith('#'): continue
    i,n,e=line.split(';',2); i=int(i); n=int(n)
    p,r2,r2t=fit(e,n,H[:,i],d,ht=Ht[:,i],dt=dt)
    print("%2d test R2 %.3f  p=%s  %s"%(i,r2t,np.round(p,3),e),flush=True)
