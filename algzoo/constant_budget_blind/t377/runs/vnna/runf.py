import numpy as np, probe, sys, json
from fx import *
x=np.random.default_rng(501).standard_normal((40000,10)); H=probe.hidden(x); d=ns(x)
xt=np.random.default_rng(502).standard_normal((40000,10)); Ht=probe.hidden(xt); dt=ns(xt)
out={}
for line in open(sys.argv[1]):
    line=line.strip()
    if not line or line.startswith('#'): continue
    i,n,e=line.split(';',2); i=int(i); n=int(n)
    p,r2,r2t=fit(e,n,H[:,i],d,ht=Ht[:,i],dt=dt)
    out[i]=dict(n=n,e=e,p=[float(v) for v in p],r2=r2t)
    print("%2d test R2 %.3f  p=%s"%(i,r2t,np.round(p,3)),flush=True)
json.dump(out,open(sys.argv[2],'w'),indent=1)
