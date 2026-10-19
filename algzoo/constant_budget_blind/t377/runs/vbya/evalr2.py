import numpy as np, probe, labels
rng=np.random.default_rng(424242)
x=rng.standard_normal((100000,10)); H=probe.hidden(x)
out=[]
for i in range(32):
    p=labels.f(i,x); h=H[:,i]
    v=h.var()
    r2=1-((h-p)**2).mean()/v if v>1e-12 else float('nan')
    out.append(r2); print(i,'R2 %.3f  var %.3g  mean %.3g'%(r2,v,h.mean()))
