import numpy as np, probe, importlib, sys
mod=sys.argv[1] if len(sys.argv)>1 else 'labels_core'
L=importlib.import_module(mod)
x=np.random.default_rng(99).standard_normal((100000,10)); H=probe.hidden(x)
for i in range(8):
    p=L.f(i,x); y=H[:,i]; print(i,'R2',round(1-((y-p)**2).mean()/y.var(),4),'corr',round(np.corrcoef(p,y)[0,1],4))
