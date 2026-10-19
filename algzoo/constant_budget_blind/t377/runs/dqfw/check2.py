import numpy as np, probe, labels
def r2(y,p): return 1-((y-p)**2).mean()/y.var() if y.var()>0 else float('nan')
rng=np.random.default_rng(7)
for nm,x in [('unif',rng.uniform(-2,2,(30000,10))),('3x gauss',3*rng.standard_normal((30000,10)))]:
    h=probe.hidden(x); print(nm,' '.join('%d:%.2f'%(i,r2(h[:,i],labels.f(i,x))) for i in range(32)))
