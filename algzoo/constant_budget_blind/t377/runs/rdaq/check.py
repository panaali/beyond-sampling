import numpy as np, probe, labels
x=np.random.default_rng(5).standard_normal((50000,10)); H=np.asarray(probe.hidden(x))
r=[]
for i in range(32):
    p=labels.f(i,x); y=H[:,i]; r.append(1-((y-p)**2).mean()/y.var())
print(' '.join('%d:%.3f'%(i,v) for i,v in enumerate(r))); print('mean %.3f median %.3f'%(np.mean(r),np.median(r)))
