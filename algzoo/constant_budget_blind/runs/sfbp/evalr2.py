import numpy as np, probe, labels
x=np.random.default_rng(13).standard_normal((50000,10)); h=probe.hidden(x)
R=[]
for i in range(32):
    p=labels.f(i,x); R.append(1-((p-h[:,i])**2).mean()/h[:,i].var())
print(' '.join(f'{i}:{r:.3f}' for i,r in enumerate(R)))
print('mean R2 %.3f median %.3f; var-weighted %.3f'%(np.mean(R),np.median(R),1-sum(((labels.f(i,x)-h[:,i])**2).mean() for i in range(32))/h.var(0).sum()))
