import numpy as np, probe, labels
x=np.random.default_rng(21).standard_normal((100000,10)); H=probe.hidden(x)
for i in range(16):
    p=labels.f(i,x); h=H[:,i]
    print(i, labels.LABELS[i][0], 'R2=%.3f'%(1-((p-h)**2).mean()/h.var()), 'corr=%.3f'%np.corrcoef(p,h)[0,1])
a=x.argmax(1); print("N11 when x7 max: mean",H[a==7,11].mean(), " N7 min",H[:,7].min(), " frac N2>0 when x2 top2", (H[np.isin(np.argsort(-x,1)[:,0:2],2).any(1),2]>0).mean())
