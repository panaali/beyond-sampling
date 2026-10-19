import numpy as np, probe
from feat import feats
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
o=np.argsort(-x,1)
print('acc',(p==o[:,1]).mean())
F=feats(x); names=list(F); X=np.stack([F[k] for k in names],1)
X1=np.c_[X,np.ones(len(x))]
for i in range(32):
    y=h[:,i]
    c=[np.corrcoef(X[:,j],y)[0,1] if y.std()>0 else 0 for j in range(X.shape[1])]
    c=np.nan_to_num(c); top=np.argsort(-np.abs(c))[:5]
    b=np.linalg.lstsq(X1,y,rcond=None)[0]; r2=1-((X1@b-y)**2).mean()/y.var()
    print(i,'mean%.2f act%.2f std%.2f linR2 %.2f'%(y.mean(),(y>0).mean(),y.std(),r2),' '.join('%s:%.2f'%(names[j],c[j]) for j in top))
