import numpy as np, probe
rng=np.random.default_rng(5)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
np.savez('d1.npz',x=x,h=h,p=p)
s=np.argsort(-x,1); sec=s[:,1]
print('acc',(p==sec).mean())
print('mean',h.mean(0).round(2))
print('fraczero',(h<=1e-9).mean(0).round(2))
# linear regression on x
X=np.c_[x,np.ones(len(x))]
for i in range(32):
    c,_,_,_=np.linalg.lstsq(X,h[:,i],rcond=None)
    r2=1-((X@c-h[:,i])**2).mean()/h[:,i].var()
    print(i,round(r2,2),c.round(2))
