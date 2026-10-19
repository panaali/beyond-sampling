import numpy as np, probe
rng=np.random.default_rng(2)
x=rng.standard_normal((100000,10))
h=probe.hidden(x)
np.save('x2.npy',x); np.save('h2.npy',h)
P=np.maximum.accumulate(x,1)
Q=np.full_like(x,-9.)
for t in range(1,10):
    Q[:,t]=np.sort(x[:,:t+1],1)[:,-2]
X=np.c_[x,P,Q[:,1:],np.ones(len(x))]
names=['x%d'%t for t in range(10)]+['P%d'%t for t in range(10)]+['Q%d'%t for t in range(1,10)]+['1']
for i in range(32):
    y=h[:,i]; m=y>0.3
    b=np.linalg.lstsq(X[m],y[m],rcond=None)[0]
    pre=X@b; r2a=1-((pre[m]-y[m])**2).mean()/y[m].var()
    r2=1-((np.maximum(pre,0)-y)**2).mean()/y.var()
    top=np.argsort(-np.abs(b[:-1]))[:8]
    print(i,'R2act %.3f R2relu %.3f'%(r2a,r2),' '.join('%s:%.2f'%(names[j],b[j]) for j in top),'c:%.2f'%b[-1])
