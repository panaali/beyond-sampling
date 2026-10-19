import numpy as np, probe
from feat import *
np.set_printoptions(precision=2,suppress=True,linewidth=150)
rng=np.random.default_rng(2)
x=rng.standard_normal((200000,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1)
g=o[:,0]*10+o[:,1]
for i in range(8):
    y=h[:,i]; p=np.zeros_like(y); p2=np.zeros_like(y); p3=np.zeros_like(y)
    for k in np.unique(g):
        m=g==k
        p[m]=y[m].mean()
        A=np.c_[s[m,0],s[m,1],np.ones(m.sum())]
        c=np.linalg.lstsq(A,y[m],rcond=None)[0]; p2[m]=A@c
        A=np.c_[s[m,0],s[m,1],x[m],np.ones(m.sum())]
        c=np.linalg.lstsq(A,y[m],rcond=None)[0]; p3[m]=A@c
    print(i,'grpmean',round(r2(y,p),3),'grp+MS',round(r2(y,p2),3),'grp+MS+x',round(r2(y,p3),3))
print('pred acc', (probe.predict(x)==o[:,1]).mean())
