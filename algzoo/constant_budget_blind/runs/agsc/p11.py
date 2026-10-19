import numpy as np, probe
from feat import r2
from scipy.optimize import least_squares
np.set_printoptions(precision=3,suppress=True,linewidth=160)
rng=np.random.default_rng(10); N=30000
x=rng.standard_normal((N,10)); h=probe.hidden(x)
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); r=np.argsort(o,1)
M,S=s[:,0],s[:,1]; am,as_=o[:,0],o[:,1]
lag=9-np.arange(10)
MSlag=(M+S)*(9-np.minimum(am,as_))
oth=x*(r>=2)
def model(c,X):
    a,A,w,ph,b=c
    k=A*np.cos(w*lag+ph)
    return np.maximum(a*MSlag+X@k+b,0)
for i in [0,1,2,3,6,7]:
    y=h[:,i]; best=None
    for X,nm in [(x,'x'),(oth,'oth')]:
        for w0 in [0.3,0.6,0.9,1.2,1.6,2.0,2.5]:
            for ph0 in [0,1.5,3,4.5]:
                res=least_squares(lambda c: model(c,X)-y,[0.2,1,w0,ph0,1])
                rr=r2(y,model(res.x,X))
                if best is None or rr>best[0]: best=(rr,nm,res.x)
    print(i,round(best[0],3),best[1],best[2])
