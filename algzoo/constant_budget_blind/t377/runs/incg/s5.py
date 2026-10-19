import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
one=np.ones(len(x))
NEG=-1e9
def gmax(S):
    if len(S)==0: return np.full(len(x),-3.0)
    return x[:,sorted(S)].max(1)
def fitP(y,k,P):
    others=[j for j in range(10) if j!=k]; Q=[j for j in others if j not in P]
    A=np.c_[gmax(P),gmax(Q),x[:,k],one]; return relufit(A,y,iters=12)
for i in range(32):
    y=h[:,i]
    if y.var()<1e-3: print(i,"dead");continue
    cors=[abs(np.corrcoef(y,x[:,j])[0,1]) for j in range(10)]
    ks=list(np.argsort(cors)[::-1][:2])
    best=(-9,)
    for k in ks:
        others=[j for j in range(10) if j!=k]
        # init P from mean by argmax
        am=np.argmax(np.delete(x,k,1),1); mm=[y[am==m].mean() for m in range(9)]
        P=set(others[m] for m in range(9) if mm[m]>np.median(mm))
        c,r=fitP(y,k,P)
        improved=True
        while improved:
            improved=False
            for j in others:
                P2=set(P)^{j}; c2,r2_=fitP(y,k,P2)
                if r2_>r+1e-4: P,c,r=P2,c2,r2_; improved=True
        if r>best[0]: best=(r,k,sorted(P),c)
    print(i,"R2 %.3f k=%d P=%s"%(best[0],best[1],best[2]),"a,b,c,d=",np.round(best[3],2),flush=True)
