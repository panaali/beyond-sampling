import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
rk=np.argsort(np.argsort(-x,1),1)  # rank of each position (0=max)
print("mean h by rank of x_k")
for i,k in [(1,2),(2,4),(11,3),(21,1),(23,6),(26,7),(29,0),(10,8),(31,8),(5,0),(12,5),(24,7),(14,8)]:
    print(i,k,[round(h[rk[:,k]==r,i].mean(),2) for r in range(10)])
# X9 template search
def below(k): return np.maximum(np.delete(x,k,1).max(1)-x[:,k],0)
def qgap(Q,R):
    rest=[j for j in R if j not in Q]
    if not Q or not rest: return np.zeros(len(x))
    return np.maximum(x[:,Q].max(1)-x[:,rest].max(1),0)
for i,k in [(3,9),(7,9),(9,9),(17,9),(18,9),(20,9),(22,9),(28,9),(30,9),(14,8),(24,7),(12,5),(31,8)]:
    y=h[:,i]; R=[j for j in range(10) if j!=k]; b=below(k)
    def sc(Q):
        return relufit(np.c_[b,qgap(Q,R)],y,iters=10)
    Q=[]; c,r=sc(Q); imp=True
    while imp:
        imp=False
        for j in R:
            Q2=sorted(set(Q)^{j}); c2,r2_=sc(Q2)
            if r2_>r+1e-4: Q,c,r=Q2,c2,r2_; imp=True
    # add S1-ish third feature?
    c3,r3=relufit(np.c_[b,qgap(Q,R),np.delete(x,k,1).max(1)],y,iters=10)
    print("X9tmpl",i,"k",k,"Q",Q,"R2 %.3f"%r,np.round(c,2)," +mex: %.3f"%r3,np.round(c3,2))
