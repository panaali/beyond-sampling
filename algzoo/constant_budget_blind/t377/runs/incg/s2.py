import numpy as np
d=np.load("d1.npz"); x=d["x"].astype(float); h=d["h"].astype(float)
N=len(x)
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
def fit(y,F):
    A=np.c_[F,np.ones(len(y))]; c=np.linalg.lstsq(A,y,rcond=None)[0]; return c, r2(y,A@c)
S=-np.sort(-x,1)
for k in [9,8,7,5]:
    others=np.delete(x,k,1); Mo=others.max(1); So=-np.sort(-others,1)
    f1=np.maximum(Mo-x[:,k],0); f2=np.maximum(So[:,1]-x[:,k],0)
    print("k",k)
    for i in range(32):
        if h[:,i].var()<1e-6: continue
        c1,a=fit(h[:,i],f1[:,None]); c2,b=fit(h[:,i],np.c_[f1,f2]); 
        if max(a,b)>0.5: print("  ",i,"R2 relu(Mo-xk) %.3f  +relu(So2-xk) %.3f"%(a,b),np.round(c2,2))
