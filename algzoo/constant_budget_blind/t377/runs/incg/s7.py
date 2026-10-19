import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"].astype(float); h=d["h"].astype(float)
one=np.ones(len(x))
for i,k in [(2,4),(11,3),(1,2),(21,1),(23,6),(26,7),(29,0)]:
    y=h[:,i]
    pre=x[:,:k].max(1) if k>0 else np.full(len(x),-9.)
    post=x[:,k+1:].max(1)
    A=x[:,k]
    print("neuron",i,"k",k)
    for name,m in [("xk is global max",(A>pre)&(A>post)),("xk prefixmax, later bigger",(A>pre)&(A<post)),("xk not prefixmax, > later",(A<pre)&(A>post)),("xk not prefix max, <later",(A<pre)&(A<post))]:
        print("  %-28s frac %.2f mean %.2f act %.2f"%(name,m.mean(),y[m].mean(),(y[m]>1e-6).mean()))
    # among xk global max: regress on xk, second-largest, etc
    m=(A>pre)&(A>post)
    S=-np.sort(-x,1)
    for feats in [["xk","S1"],["xk","pre","post"],["xk","S1","pre"]]:
        D={"xk":A,"S1":S[:,1],"pre":np.maximum(pre,-3),"post":post}
        M=np.c_[[D[f][m] for f in feats]].T
        c,r=relufit(np.c_[M,one[m]],y[m]); print("   globalmax fit",feats,"%.3f"%r,np.round(c,2))
    m2=~m
    for feats in [["xk","S0","S1"],["xk","pre","post"]]:
        D={"xk":A,"S0":S[:,0],"S1":S[:,1],"pre":np.maximum(pre,-3),"post":post}
        M=np.c_[[D[f][m2] for f in feats]].T
        c,r=relufit(np.c_[M,one[m2]],y[m2]); print("   notmax fit",feats,"%.3f"%r,np.round(c,2))
