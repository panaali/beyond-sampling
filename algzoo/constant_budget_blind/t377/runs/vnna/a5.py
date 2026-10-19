import numpy as np, probe
rng=np.random.default_rng(5)
x=rng.standard_normal((100000,10))
H=probe.hidden(x)
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]; a3=s[:,2]
srt=-np.sort(-x,1)
cell=am*10+a2
pm9=x[:,:9].max(1); am9=x[:,:9].argmax(1)
for i in range(32):
    h=H[:,i]
    mu=np.zeros(100); 
    for c in range(100):
        m=cell==c
        if m.any(): mu[c]=h[m].mean()
    r_cell=1-((h-mu[cell])**2).mean()/h.var()
    # cell + linear in top3 values within cell (global slopes)
    X=np.c_[np.eye(100)[cell],srt[:,:3]]
    c,*_=np.linalg.lstsq(X,h,rcond=None); r_cl=1-((h-X@c)**2).mean()/h.var()
    mu1=np.array([h[am==k].mean() for k in range(10)]); r_am=1-((h-mu1[am])**2).mean()/h.var()
    print(i,"R2 am %.2f  am,a2 %.2f  +top3lin %.2f"%(r_am,r_cell,r_cl), "slopes",np.round(c[100:],2))
