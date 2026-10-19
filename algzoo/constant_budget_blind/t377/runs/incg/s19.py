import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"][:20000].astype(float); h=d["h"][:20000].astype(float)
xv=d["x"][20000:40000].astype(float); hv=d["h"][20000:40000].astype(float)
def RI(x):
    rec=np.stack([x[:,0]]+[np.maximum(x[:,t]-x[:,:t].max(1),0) for t in range(1,10)],1)
    p2=[np.sort(x[:,:t+1],1)[:,-2] for t in range(1,10)]
    inc2=np.stack([p2[0]]+[p2[t]-p2[t-1] for t in range(1,9)],1)
    return rec,inc2
rec,inc2=RI(x); recv,inc2v=RI(xv)
for i in [15,0,4,19]:
    best=(-9,)
    for g in np.linspace(0.5,1.1,13):
        for dl in np.linspace(0.5,1.1,13):
            wr=g**np.arange(10); wi=dl**np.arange(9)
            A=np.c_[rec@wr, inc2@wi]; c,r=relufit(A,h[:,i],iters=10)
            if r>best[0]: best=(r,g,dl,c)
    r,g,dl,c=best; wr=g**np.arange(10); wi=dl**np.arange(9)
    rv=r2(hv[:,i],np.maximum(np.c_[recv@wr,inc2v@wi]@c,0))
    print(i,"gamma %.2f delta %.2f"%(g,dl),np.round(c,2),"R2 train %.3f val %.3f"%(r,rv))
# neuron 10
pm7=x[:,:8].max(1); p27=np.sort(x[:,:8],1)[:,-2]
for nm,A in [("x8,pm7,p2nd7",np.c_[x[:,8],pm7,p27]),("x8,pm7,p2nd7,x9",np.c_[x[:,8],pm7,p27,x[:,9]]),("x8,pm7,p2nd7,rec7",np.c_[x[:,8],pm7,p27,np.maximum(x[:,7]-x[:,:7].max(1),0)])]:
    c,r=relufit(A,h[:,10],iters=30); print("n10",nm,"%.3f"%r,np.round(c,2))
