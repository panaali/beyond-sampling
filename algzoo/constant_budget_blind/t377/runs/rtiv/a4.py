import numpy as np
d=np.load('d1.npz');x=d['x'][:40000];H=d['H'][:40000]
pm=np.maximum.accumulate(x,1)
for i in range(16):
    y=H[:,i];best=[]
    for t in range(10):
        for u in range(10):
            f=pm[:,u]-x[:,t]
            if u<t: f=np.maximum(f,0)
            a=(f@y)/(f@f);r=1-((y-a*f)**2).mean()/y.var();best.append((r,t,u,a))
    best.sort(reverse=True);print(i,[(round(b[0],3),b[1],b[2],round(b[3],1)) for b in best[:3]])
