import numpy as np
from rfit import relufit, r2
d=np.load("d1.npz"); x=d["x"].astype(float); h=d["h"].astype(float)
S=-np.sort(-x,1); o=np.argsort(-x,1); am=o[:,0]; sp=o[:,1]
for i in [0,15,4,5,6,12,19,31,10,29]:
    y=h[:,i]; print("neuron",i, "act%.2f"%(y>0).mean())
    rows=[]
    for p in range(10):
        m=am==p
        c,r=relufit(np.c_[S[m,0],S[m,1],S[m,2]],y[m],iters=15)
        rows.append("p%d:[%s]R%.2f"%(p,",".join("%.1f"%v for v in c),r))
    print("  by argmax, fit relu(a S0+b S1+c S2):"," ".join(rows))
