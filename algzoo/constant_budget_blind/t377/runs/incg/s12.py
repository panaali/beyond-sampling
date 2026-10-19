import numpy as np
d=np.load("d1.npz"); x=d["x"].astype(float); h=d["h"].astype(float)
o=np.argsort(-x,1)
for i in [16,25]:
    y=h[:,i]; a=y>1e-4
    print(i,"actfrac",a.mean(),"max",y.max(),"var",y.var())
    print("  argmax among active",np.bincount(o[a,0],minlength=10),"second",np.bincount(o[a,1],minlength=10))
    print("  mean x among active",np.round(x[a].mean(0),2))
    top=np.argsort(-y)[:5]
    for t in top: print("   ",np.round(y[t],2),np.round(x[t],2))
