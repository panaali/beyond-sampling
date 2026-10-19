import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((50000,10))
H=probe.hidden(x); p=probe.predict(x)
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
print("acc",(p==a2).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
srt=-np.sort(-x,1)
for i in range(32):
    h=H[:,i]
    fr=(h>0).mean()
    # mean by argmax pos, by 2nd pos
    m1=[h[am==k].mean() for k in range(10)]
    m2=[h[a2==k].mean() for k in range(10)]
    cx=[np.corrcoef(h,x[:,k])[0,1] for k in range(10)]
    print(i,"fr%.2f mean%.2f"%(fr,h.mean()))
    print("  am",np.array(m1)); print("  a2",np.array(m2)); print("  cx",np.array(cx))
