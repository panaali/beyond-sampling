import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10)).astype(np.float32)
h=probe.hidden(x); h=np.asarray(h)
p=np.asarray(probe.predict(x))
srt=np.argsort(-x,1); sec=srt[:,1]
print("acc",(p==sec).mean())
np.savez("d1.npz",x=x,h=h)
S=-np.sort(-x,1)
am=srt[:,0]
for i in range(32):
    hi=h[:,i]
    cx=[np.corrcoef(hi,x[:,j])[0,1] for j in range(10)]
    cs=[np.corrcoef(hi,S[:,j])[0,1] for j in range(3)]
    # mean by argmax position
    mp=[hi[am==j].mean() for j in range(10)]
    ms=[hi[sec==j].mean() for j in range(10)]
    print(i,"mean %.2f act %.2f"%(hi.mean(),(hi>1e-6).mean()))
    print("  cx",np.round(cx,2)," cS",np.round(cs,2))
    print("  byMax",np.round(mp,2)); print("  bySec",np.round(ms,2))
