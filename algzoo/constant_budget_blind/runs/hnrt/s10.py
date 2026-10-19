import numpy as np, probe
rng=np.random.default_rng(13); x=rng.standard_normal((200000,10)); H=probe.hidden(x)
o=np.argsort(-x,1); p1,p2=o[:,0],o[:,1]
np.save('x200.npy',x); np.save('H200.npy',H)
for i in [2,13,9,1,11,8,5]:
    T=np.zeros((10,10))
    for a in range(10):
        for b in range(10):
            m=(p1==a)&(p2==b); T[a,b]=H[m,i].mean() if m.any() else np.nan
    print("neuron",i,"rows=argmax pos, cols=second pos")
    print("\n".join(" ".join("%5.1f"%v if not np.isnan(v) else "   - " for v in r) for r in T))
