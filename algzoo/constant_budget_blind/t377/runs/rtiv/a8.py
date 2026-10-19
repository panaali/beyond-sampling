import numpy as np
d=np.load('d1.npz');x=d['x'];H=d['H']
o=np.argsort(-x,1);am=o[:,0]
pm=np.maximum.accumulate(x,1)
# neuron 8: h vs pm_u - x7 by argmax position
for i,t in [(8,7),(3,9),(6,9),(12,9),(14,4)]:
    print("neuron",i,"target x%d"%t)
    for k in range(10):
        m=am==k; y=H[m,i]
        f=np.maximum(x[m].max(1)-x[m,t],0)
        a=(f@y)/max(f@f,1e-9); r=1-((y-a*f)**2).mean()/max(y.var(),1e-9)
        print(f"  am={k} n={m.sum()} mean h={y.mean():.1f}  fit c*(M-x{t}): c={a:.1f} r2={r:.3f}")
