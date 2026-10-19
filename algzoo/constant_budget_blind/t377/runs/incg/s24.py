import numpy as np, probe, labels
rng=np.random.default_rng(777); x=rng.standard_normal((100000,10))
h=np.asarray(probe.hidden(x.astype(np.float32))).astype(float)
for i in range(32):
    p=labels.f(i,x); y=h[:,i]
    print(i,labels.LABELS[i][0],labels.CONSTANTS[i],"R2 %.3f"%(1-((y-p)**2).mean()/y.var()),"var %.3g"%y.var())
