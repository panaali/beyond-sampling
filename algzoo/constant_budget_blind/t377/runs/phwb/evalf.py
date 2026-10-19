import numpy as np, probe, labels
x=np.random.default_rng(424242).standard_normal((100000,10)); h=probe.hidden(x)
for i in range(16):
    p=labels.f(i,x); print(i,labels.LABELS[i][0],round(1-((p-h[:,i])**2).mean()/h[:,i].var(),3), len(labels.LABELS[i][1].split()))
