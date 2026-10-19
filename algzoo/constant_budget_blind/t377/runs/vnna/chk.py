import numpy as np, probe, labels
x=np.random.default_rng(777).standard_normal((50000,10)); H=probe.hidden(x)
r=[]
for i in range(32):
    p=labels.f(i,x); r.append(1-((p-H[:,i])**2).mean()/H[:,i].var())
print(" ".join("%d:%.3f"%(i,v) for i,v in enumerate(r))); print("mean %.3f median %.3f"%(np.mean(r),np.median(r)))
# readout sanity: does the formula-hidden predict argmax2 via linear readout fitted on true H? (just correlation check)
print(max(labels.CONSTANTS.values()))
