import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); p=probe.predict(x)
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
print("acc",(p==a2).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print("mean",H.mean(0)); print("std",H.std(0)); print("frac>0",(H>0).mean(0))
# homogeneity
print("homog err",np.abs(probe.hidden(2*x[:1000])-2*H[:1000]).max())
# corr with x_t
C=np.array([[np.corrcoef(H[:,i],x[:,t])[0,1] if H[:,i].std()>0 else 0 for t in range(10)] for i in range(16)])
print("corr with x_t");print(C)
# mean by argmax position
M=np.array([[H[am==k,i].mean() for k in range(10)] for i in range(16)]);print("mean by argmax");print(M)
M=np.array([[H[a2==k,i].mean() for k in range(10)] for i in range(16)]);print("mean by arg2nd");print(M)
np.savez('/tmp/claude-0/-home-user-clean-neurons/aa1e53e7-191b-5474-9620-48e6f2b9b20d/scratchpad/t377/rtiv/d1.npz',x=x,H=H)
