import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); p=probe.predict(x)
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
print("acc",(p==a2).mean())
print("mean",h.mean(0).round(2)); print("frac>0",(h>0).mean(0).round(2)); print("std",h.std(0).round(2))
# corr with x_j
C=np.array([[np.corrcoef(h[:,i],x[:,j])[0,1] for j in range(10)] for i in range(8)])
print("corr x_j\n",C.round(2))
srt=-np.sort(-x,1)
feats={'max':srt[:,0],'sec':srt[:,1],'third':srt[:,2],'x9':x[:,9],'max-sec':srt[:,0]-srt[:,1]}
for i in range(8):
  print(i,{k:round(np.corrcoef(h[:,i],v)[0,1],2) for k,v in feats.items()})
# mean h by argmax and arg2
for i in range(8):
  print(i,"by am",np.array([h[am==k,i].mean() for k in range(10)]).round(2),"by a2",np.array([h[a2==k,i].mean() for k in range(10)]).round(2))
np.save('x.npy',x); np.save('h.npy',h)
