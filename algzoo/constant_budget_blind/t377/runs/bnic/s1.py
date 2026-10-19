import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
h=probe.hidden(x); np.save('x1.npy',x); np.save('h1.npy',h)
p=probe.predict(x)
s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
print('acc',(p==a2).mean())
print('mean',h.mean(0).round(2)); print('frac>0',(h>0).mean(0).round(2)); print('std',h.std(0).round(2))
# correlations with x_j
C=np.array([[np.corrcoef(h[:,i],x[:,j])[0,1] if h[:,i].std()>0 else 0 for j in range(10)] for i in range(16)])
print('corr with x_j'); print(C.round(2))
# mean of h by argmax position and by second position
for name,a in [('argmax',am),('arg2',a2)]:
  M=np.array([[h[a==k,i].mean() for k in range(10)] for i in range(16)])
  print('mean by',name); print(M.round(2))
