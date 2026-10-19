import numpy as np, probe
rng=np.random.default_rng(1)
x=rng.standard_normal((100000,10))
H=probe.hidden(x); p=probe.predict(x)
np.save('x1.npy',x); np.save('H1.npy',H)
o=np.argsort(-x,1); am=o[:,0]; sm=o[:,1]
print('acc',(p==sm).mean())
np.set_printoptions(precision=2,suppress=True,linewidth=200)
print('mean',H.mean(0)); print('frac>0',(H>0).mean(0))
# corr with x_t
C=np.array([[np.corrcoef(H[:,i],x[:,t])[0,1] for t in range(10)] for i in range(16)])
print('corr x_t\n',C)
# mean by argmax pos and second pos
for i in range(16):
  print(i,'am',np.array([H[am==k,i].mean() for k in range(10)]),'sm',np.array([H[sm==k,i].mean() for k in range(10)]))
