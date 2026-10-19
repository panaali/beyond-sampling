import numpy as np,sys
x=np.load('x1.npy'); h=np.load('h1.npy')
o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
np.set_printoptions(linewidth=200,suppress=True,precision=1)
for i in map(int,sys.argv[1:]):
  T=np.zeros((10,10))
  for a in range(10):
    for b in range(10):
      m=(am==a)&(a2==b); T[a,b]=h[m,i].mean() if m.any() else np.nan
  print('neuron',i,'rows am, cols a2'); print(T)
