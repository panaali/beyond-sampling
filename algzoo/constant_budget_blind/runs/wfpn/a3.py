import numpy as np, probe
np.set_printoptions(suppress=True,linewidth=150)
for s in [1,-1]:
  X=np.zeros((10,10))
  for k in range(10): X[k,k]=s
  print("impulse",s,"at pos k (rows k=0..9)"); print(probe.hidden(X).round(2))
x=np.load('x.npy')[:5]
print(np.c_[probe.hidden(x),probe.hidden(2*x)/2].round(2))
