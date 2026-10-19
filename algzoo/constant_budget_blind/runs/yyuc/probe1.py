import numpy as np, probe
np.set_printoptions(precision=1,suppress=True,linewidth=220)
b=-1.0
X=np.full((10,10),b)
for p in range(10): X[p,p]=1.0
Hs=probe.hidden(X)
print('background -1, spike +1 at p (rows p=0..9), cols neurons')
print(np.round(Hs,1))
print('all background', probe.hidden(np.full((1,10),b)).round(1))
print('all zero but spike p, bg 0')
X=np.zeros((10,10))
for p in range(10): X[p,p]=1.0
print(probe.hidden(X).round(1))
