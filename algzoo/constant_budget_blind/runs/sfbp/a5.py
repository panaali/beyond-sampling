import numpy as np,sys
from flib import flib,greedy
d=np.load('d1.npz'); x,h=d['x'][:20000],d['h'][:20000]
F=flib(x)
for i in map(int,sys.argv[1:]):
    s,_=greedy(F,h[:,i],4,h[:,i]>1e-6); print(i,s)
