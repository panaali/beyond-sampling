import numpy as np,sys; from lib2 import *
d=np.load('d1.npz'); x=d['x'][:8000].astype(float); h=d['h'][:8000]
F=library(x)
for i in map(int,sys.argv[1:]):
    o,c=greedy(F,h[:,i]); print(i,o,np.round(c,2),flush=True)
