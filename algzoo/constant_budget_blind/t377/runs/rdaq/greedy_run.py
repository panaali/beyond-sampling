import numpy as np,sys
from feats import *
np.set_printoptions(precision=2,suppress=True)
x=np.load('x1.npy')[:5000];H=np.load('H1.npy')[:5000]
C=catalog(x)
for i in map(int,sys.argv[1:]):
    for ch,r,c in greedy(C,H[:,i]):
        pass
        print(i,ch,'%.3f'%r,c,flush=True)
