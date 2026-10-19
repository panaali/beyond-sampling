import numpy as np,sys
from feats import *
from cat2 import catalog2
np.set_printoptions(precision=2,suppress=True)
x=np.load('x1.npy')[:4000];H=np.load('H1.npy')[:4000]
C=catalog2(x)
for i in map(int,sys.argv[1:]):
    for ch,r,c in greedy(C,H[:,i],k=4,it=8):
        print(i,ch,'%.3f'%r,c,flush=True)
