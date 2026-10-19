import numpy as np,sys
from relufit import relufit
d=np.load('d1.npz'); x,h=d['x'][:40000],d['h'][:40000]
srt=-np.sort(-x,1); s=np.argsort(-x,1)
cols=[]
for j in range(10):
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        cols.append((s[:,a]==j)*srt[:,b])
A=np.stack(cols,1)
np.set_printoptions(linewidth=200)
for i in map(int,sys.argv[1:]):
    co,R=relufit(A,h[:,i]); C=co.reshape(10,4).T
    print(i,'R2=%.3f'%R); print(' max*Q1',C[0].round(1)); print(' max*Q2',C[1].round(1)); print(' sec*Q1',C[2].round(1)); print(' sec*Q2',C[3].round(1))
