import numpy as np,sys
from relufit import relufit
d=np.load('d1.npz'); x,h=d['x'][:40000],d['h'][:40000]
p=x[:,:9]; Q=-np.sort(-p,1); s=np.argsort(-p,1)
cols=[]
for j in range(9):
    for a,b in [(0,0),(0,1),(1,0),(1,1)]:
        cols.append((s[:,a]==j)*Q[:,b])
A=np.c_[np.stack(cols,1),Q[:,2],x[:,9]]
np.set_printoptions(linewidth=200)
for i in map(int,sys.argv[1:]):
    co,R=relufit(A,h[:,i]); C=co[:36].reshape(9,4).T
    print(i,'R2=%.3f'%R,'Q3 %.2f x9 %.2f'%(co[36],co[37]))
    for nm,row in zip(['am*Q1','am*Q2','as*Q1','as*Q2'],C): print('  ',nm,row.round(1))
