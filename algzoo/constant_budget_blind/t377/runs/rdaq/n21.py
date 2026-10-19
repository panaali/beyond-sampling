import numpy as np
from featlib import *
from feats import fit_relu, r2
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x1.npy')[:30000];H=np.load('H1.npy')[:30000]; y=H[:,21]
rk=np.argsort(np.argsort(-x[:,:9],1),1)
for K in [1,2,3]:
    low=x[:,:9]*(rk>=K)
    X=np.column_stack([low,-np.sort(-x[:,:9],1)[:,:K]])
    r,c=r2(X,y); print('K',K,'R2 %.3f'%r,c)
# clipped values
for t in [-1,-0.5,0,0.5]:
    X=np.column_stack([np.maximum(x[:,:9],t)]); print('max(x,%.1f)'%t,'%.3f'%r2(X,y)[0],np.round(r2(X,y)[1],2))
    X=np.column_stack([np.minimum(x[:,:9],t)]); print('min(x,%.1f)'%t,'%.3f'%r2(X,y)[0],np.round(r2(X,y)[1],2))
