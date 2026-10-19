import numpy as np
from feats import *
np.set_printoptions(precision=3,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
s8=-np.sort(-x[:,:9],1); s=-np.sort(-x,1)
B=np.column_stack([x[:,9],s8[:,0],s8[:,1],s8[:,2]])
B2=np.column_stack([s[:,0],s[:,1],s[:,2],s[:,3]])
for i in [6,31,19,18,22,26,5,12,9,3,24]:
    r,c=fit_relu(B,H[:,i],15); r2_,c2=fit_relu(B2,H[:,i],15)
    print(i,'[x9,M8,S8,T8] %.4f'%r,c,' | [s1..s4] %.4f'%r2_,c2)
