import numpy as np
from featlib import *
np.set_printoptions(precision=2,suppress=True,linewidth=200)
x=np.load('x1.npy');H=np.load('H1.npy')
o=np.argsort(-x,1); a=o[:,0]; b=o[:,1]
for i in [11,17,30,10]:
    T=np.full((10,10),np.nan)
    for p in range(10):
        for q in range(10):
            m=(a==p)&(b==q)
            if m.sum()>30: T[p,q]=H[m,i].mean()
    print(i,'mean h by (argmax row, argsecond col) over all 10; total var %.2f'%H[:,i].var())
    print(T)
    # fraction of variance explained by this table
    pred=np.nan_to_num(T)[a,b]; print('  R2 of table %.3f'%(1-((H[:,i]-pred)**2).mean()/H[:,i].var()))
