import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
M9=x[:,:9].max(1); M10=x.max(1)
for i,j in [(27,0),(25,1),(29,1),(0,2),(28,3),(4,4),(13,5),(7,6),(24,7),(3,8),(16,4),(23,7),(14,3),(21,4)]:
    xj=x[:,j]; so=x.sum(1)-xj; sb=x[:,:j].sum(1); sa=x[:,j+1:].sum(1)
    Mb=x[:,:j].max(1) if j>0 else np.full(len(x),-5.)
    Ma=x[:,j+1:9].max(1) if j<8 else np.full(len(x),-5.)
    Mo=np.delete(x,j,1).max(1)
    res=[]
    res.append(('A',)+fit_relu(np.column_stack([xj,so]),H[:,i]))
    res.append(('Bsplit',)+fit_relu(np.column_stack([xj,sb,sa,x[:,9]]),H[:,i]))
    res.append(('C Mo',)+fit_relu(np.column_stack([xj,so,Mo]),H[:,i]))
    res.append(('D MbMa',)+fit_relu(np.column_stack([xj,so,Mb,Ma]),H[:,i]))
    g=xj>=M9
    res.append(('Gmax9',)+fit_gated(np.column_stack([xj,so]),H[:,i],g))
    g2=xj>=Mb
    res.append(('Grec',)+fit_gated(np.column_stack([xj,so]),H[:,i],g2))
    print(i,j,' | '.join('%s %.3f %s'%(n,r,np.round(c,2)) for n,r,c in res))
