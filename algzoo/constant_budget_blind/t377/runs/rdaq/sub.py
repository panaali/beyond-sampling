import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy');H=np.load('H1.npy')
o9=np.argsort(-x[:,:9],1)
s9=-np.sort(-x[:,:9],1)
for i,j in [(4,4),(0,2),(27,0),(25,1),(29,1),(28,3),(10,8),(14,3)]:
    for nm,m in [('top1',o9[:,0]==j),('top2',o9[:,1]==j),('rest',(o9[:,0]!=j)&(o9[:,1]!=j))]:
        y=H[m,i]; X=np.column_stack([x[m],s9[m,:3]])
        act=(y>1e-6).mean()
        r,c=fit_relu(X,y,15)
        print(i,j,nm,'act %.2f R2 %.3f'%(act,r),'x:',c[:10],'s1-3:',c[10:13],'b %.2f'%c[-1])
