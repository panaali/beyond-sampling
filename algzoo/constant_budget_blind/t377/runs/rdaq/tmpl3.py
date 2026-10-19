import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
o9=np.argsort(-x[:,:9],1)
for i,j in [(27,0),(25,1),(29,1),(0,2),(28,3),(4,4),(13,5),(7,6),(24,7),(3,8),(10,8),(14,3)]:
    xj=x[:,j]; I1=(o9[:,0]==j)*1.; I2=(o9[:,1]==j)*1.
    so=x[:,:9].sum(1)-xj*(j<9)
    oth=np.delete(x[:,:9],j,1); s=-np.sort(-oth,1)
    out=[]
    for nm,X,g in [('Tb',[xj,so,I1,I2],None),('Tb9',[xj,so,x[:,9],I1+I2],None),('TM',[xj,s[:,1],I1,I2],None),
                   ('Gtop2',[xj,so,I1],(I1+I2)>0),('Gtop2S',[xj,so,s[:,1],I1],(I1+I2)>0),('Gnot2',[xj,so,s[:,1]],(I1+I2)==0)]:
        X=np.column_stack(X)
        r,c=(fit_relu(X,H[:,i]) if g is None else fit_gated(X,H[:,i],g))
        out.append('%s %.3f %s'%(nm,r,np.round(c,2)))
    print(i,j,' | '.join(out))
