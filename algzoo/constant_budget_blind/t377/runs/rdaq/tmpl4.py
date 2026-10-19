import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
o9=np.argsort(-x[:,:9],1); am=o9[:,0]*1.; a2=o9[:,1]*1.
for i,j in [(27,0),(25,1),(0,2),(4,4),(13,5),(7,6),(29,1),(28,3),(24,7),(3,8),(10,8)]:
    xj=x[:,j]; I1=(o9[:,0]==j)*1.; I2=(o9[:,1]==j)*1.
    I2a=I2*(am>j); I2b=I2*(am<j)
    so=x[:,:9].sum(1)-xj*(j<9)
    oth=np.delete(x[:,:9],j,1); s=-np.sort(-oth,1)
    out=[]
    if i in (27,25,0,4,13,7):
        T=[('Pa',[xj,so,I1,I2],None),('Pb',[xj,so,I1,I2a],None),('Pc',[xj,so,I1,I2b],None),('Pd',[xj,s[:,1],I1,I2],None),('Pe',[xj,so,I1+I2,I2a],None),('Pf',[xj,so,I1,I2,am],None)]
    else:
        g=(I1+I2)==0
        T=[('N1',[xj,s[:,1]],g),('N2',[xj,s[:,1],s[:,0]],g),('N3',[xj,s[:,1],am],g),('N4',[xj,s[:,1],am,s[:,0]],g),('N5',[xj,s[:,1],so],g)]
    for nm,X,g in T:
        X=np.column_stack(X)
        r,c=(fit_relu(X,H[:,i]) if g is None else fit_gated(X,H[:,i],g))
        out.append('%s %.3f %s'%(nm,r,np.round(c,2)))
    print(i,j,' | '.join(out))
