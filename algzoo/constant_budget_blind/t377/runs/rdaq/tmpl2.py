import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:20000];H=np.load('H1.npy')[:20000]
for i,j in [(27,0),(25,1),(29,1),(0,2),(28,3),(4,4),(13,5),(7,6),(24,7),(3,8),(10,8),(11,8),(14,3),(17,6)]:
    xj=x[:,j]
    for tag,lim in [('all',10),('0..8',9)]:
        oth=np.delete(x[:,:lim],j,1) if j<lim else x[:,:lim]
        so=-np.sort(-oth,1); Mo,So,To=so[:,0],so[:,1],so[:,2]; sm=oth.sum(1)
        out=[]
        for nm,X in [('P1',[xj,So]),('P2',[xj,Mo,So]),('P2T',[xj,Mo,So,To]),('P3',[xj,Mo,So,sm])]:
            r,c=fit_relu(np.column_stack(X),H[:,i]); out.append('%s %.3f %s'%(nm,r,np.round(c,2)))
        print(i,j,tag,' | '.join(out))
