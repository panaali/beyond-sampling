import numpy as np
from feats import *
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy')[:30000];H=np.load('H1.npy')[:30000]
s9=-np.sort(-x[:,:9],1); o=np.argsort(-x[:,:9],1)
A=np.eye(9)[o[:,0]]; B=np.eye(9)[o[:,1]]
for i in [1,2,8,15,20,5,12,26,6,18,19,31,0,4,7,13,25,27,28,29,24,3,10,11,14,17,21,30]:
    X=np.column_stack([s9[:,0],s9[:,1],x[:,9],A[:,1:],B[:,1:]])
    r,c=fit_relu(X,H[:,i],15)
    X2=np.column_stack([s9[:,0],s9[:,1],s9[:,2],x[:,9],A[:,1:]])
    r2_,c2=fit_relu(X2,H[:,i],15)
    print(i,'R2 am+a2 %.3f am-only %.3f'%(r,r2_),'M,S,x9',c[:3],'b',c[-1].round(2),'\n   offs am',np.r_[0,c[3:11]],'\n   offs a2',np.r_[0,c[11:19]])
