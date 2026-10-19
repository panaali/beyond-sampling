import numpy as np,sys
from lib import fit_relu
x=np.load('x2.npy')[:50000]; h=np.load('h2.npy')[:50000]
p=x[:,:9]; o=np.argsort(-p,1); s=np.sort(p,1)[:,::-1]
am=o[:,0]; a2=o[:,1]
X=np.c_[x[:,9],s[:,0],s[:,1],s[:,2]]+0
OH=np.c_[[(am==k)*1. for k in range(1,9)]].T
OH2=np.c_[[(a2==k)*1. for k in range(1,9)]].T
XX=np.c_[X,OH,OH2]
np.set_printoptions(linewidth=250)
for i in range(32):
    b,r=fit_relu(XX,h[:,i],it=12)
    print('%2d R2 %.3f x9 %5.2f P %5.2f Q %5.2f R %5.2f |am1-8'%(i,r,*b[:4]),' '.join('%5.2f'%v for v in b[4:12]),'|a2 1-8',' '.join('%5.2f'%v for v in b[12:20]),'c %.2f'%b[-1])
