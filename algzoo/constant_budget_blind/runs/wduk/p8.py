import numpy as np
from scipy.optimize import least_squares
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
o=np.argsort(-x,1);s=-np.sort(-x,1)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
N=len(x);E=np.eye(10)
XM=E[o[:,0]]*s[:,:1]; XS=E[o[:,1]]*s[:,1:2]; XT=E[o[:,2]]*s[:,2:3]
X=np.c_[XM,XS,x[:,9]]
X3=np.c_[XM,XS,XT,x[:,9]]
for i in range(16):
    hi=h[:,i]
    out=[]
    for XX in (X,X3):
        c,*_=np.linalg.lstsq(XX,hi,rcond=None)
        r=least_squares(lambda th:np.maximum(0,XX@th)-hi,c)
        out.append((1-((np.maximum(0,XX@r.x)-hi)**2).mean()/hi.var(),r.x))
    print(i,'R2 MS %.3f  MST %.3f'%(out[0][0],out[1][0]));print('  G',out[0][1][:10]);print('  H',out[0][1][10:20],'x9 %.1f'%out[0][1][20])
