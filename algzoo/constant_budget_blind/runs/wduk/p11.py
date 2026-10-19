import numpy as np
from scipy.optimize import least_squares
x=np.load('x1.npy')[:30000];h=np.load('h1.npy')[:30000]
o=np.argsort(-x[:,:9],1);s=-np.sort(-x[:,:9],1)
np.set_printoptions(precision=1,suppress=True,linewidth=160)
E=np.eye(9)
XM=E[o[:,0]]*s[:,:1]; XS=E[o[:,1]]*s[:,1:2]; XT=E[o[:,2]]*s[:,2:3]
Xa=np.c_[XM,XS,x[:,9]]
Xb=np.c_[XM,XS,XT,x[:,9]]
# before/after: H indexed by (Q8, Q8<P8)
bef=(o[:,1]<o[:,0])[:,None]
Xc=np.c_[XM,XS*bef,XS*(~bef),x[:,9]]
for i in range(16):
    hi=h[:,i];out=[]
    for XX in (Xa,Xb,Xc):
        c,*_=np.linalg.lstsq(XX,hi,rcond=None)
        r=least_squares(lambda th:np.maximum(0,XX@th)-hi,c)
        out.append((1-((np.maximum(0,XX@r.x)-hi)**2).mean()/hi.var(),r.x))
    print(i,'R2 GH %.3f GHT %.3f GHbefaft %.3f'%(out[0][0],out[1][0],out[2][0]));print('  G',out[0][1][:9]);print('  H',out[0][1][9:18],'u %.1f'%out[0][1][18])
    print('  Hbef',out[2][1][9:18]);print('  Haft',out[2][1][18:27])
