import numpy as np
np.set_printoptions(precision=1,suppress=True,linewidth=200)
x=np.load('x4.npy'); G=np.load('G4.npy'); pre=x[:,:9]
o=np.argsort(-pre,1); am=o[:,0]; a2=o[:,1]; s=-np.sort(-pre,1)
for i,S,d in [(3,[1,4,7],1.1),(17,[0,1,2,3],1.0),(20,[0,3,4],1.0)]:
    b=np.zeros(9); b[S]=d; m=(pre+b).max(1)
    g=G[:,i]; A=np.c_[m,np.ones(len(g))]; c,*_=np.linalg.lstsq(A,g,rcond=None); r=g-A@c
    T=np.array([[r[(am==u)&(a2==v)].mean() if ((am==u)&(a2==v)).sum()>20 else np.nan for v in range(9)] for u in range(9)])
    print(i,'resid by am(row) a2(col)\n',T)
    print('corr resid with s2, gap', np.corrcoef(r,s[:,1])[0,1], np.corrcoef(r,s[:,0]-s[:,1])[0,1])
