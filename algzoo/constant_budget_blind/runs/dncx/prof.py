import numpy as np; from lib2 import fitrelu
d=np.load('d1.npz'); x=d['x'][:40000].astype(float); h=d['h'][:40000]
y=x[:,:9]; s=np.sort(y,1)[:,::-1]; a=np.argsort(-y,1); A,B,C=s[:,0],s[:,1],s[:,2]; pA,pB=a[:,0],a[:,1]; x9=x[:,9]
X=np.c_[[A*(pA==j) for j in range(9)]+[B*(pB==j) for j in range(9)]+[x9,C]].T
np.set_printoptions(linewidth=200)
for i in range(32):
    r2,c=fitrelu(X,h[:,i],8)
    print(i,'R2 %.3f'%r2,'A:',np.round(c[:9],1),'B:',np.round(c[9:18],1),'x9 %.1f C %.1f'%(c[18],c[19]))
