import numpy as np; from lib2 import fitrelu,R
d=np.load('d1.npz'); x=d['x'][:40000].astype(float); h=d['h'][:40000]
y=x[:,:9]; s=np.sort(y,1)[:,::-1]; A,B,C=s[:,0],s[:,1],s[:,2]; x9=x[:,9]
np.set_printoptions(linewidth=200)
for i in [2,15,19,22,26,27,11,10,14,24,30,20,25,28,3,8,4]:
    res=[]
    for j in range(9):
        X=np.c_[x[:,j],A,B,C,x9]; r2,c=fitrelu(X,h[:,i],8); res.append((r2,j,c))
    res.sort(key=lambda t:-t[0]); r2,j,c=res[0]
    print(i,'best j',j,'R2 %.3f'%r2,'coef [xj,A,B,C,x9]',np.round(c,2),' 2nd j',res[1][1],'%.3f'%res[1][0])
