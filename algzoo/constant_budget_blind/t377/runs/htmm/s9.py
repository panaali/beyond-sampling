import numpy as np,sys
from lib import fit_relu
x=np.load('x2.npy'); h=np.load('h2.npy')
p=x[:,:9]; rk=np.argsort(np.argsort(-p,1),1); s=np.sort(p,1)[:,::-1]
def t2(a): 
    s=np.sort(a,1); return s[:,-1],s[:,-2]
for i,k in [(7,6),(0,2),(27,0),(4,4)]:
    y=h[:,i]
    A,A2=t2(x[:,:k]) if k>1 else ((x[:,0],np.full(len(x),-5.)) if k==1 else (np.full(len(x),-5.),)*2)
    B,B2=t2(x[:,k+1:9]) if k<7 else (x[:,8],np.full(len(x),-5.))
    for g in [0,1,2,3]:
        m=rk[:,k]==g
        X=np.c_[x[:,k],A,A2,B,B2,x[:,9]]
        b,r=fit_relu(X[m],y[m]); print(i,k,'rank',g,'[xk,A,A2,B,B2,x9,c]',np.round(b,2),'R2 %.3f'%r,'mean %.2f'%y[m].mean())
