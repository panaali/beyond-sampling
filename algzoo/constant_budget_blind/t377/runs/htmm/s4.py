import numpy as np,sys
from lib import fit_relu
x=np.load('x2.npy'); h=np.load('h2.npy')
def top2(a):
    if a.shape[1]==0: return np.full(len(a),-6.),np.full(len(a),-6.)
    if a.shape[1]==1: return a[:,0],np.full(len(a),-6.)
    s=np.sort(a,1); return s[:,-1],s[:,-2]
rk=np.argsort(np.argsort(-x,1),1)
for spec in sys.argv[1:]:
    i,k=map(int,spec.split(':'))
    y=h[:,i]
    A,A2=top2(x[:,:k]); B,B2=top2(x[:,k+1:])
    X=np.c_[x[:,k],A,A2,B,B2]
    b,r=fit_relu(X,y); print('n%d k%d fit[xk,A,A2,B,B2,c]'%(i,k),np.round(b,2),'R2 %.3f'%r)
    print('  mean by rank of xk:',' '.join('%d:%.2f/%.2f'%(q,y[rk[:,k]==q].mean(),(y[rk[:,k]==q]>0).mean()) for q in range(10)))
    for q in [0,1,2]:
        m=rk[:,k]==q; b,r=fit_relu(X[m],y[m]); print('  rank%d'%q,np.round(b,2),'R2 %.3f'%r)
