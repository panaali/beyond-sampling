import numpy as np
x=np.load('x2.npy')[:60000];h=np.load('h2.npy')[:60000]
N=len(x); s=-np.sort(-x,1); M=s[:,0]
np.set_printoptions(precision=2,suppress=True,linewidth=220)
def fit(X,y):
    g=y>0.3
    for it in range(4):
        c,*_=np.linalg.lstsq(X[g],y[g],rcond=None); g=(y>0.3)|(X@c>0)
    pred=np.maximum(X@c,0); return 1-((y-pred)**2).mean()/y.var(),c
one=np.ones((N,1))
for i in [3,13,15]:
    for th in [0,0.5,1.0]:
        r,c=fit(np.c_[one,np.maximum(x-th,0),M],h[:,i]); print(i,'relu th',th,'R2 %.3f'%r,c)
    for th in [0.5,1.0]:
        r,c=fit(np.c_[one,np.maximum(x-th,0),x,M],h[:,i]); print(i,'relu+lin th',th,'R2 %.3f'%r,c)
    # relu(x_j - M*frac)
    for fr in [0.5,0.8]:
        r,c=fit(np.c_[one,np.maximum(x-fr*M[:,None],0),M],h[:,i]); print(i,'relu(x-%.1fM)'%fr,'R2 %.3f'%r,c)
