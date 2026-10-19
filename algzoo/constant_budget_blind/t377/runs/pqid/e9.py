import numpy as np
x=np.load('x2.npy')[:80000];h=np.load('h2.npy')[:80000]
N=len(x); R=np.maximum.accumulate(x,1)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
def fit(X,y):
    g=y>0.3
    for it in range(5):
        c,*_=np.linalg.lstsq(X[g],y[g],rcond=None); g=(y>0.3)|(X@c>0)
    pred=np.maximum(X@c,0); return 1-((y-pred)**2).mean()/y.var(),c
one=np.ones((N,1))
nov=np.maximum(x[:,1:]-R[:,:-1],0)
s=-np.sort(-x,1)
for i in [3,13,15,10]:
    r,c=fit(np.c_[one,R],h[:,i]); print(i,'R_t',' %.3f'%r,c)
    r,c=fit(np.c_[one,R,x],h[:,i]); print(i,'R_t+x',' %.3f'%r,c)
    r,c=fit(np.c_[one,R,x,s],h[:,i]); print(i,'R_t+x+sorted',' %.3f'%r)
    # running second max
    S2=np.array([np.sort(x[:,:t+1],1)[:,-2] for t in range(1,10)]).T
    r,c=fit(np.c_[one,R,S2],h[:,i]); print(i,'R_t+S2_t',' %.3f'%r,c)
