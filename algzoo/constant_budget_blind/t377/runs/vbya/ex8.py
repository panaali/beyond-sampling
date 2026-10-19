import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy'); H=np.load('H1.npy')
P=np.maximum.accumulate(x,1); Q=np.maximum.accumulate(x[:,::-1],1)[:,::-1]
def fit(h,F,m):
    A=np.c_[F[m],np.ones(m.sum())]; c,*_=np.linalg.lstsq(A,h[m],rcond=None); r=h[m]-A@c; return c,1-r.var()/h[m].var()
for i,t in [(29,0),(21,1),(1,2),(11,3),(2,4),(23,6),(26,7)]:
    h=H[:,i]; act=h>1e-6
    before=P[:,t-1] if t>0 else np.zeros(len(h))
    after=Q[:,t+1]
    c,r=fit(h,np.c_[x[:,t],before,after],act); print(i,'t',t,'x_t,maxbefore,maxafter',c,'R2 %.3f'%r)
    c,r=fit(h,np.c_[x],act); print('     x0..9',c,'R2 %.3f'%r)
