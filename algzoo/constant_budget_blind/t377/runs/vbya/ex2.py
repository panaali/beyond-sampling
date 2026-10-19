import numpy as np
x=np.load('x1.npy'); H=np.load('H1.npy')
np.set_printoptions(precision=2,suppress=True,linewidth=200)
P=np.maximum.accumulate(x,1)  # prefix max
am=np.argmax(x,1)
def ols(h,F,mask=None):
    if mask is None: mask=np.ones(len(h),bool)
    A=np.c_[F[mask],np.ones(mask.sum())]
    c,*_=np.linalg.lstsq(A,h[mask],rcond=None)
    r=h[mask]-A@c
    return c,1-r.var()/h[mask].var()
for i in [3,7,9,17,18,20,22,28,30,14,24,12,31]:
    h=H[:,i]; act=h>1e-6
    F=np.c_[x,P[:,8]]
    c,r2=ols(h,F,act)
    print(i,'act-OLS x0..x9,P8:',c,'R2 %.3f'%r2)
