import numpy as np
np.set_printoptions(precision=2,suppress=True,linewidth=220)
x=np.load('x1.npy'); H=np.load('H1.npy')
def fit(h,F,m):
    A=np.c_[F[m],np.ones(m.sum())]; c,*_=np.linalg.lstsq(A,h[m],rcond=None); r=h[m]-A@c; return c,1-r.var()/h[m].var()
for i,t in [(29,0),(21,1),(1,2),(11,3),(2,4),(23,6),(26,7),(14,8),(24,7),(12,5),(31,8),(27,8),(10,8),(13,9),(8,9)]:
    h=H[:,i]; act=h>1e-6
    oth=np.delete(x,t,1); so=-np.sort(-oth,1)
    bef=x[:,:t]; aft=x[:,t+1:]
    mb=bef.max(1) if t>0 else np.full(len(h),-9.); ma=aft.max(1) if t<9 else np.full(len(h),-9.)
    c,r=fit(h,np.c_[x[:,t],so[:,0],so[:,1],so[:,2]],act)
    c2,r2=fit(h,np.c_[x[:,t],np.maximum(mb,-4),np.maximum(ma,-4)],act)
    print(i,t,'x_t,o1,o2,o3',c,'R2 %.3f'%r,'| x_t,maxbef,maxaft',c2,'R2 %.3f'%r2)
