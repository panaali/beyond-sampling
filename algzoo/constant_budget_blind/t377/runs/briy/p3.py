import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
M=np.maximum.accumulate(x,1)  # M[:,k]=max x0..xk
def r2(A,y):
    A=np.c_[A,np.ones(len(y))]; c,*_=np.linalg.lstsq(A,y,rcond=None); return 1-((A@c-y)**2).mean()/y.var(),c
np.set_printoptions(linewidth=200)
for i in range(16):
    y=h[:,i]; act=y>1e-6
    a,c=r2(np.c_[x,M[:,1:]][act],y[act])
    print(i,'act %.2f R2act %.3f'%(act.mean(),a)); print('  x',np.round(c[:10],1)); print('  M1..9',np.round(c[10:19],1),'b',round(c[-1],1))
