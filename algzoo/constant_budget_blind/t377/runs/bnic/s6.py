import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
M=np.maximum.accumulate(x,1); D=M-x; T=M[:,9:]-x
np.set_printoptions(linewidth=200,suppress=True)
def fit(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); e=y-F@c; return 1-e.var()/y.var(),c
for i in range(16):
    y=h[:,i]
    rD,cD=fit(D,y); rT,cT=fit(T,y); rDM,cDM=fit(np.c_[D,M[:,9]],y)
    print(i,'D %.3f'%rD,cD[:10].round(1),'| T %.3f'%rT,cT[:10].round(1),'| D+M9 %.3f'%rDM, cDM[10].round(1))
