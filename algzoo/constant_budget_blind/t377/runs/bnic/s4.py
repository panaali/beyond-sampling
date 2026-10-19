import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
N=len(x); M=np.maximum.accumulate(x,1); s=np.argsort(-x,1); am=s[:,0]; a2=s[:,1]
srt=-np.sort(-x,1)
I1=np.eye(10)[am]; I2=np.eye(10)[a2]
def r2(F,y):
    F=np.c_[F,np.ones(len(y))]; c,*_=np.linalg.lstsq(F,y,rcond=None); e=y-F@c; return 1-e.var()/y.var()
base=np.c_[srt[:,0],srt[:,1],x[:,9]]
for i in range(16):
    y=h[:,i]
    print(i, 'vals %.3f'%r2(base,y),'I1 %.3f'%r2(I1,y),'I2 %.3f'%r2(I2,y),'I1I2 %.3f'%r2(np.c_[I1,I2],y),'all %.3f'%r2(np.c_[base,I1,I2,I1*srt[:,:1],I2*srt[:,1:2]],y),
      'x+M+I %.3f'%r2(np.c_[x,M,I1,I2],y))
