import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
s=np.sort(x,1)[:,::-1]
def r2(A,y):
    A=np.c_[A,np.ones(len(y))]; c,*_=np.linalg.lstsq(A,y,rcond=None); return 1-((A@c-y)**2).mean()/y.var(),c
for i in range(16):
    y=h[:,i]
    a,_=r2(x,y); b,_=r2(np.c_[x,s[:,0]],y); c,_=r2(np.c_[x,s[:,:3]],y)
    d,cc=r2(np.c_[x,s[:,:2]],y)
    print(i,'lin x %.3f  +max %.3f  +top3 %.3f'%(a,b,c),'coef',np.round(cc[:12],1))
