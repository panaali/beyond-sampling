import numpy as np
x=np.load('x1.npy'); h=np.load('h1.npy')
M=np.maximum.accumulate(x,1)
np.set_printoptions(linewidth=200,suppress=True)
for i in range(16):
    F=np.c_[x,M,np.ones(len(x))]; c,*_=np.linalg.lstsq(F,h[:,i],rcond=None)
    print(i,'x:',c[:10].round(1),'\n   M:',c[10:20].round(1),'b',c[20].round(1))
