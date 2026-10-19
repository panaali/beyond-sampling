import numpy as np
np.set_printoptions(suppress=True,linewidth=200,precision=2)
H=np.load('Hpre.npy'); x=np.load('xpre.npy'); N=len(x)
for i in [3,4,2]:
  print("neuron",i)
  for L in [6,8,10]:
    y=H[L-1,:,i]; xx=x[:,:L]; M=np.maximum.accumulate(xx,1)
    B=np.c_[xx[:,::-1],M[:,::-1]]  # lag order: x_t, x_{t-1},...; M_t, M_{t-1},...
    m=y>0.3
    c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
    print(L,"x lags",c[:L],"\n   M lags",c[L:])
