import numpy as np
np.set_printoptions(suppress=True,linewidth=200,precision=2)
H=np.load('Hpre.npy'); x=np.load('xpre.npy')
B=np.c_[H[8],x[:,9]]
for i in range(8):
  y=H[9,:,i]; m=y>0.05
  c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  print(i,c, "resid",np.abs(B[m]@c-y[m]).max().round(4))
