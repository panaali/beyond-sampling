import numpy as np
np.set_printoptions(suppress=True,linewidth=160)
x=np.load('x.npy'); h=np.load('h.npy')
M=np.maximum.accumulate(x,1); G=M-x
mn=np.minimum.accumulate(x,1)
def ev(B,i):
  m=h[:,i]>0
  c,*_=np.linalg.lstsq(B[m],h[m,i],rcond=None)
  for it in range(3):
    pr=B@c; mm=(pr>0)|m; c,*_=np.linalg.lstsq(B[mm],h[mm,i],rcond=None)
  pr=np.maximum(B@c,0); return 1-((pr-h[:,i])**2).mean()/h[:,i].var(), c
for i in range(8):
  r1,c1=ev(G[:,1:],i); r2,c2=ev(np.c_[x,M[:,1:]],i); r3,c3=ev(np.c_[G[:,1:],M[:,9],x[:,9]],i)
  print(i,round(r1,3),c1.round(2),"| x,M:",round(r2,3),"| G+M9+x9",round(r3,3))
