import numpy as np
np.set_printoptions(suppress=True,linewidth=160)
x=np.load('x.npy'); h=np.load('h.npy')
def ev(B,y):
  m=y>0
  c,*_=np.linalg.lstsq(B[m],y[m],rcond=None)
  for it in range(4):
    pr=B@c; mm=(pr>0)|m; c,*_=np.linalg.lstsq(B[mm],y[mm],rcond=None)
  pr=np.maximum(B@c,0); return round(1-((pr-y)**2).mean()/y.var(),4), c.round(3)
s9=-np.sort(-x[:,:9],1)
print("n5 sorted0..8 + x9", ev(np.c_[s9,x[:,9]],h[:,5]))
print("n5 positional x0..8 + sorted + x9", ev(np.c_[x[:,:9],s9,x[:,9]],h[:,5]))
print("n5 M8,x9", ev(np.c_[s9[:,0],x[:,9]],h[:,5]))
print("n5 top2 mean,x9", ev(np.c_[s9[:,:2].sum(1),x[:,9]],h[:,5]))
print("n5 top4 mean,x9", ev(np.c_[s9[:,:4].sum(1),x[:,9]],h[:,5]))
print("n5 relu(x)sum,x9", ev(np.c_[np.maximum(x[:,:9],0).sum(1),x[:,9]],h[:,5]))
s8=-np.sort(-x[:,:8],1)
print("n4 sorted0..7 + x8 + x9", ev(np.c_[s8,x[:,8],x[:,9]],h[:,4]))
print("n4 pos+sorted", ev(np.c_[x[:,:8],s8,x[:,8],x[:,9]],h[:,4]))
