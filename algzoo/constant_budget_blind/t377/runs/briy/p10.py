import numpy as np
from dic3 import relufit, pref
x=np.load('x1.npy'); h=np.load('h1.npy')
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]
np.set_printoptions(linewidth=200,suppress=True,precision=2)
M8,S8,A8=pref(x,8)
m=am==6; y=h[m,14]
A=np.c_[x[m],s[m,1]]; c,*_=np.linalg.lstsq(A,y,rcond=None); print('n14 am=6 x0..9,s1',c, 1-((A@c-y)**2).mean()/y.var())
B=np.c_[x,M8]
for i in [12,15,3,9,4,13,0,5,7,8]:
    c,r=relufit(B,h[:,i]); print(i,'R2 %.3f'%r,'x0..x9,M8:',c)
