import numpy as np
x=np.load('x1.npy');h=np.load('h1.npy')
o=np.argsort(-x,1);s=-np.sort(-x,1)
np.set_printoptions(precision=2,suppress=True,linewidth=150)
m=o[:,0]==9
for i in [15,10]:
  for a in range(9):
    mm=m&(o[:,1]==a)
    c,*_=np.linalg.lstsq(x[mm],h[mm,i],rcond=None)
    r=1-((h[mm,i]-x[mm]@c)**2).mean()/h[mm,i].var()
    print(i,a,c,'r2 %.3f'%r,(h[mm,i]==0).mean())
# neuron 10 when amax!=9
mm=~m
X=np.c_[x[mm,9],s[mm,0],s[mm,1]]
c,*_=np.linalg.lstsq(X[h[mm,10]>0],h[mm,10][h[mm,10]>0],rcond=None);print('10 active nonmax9 coef on x9,max,sec',c)
