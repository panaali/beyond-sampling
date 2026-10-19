import numpy as np,sys
x=np.load('x1.npy'); h=np.load('h1.npy')
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
np.set_printoptions(linewidth=200,suppress=True)
for i in map(int,sys.argv[1:]):
  print('neuron',i,'   cols: s0 s1 s2 x0..x9')
  for k in range(10):
    m=am==k; y=h[m,i]
    A=np.c_[s[m,0],s[m,1],s[m,2],x[m]]
    A[np.arange(m.sum()),3+k]=0  # remove x_am (=s0)
    c,*_=np.linalg.lstsq(A,y,rcond=None); r=1-((A@c-y)**2).mean()/y.var()
    print(' am',k,'R2 %.3f'%r,np.round(c,1))
