import numpy as np,sys
x=np.load('x1.npy'); h=np.load('h1.npy')
s=np.sort(x,1)[:,::-1]; o=np.argsort(-x,1); am=o[:,0]; a2=o[:,1]
for i in map(int,sys.argv[1:]):
  print('neuron',i)
  for k in range(10):
    m=am==k; y=h[m,i]
    A=np.c_[s[m,0],s[m,1],x[m,9],x[m,8],x[m,7]]
    c,*_=np.linalg.lstsq(A,y,rcond=None); r=1-((A@c-y)**2).mean()/y.var()
    print(' am',k,'R2 %.3f'%r,'s0,s1,x9,x8,x7',np.round(c,1), 'act%.2f'%(y>0).mean())
