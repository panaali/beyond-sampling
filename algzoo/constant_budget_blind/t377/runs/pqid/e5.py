import numpy as np
x=np.load('x2.npy');h=np.load('h2.npy')
M=x.max(1); M08=x[:,:9].max(1)
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for name,MM in [('Mall',M),('M08',M08)]:
  print(name)
  for i in range(16):
    X=np.c_[np.ones(len(x)),MM[:,None]-x]
    g=h[:,i]>0.3
    for it in range(3):
        c,*_=np.linalg.lstsq(X[g],h[g,i],rcond=None)
        pr=X@c; g=(h[:,i]>0.3)|(pr>0)
    pred=np.maximum(X@c,0)
    r2=1-((h[:,i]-pred)**2).mean()/h[:,i].var()
    print(i,'R2 %.3f'%r2,c)
