import numpy as np
x=np.load('x1.npy'); H=np.load('H1.npy')
M=np.maximum.accumulate(x,1)
s=-np.sort(-x,1)
F=np.concatenate([x,M,s[:,:3]],1)
names=[f'x{t}' for t in range(10)]+[f'M{t}' for t in range(10)]+['s1','s2','s3']
np.set_printoptions(precision=2,suppress=True,linewidth=220)
for i in range(16):
  a=H[:,i]>1e-6
  A=F[a]; y=H[a,i]
  c,*_=np.linalg.lstsq(A,y,rcond=None)
  r2=1-((A@c-y)**2).mean()/y.var()
  print(i,'n',a.sum(),'R2 %.3f'%r2, ' '.join(f'{n}:{v:.1f}' for n,v in zip(names,c) if abs(v)>0.8))
