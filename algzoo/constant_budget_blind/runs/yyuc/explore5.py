import numpy as np
x=np.load('x1.npy'); H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; sm=o[:,1]
s=-np.sort(-x,1)
np.set_printoptions(precision=1,suppress=True,linewidth=220)
for i in range(16):
  if i in (0,3,6,15): continue
  y=H[:,i]; T=[]
  for k in range(10):
    m=(am==k)&(y>1e-6)
    if m.sum()<200: T.append([np.nan]*6); continue
    F=np.stack([s[m,0],s[m,1],s[m,2],x[m,9],x[m,8]],1)
    c,*_=np.linalg.lstsq(F,y[m],rcond=None)
    r2=1-((F@c-y[m])**2).mean()/y[m].var()
    T.append(list(c)+[r2])
  T=np.array(T).T
  print('neuron',i,' rows s1,s2,s3,x9,x8,R2 ; cols am=0..9\n',T)
