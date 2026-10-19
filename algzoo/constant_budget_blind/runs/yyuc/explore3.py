import numpy as np
x=np.load('x1.npy'); H=np.load('H1.npy')
o=np.argsort(-x,1); am=o[:,0]; sm=o[:,1]
s=-np.sort(-x,1)
def grpfit(groups,F,y):
  pred=np.zeros_like(y)
  for g in np.unique(groups):
    m=groups==g
    c,*_=np.linalg.lstsq(F[m],y[m],rcond=None); pred[m]=F[m]@c
  return 1-((pred-y)**2).mean()/y.var()
F1=np.stack([s[:,0],s[:,1],s[:,2]],1)
F2=np.stack([s[:,0],s[:,1],s[:,2],x[:,9],x[:,8],s[:,3]],1)
for i in range(16):
  y=H[:,i]
  print(i,'am:%.3f'%grpfit(am,F1,y),'sm:%.3f'%grpfit(sm,F1,y),'am,sm:%.3f'%grpfit(am*10+sm,F1,y),'am,sm+x9x8s4:%.3f'%grpfit(am*10+sm,F2,y))
