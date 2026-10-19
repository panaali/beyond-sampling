import numpy as np,sys
x=np.load('x1.npy');h=np.load('h1.npy')
o=np.argsort(-x,1);s=-np.sort(-x,1)
np.set_printoptions(precision=0,suppress=True,linewidth=150)
for i in map(int,sys.argv[1:]):
  CM=np.full((10,10),np.nan);CS=CM.copy()
  for a in range(10):
    for b in range(10):
      m=(o[:,0]==a)&(o[:,1]==b)
      if m.sum()<50: continue
      c,*_=np.linalg.lstsq(s[m,:2],h[m,i],rcond=None);CM[a,b],CS[a,b]=c
  print('neuron',i,'coefM rows=amax cols=asec');print(CM);print('coefS');print(CS)
