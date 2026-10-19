import numpy as np
G=np.load('G.npy')
k=9-np.arange(9)
np.set_printoptions(precision=1,suppress=True,linewidth=200)
for i in range(16):
  g=G[:9,i]; best=(1e9,)
  for T in np.linspace(2.5,5,101):
    w=2*np.pi/T
    F=np.stack([np.ones(9),np.cos(w*k),np.sin(w*k)],1)
    c,*_=np.linalg.lstsq(F,g,rcond=None); e=((F@c-g)**2).sum()
    if e<best[0]: best=(e,T,c,F@c)
  e,T,c,fit=best
  print(i,'T=%.2f'%T,'rel err %.3f'%(e/((g-g.mean())**2).sum()),'G',g,'fit',fit)
