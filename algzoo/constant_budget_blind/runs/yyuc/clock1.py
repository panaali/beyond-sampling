import numpy as np
from fitlib import *
x=np.load('x1.npy')[:40000]; H=np.load('H1.npy')[:40000]
d=feats(x); T=3.25
th=lambda p: 2*np.pi*(9-p)/T
ca,sa=np.cos(th(d['pa'])),np.sin(th(d['pa'])); cb,sb=np.cos(th(d['pb'])),np.sin(th(d['pb']))
P1,P2,P3,x9=d['P1'],d['P2'],d['P3'],d['x9']
for i in range(16):
  y=H[:,i]
  F7=np.stack([P1,P1*ca,P1*sa,P2,P2*cb,P2*sb,P3,x9],1)
  c7,r7,_=relufit(F7,y)
  ph1=np.arctan2(c7[2],c7[1]); ph2=np.arctan2(c7[5],c7[4])
  q=lambda a: np.round(a/(np.pi/12))*(np.pi/12)
  ph1,ph2=q(ph1),q(ph2)
  F5=np.stack([P1,P1*np.cos(th(d['pa'])-ph1),P2,P2*np.cos(th(d['pb'])-ph2),x9],1)
  c5,r5,_=relufit(F5,y)
  print(i,'full8 R2 %.3f'%r7,' 5c R2 %.3f'%r5,'ph1 %d ph2 %d'%(np.degrees(ph1),np.degrees(ph2)),c5.round(2),flush=True)
