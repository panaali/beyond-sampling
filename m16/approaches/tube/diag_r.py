import sys, time; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=360*20
for name,fn in [('23',faces23),('12',faces12)]:
    fc=fn(n); Z=fc['Z']; full,_,cf=F_faces(M,fc,60)
    print(name,'full',Z*full.mean(),'cells',cf.mean(),'per-face sd',Z*full.std(),'P(F>0)',(full>0).mean())
    for rmax in [0.02,0.05,0.1,0.2,0.4,0.8]:
        f2=dict(fc); f2['hi']=np.minimum(fc['hi'],rmax)
        F,R,C=F_faces(M,f2,60)
        print(' rmax',rmax,'e',Z*F.mean(),'cells',C.mean(),'sd',Z*F.std())
