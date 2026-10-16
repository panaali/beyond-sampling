import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=3600; sh=(np.arange(1,10)*0.6180339887)%1
for name,fn in [('23',faces23),('12',faces12)]:
    fc=fn(n,shift=sh); Z=fc['Z']
    Fe,_,Ce=F_faces(M,fc,60,radial=True); Fe*=Z
    print(name,'exact',Fe.mean(),'sd',Fe.std(),'cells',Ce.mean())
    for K in [1,2,3]:
        Fa,_,Ca=F_faces(M,fc,K,radial=True,extrap=True); Fa*=Z
        D=Fe-Fa
        print(' K',K,'extrap',Fa.mean(),'sd',Fa.std(),'cells',Ca.mean(),'corr',np.corrcoef(Fe,Fa)[0,1],'resid sd',D.std(),'P(D!=0)',(np.abs(D)>1e-12).mean())
