import sys; sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=4096
from tube import _qmc
V=(_qmc(n,12)+np.arange(1,13)*0.7548776662)%1
for name,fn,tr in [('23',faces23,0.029203),('12',faces12,0.0092235)]:
    fc=fn(n); Z=fc['Z']
    for k0 in [0.01,0.02,0.04,0.08,0.16]:
        F,C=F_hybrid(M,fc,k0,V[:,10:12]); F*=Z
        print(name,'k0',k0,'est',round(F.mean(),5),'truth',tr,'sd',round(F.std(),4),'cost',round(C.mean(),2),'var*cost',F.var()*C.mean())
