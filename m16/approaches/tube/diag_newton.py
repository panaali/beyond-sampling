import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube'); sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from tube import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0)
n=4096
for fn in (faces23,faces12):
    f=fn(n); Fe,_,Ce=F_faces(M,f,400,radial=True); Fe*=f['Z']
    print('exact',Fe.mean(),Fe.std(),Ce.mean())
    for J in [1,2,3,5,8]:
        Fn,Cn=F_newton(M,f,J); Fn*=f['Z']
        print(' J',J,'newton',Fn.mean(),'sd',Fn.std(),'cost',Cn.mean(),'corr',np.corrcoef(Fe,Fn)[0,1],'diff sd',(Fe-Fn).std(),'var*cost',Fn.var()*Cn.mean())
