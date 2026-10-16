import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube')
import numpy as np, time
from estimator import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0); truth=0.95317945
for B in [4000,16000,64000]:
    t=time.time(); res=[]
    for r in range(6 if B<64000 else 3):
        sh=SHIFT if r==0 else np.random.default_rng(r).random(11)   # random shifts: diagnosis of quadrature error only
        a,u,p=est_tube(M,B,shift=sh,parts=True); res.append(a)
    res=np.array(res)
    print(B,'used',round(u),'ests',np.round(res,5),'rmse',np.sqrt(np.mean((res-truth)**2)),'sampling sd',np.sqrt(truth*(1-truth)/u),round(time.time()-t,1),p)
