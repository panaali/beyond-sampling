import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from estimator import *
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0); truth=0.95317945
B=int(sys.argv[1]) if len(sys.argv)>1 else 4000
for name,fn in [("A",est_tube),("B",est_tube_newton)]:
  for imp in [True]:
    res=[];P=[];U=[]
    for r in range(40):
        a,u,p=fn(M,B,shift=np.random.default_rng(100+r).random(11),parts=True,imp=imp); res.append(a);P.append([p['e23'],p['e12'],p['eb']]);U.append(u)
    res=np.array(res);P=np.array(P)
    print(name,'imp',imp,'used',np.mean(U),'mean',res.mean(),'bias',res.mean()-truth,'sd',res.std(),'rmse',np.sqrt(np.mean((res-truth)**2)),'samp sd',np.sqrt(truth*(1-truth)/np.mean(U)),'part sds',P.std(0),'part means',P.mean(0))
