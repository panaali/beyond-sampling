"""Diagnosis: spread of the estimate over 30 random shifts of the quadrature rule (original model)."""
import sys; sys.path.insert(0,'/home/user/clean-neurons/m16/approaches/tube'); sys.path.insert(0,'/home/user/clean-neurons/m16')
import numpy as np, warnings; warnings.filterwarnings('ignore')
from estimator import est_tube
from arc_eval import Wi0, Wh0, Wo0
M=(Wi0,Wh0,Wo0); truth=0.95317945
for B in [1000,16000]:
    r=[est_tube(M,B,shift=np.random.default_rng(500+i).random(11)) for i in range(30)]
    e=np.array([a for a,_ in r]); u=np.mean([b for _,b in r])
    print(B,'used',round(u),'mean',e.mean(),'sd',e.std(),'rmse',np.sqrt(np.mean((e-truth)**2)),'sampling sd',np.sqrt(truth*(1-truth)/u),'MSE gain',truth*(1-truth)/u/np.mean((e-truth)**2))
