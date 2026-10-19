import numpy as np
def relufit(A,y,iters=15):
    m=y>1e-6
    for it in range(iters):
        co=np.linalg.lstsq(A[m],y[m],rcond=None)[0]
        p=A@co; m=(y>1e-6)|(p>0)
    p=np.maximum(A@co,0); return co,1-((p-y)**2).mean()/y.var()
