import numpy as np
def feats(x):
    o=np.argsort(-x,1); s=np.take_along_axis(x,o,1)
    return o,s
def relufit(F,h,iters=30):
    # fit h ~ relu(F w)
    w=np.linalg.lstsq(F,h,rcond=None)[0]
    for _ in range(iters):
        m=(h>0)|(F@w>0)
        w=np.linalg.lstsq(F[m],h[m],rcond=None)[0]
    return w
def r2(p,h): return 1-((p-h)**2).mean()/h.var()
