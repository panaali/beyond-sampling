import numpy as np
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
def relufit(A,y,iters=30):
    # A includes ones column. alternate: fit LS on set where y>0 or pred>0
    act=y>1e-6
    c=np.linalg.lstsq(A[act],y[act],rcond=None)[0]
    for _ in range(iters):
        z=A@c; m=(y>1e-6)|(z>0)
        # for points with y=0 and z>0 target 0; points with y>0 target y
        c_new=np.linalg.lstsq(A[m],y[m],rcond=None)[0]
        if np.allclose(c_new,c,atol=1e-6): break
        c=c_new
    return c, r2(y,np.maximum(A@c,0))
