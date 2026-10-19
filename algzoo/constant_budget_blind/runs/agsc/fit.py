import numpy as np
from scipy.optimize import least_squares
from feat import r2
def fitrelu(A,y,c0=None):
    if c0 is None: c0=np.linalg.lstsq(A,y,rcond=None)[0]
    res=least_squares(lambda c: np.maximum(A@c,0)-y, c0)
    return res.x, r2(y,np.maximum(A@res.x,0))
