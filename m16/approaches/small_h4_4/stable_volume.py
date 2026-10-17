"""Stable signed-orthant simplicial Gaussian mass in R4.

Conditional on exact cone geometry/arithmetic, error <= determinant * tail/(6*pi²),
tail=(12*n+9)/3**(n-1). Global disjoint-cone accuracy error <= 16 times that factor.
"""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(None)
def rule(n):
    x,w=leggauss(n);x=(x+1)/2;w=w/2
    u,v,z=np.meshgrid(x,x,x,indexing='ij')
    wu,wv,wz=np.meshgrid(w,w,w,indexing='ij')
    y=np.stack((u,(1-u)*v,(1-u)*(1-v)*z,(1-u)*(1-v)*(1-z)),axis=-1).reshape(-1,4)
    weight=(wu*wv*wz*(1-u)**2*(1-v)).ravel()
    return y,weight


def mass(rays,n=20):
    """Columns rays share an orthant, each has L1 norm one."""
    y,w=rule(n)
    points=y@rays.T
    s=np.sum(points*points,axis=1)
    det=abs(np.linalg.det(rays))
    p=det/(2*np.pi**2)*np.dot(w,1/(s*s))
    bound=det/(6*np.pi**2)*(12*n+9)/3.**(n-1)
    return float(p),float(bound)
