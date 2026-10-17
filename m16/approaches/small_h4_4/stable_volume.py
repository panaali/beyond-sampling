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


def chebyshev_tail(a,b,degree):
    """Uniform tail for s^-2 on[a,b], after the given Chebyshev degree."""
    delta=np.sqrt(a*b)
    center=(a+b)/2
    q=((b-a)/2)/(center+delta)
    if q==0:return 0.
    first=degree+1
    return float(2*q**first*(center/delta**3/(1-q)+(first/(1-q)+q/(1-q)**2)/delta**2))


def adaptive_mass(rays,total_tolerance=1e-8):
    """Geometry-only order selection; ideal global determinant budget is16.

    Returned error bounds quadrature only, not floating ray/determinant arithmetic.
    """
    gram=rays.T@rays
    a=max(.25,float(np.min(gram)))
    b=max(a,float(np.max(np.diag(gram))))
    determinant=abs(np.linalg.det(rays))
    for n in (2,3,4,6,8,12,16,20,24):
        tail=chebyshev_tail(a,b,n-2)
        per_determinant=tail/(6*np.pi**2)
        if per_determinant<=total_tolerance/16:break
    else:raise ArithmeticError('requested cone quadrature tolerance exceeds fixed order ladder')
    y,w=rule(n);points=y@rays.T;s=np.sum(points*points,axis=1)
    value=determinant/(2*np.pi**2)*np.dot(w,1/(s*s))
    return float(value),float(determinant*per_determinant),dict(order=n,lower_norm2=a,upper_norm2=b)
