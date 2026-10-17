"""Positive weighted-sphere quadrature with an analytic truncation bound.

The bound assumes the supplied geometry and real arithmetic are exact. It is
not a certificate for floating-point clipping, determinants, or function values.
"""
from functools import lru_cache
from math import exp,lgamma,log,pi
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=12)
def rule(order):
    nodes,weights=leggauss(order)
    nodes=(nodes+1)/2;weights=weights/2
    s,t=np.meshgrid(nodes,nodes,indexing="ij")
    bary=np.array([s,(1-s)*t,(1-s)*(1-t)]).reshape(3,-1)
    w=(weights[:,None]*weights[None,:]*(1-s)).reshape(-1)
    return bary,w


def triangle_moment(vertices,power,absolute_tolerance,max_order=128):
    rays=np.asarray(vertices,dtype=float)
    rays=rays/np.sum(np.abs(rays),axis=1)[:,None]
    determinant=abs(float(np.linalg.det(rays)))
    if determinant==0:
        return 0.,dict(quadrature_nodes=0,quadrature_error_bound=0.,order=0)
    if power<0 or int(power)!=power or np.min(rays[:,0])<0:
        raise ValueError("need integer nonnegative power and coordinate0 nonnegative")
    if absolute_tolerance<=0:
        raise ValueError("need a positive tolerance")
    # A separating unit direction bounds ||sum y_i rays_i|| from below.
    center=rays.sum(axis=0)
    center=center/np.linalg.norm(center)
    minimum_projection=float(np.min(rays@center))
    signs=np.sign(rays.sum(axis=0))
    sign_projection=float(np.min(rays@signs))
    qmin=max(0.,minimum_projection)**2
    if sign_projection>0:
        qmin=max(qmin,sign_projection**2/float(signs@signs))
    qmax=float(np.max(np.sum(rays*rays,axis=1)))
    if qmin<=0:
        raise ArithmeticError("no positive radial lower bound; subdivide the triangle")
    qmin=min(qmin,qmax)
    center_q=(qmin+qmax)/2
    ratio=max(0.,(qmax-qmin)/(qmax+qmin))
    exponent=(power+3)/2
    amax=float(np.max(rays[:,0]))
    numerator_bound=amax**power
    chosen=None
    for order in (4,8,12,16,24,32,48,64,96,128):
        if order>max_order:
            continue
        degree=(2*order-2-power)//2
        if degree<0:
            continue
        first_omitted=degree+1
        next_ratio=ratio*(exponent+first_omitted)/(first_omitted+1)
        if next_ratio>=1:
            continue
        if ratio==0 or numerator_bound==0:
            error=0.
        else:
            log_term=(lgamma(exponent+first_omitted)-lgamma(exponent)
                      -lgamma(first_omitted+1)+first_omitted*log(ratio))
            error=(determinant*numerator_bound*center_q**(-exponent)
                   *exp(log_term)/(1-next_ratio))
        if error<=absolute_tolerance:
            chosen=(order,error,degree);break
    if chosen is None:
        raise ArithmeticError(f"quadrature bound exceeds tolerance at order {max_order}")
    order,error,degree=chosen
    misses_before=rule.cache_info().misses
    bary,weights=rule(order)
    generated=rule.cache_info().misses-misses_before
    points=rays.T@bary
    q=np.sum(points*points,axis=0)
    integral=determinant*float(np.dot(weights,points[0]**power/q**exponent))
    return integral,dict(quadrature_nodes=order*order,quadrature_error_bound=error,
                         order=order,polynomial_degree=degree,
                         legendre_rule_generations=generated,
                         legendre_rule_generation_nodes=generated*order,
                         qmin=qmin,qmax=qmax,series_ratio=ratio)


def polygon_tail_mass(vertices,prefix_dimension,absolute_tolerance=1e-12,max_order=128):
    if prefix_dimension<1:
        raise ValueError("prefix dimension must be positive")
    vertices=np.asarray(vertices,dtype=float)
    if len(vertices)<3:
        return 0.,dict(quadrature_nodes=0,quadrature_error_bound=0.,max_order=0)
    factor=prefix_dimension/(2*pi)
    triangles=len(vertices)-2
    mass=0.;error=0.;nodes=0;orders=[];generations=0;generation_nodes=0
    for b,c in zip(vertices[1:-1],vertices[2:]):
        value,meta=triangle_moment(np.array([vertices[0],b,c]),prefix_dimension-1,
                                   absolute_tolerance/(factor*triangles),max_order)
        mass+=factor*value;error+=factor*meta["quadrature_error_bound"]
        nodes+=meta["quadrature_nodes"];orders.append(meta["order"])
        generations+=meta.get("legendre_rule_generations",0)
        generation_nodes+=meta.get("legendre_rule_generation_nodes",0)
    return mass,dict(quadrature_nodes=nodes,quadrature_error_bound=error,
                     max_order=max(orders),roundoff_certified=False,
                     legendre_rule_generations=generations,
                     legendre_rule_generation_nodes=generation_nodes)
