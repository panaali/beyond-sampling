"""Positive exact-degree polynomial rank integrals for a conditioned prefix sign."""
import numpy as np
from scipy.special import ndtr

NODES=(1+np.array([-1.,1.])/np.sqrt(3.))/2


def G(q,sign):
    return np.maximum(0.,2*q-1) if sign>0 else np.minimum(1.,2*q)


def integral(function,lo,hi,stats):
    """Two positive nodes, exact for polynomial degree<=3 in real arithmetic."""
    hi=np.maximum(lo,hi)
    width=hi-lo
    q=lo[...,None]+width[...,None]*NODES
    values=function(q)
    stats['rank_polynomial_nodes']+=q.size
    stats['rank_interval_scalar_ops']+=8*width.size
    return width*np.mean(values,axis=-1)


def split_integral(function,lo,hi,stats):
    middle=np.minimum(hi,np.maximum(lo,.5))
    return integral(function,lo,middle,stats)+integral(function,middle,hi,stats)


def half_integral(function,lo,hi,sign,stats):
    if sign>0:lo=np.maximum(lo,.5)
    else:hi=np.minimum(hi,.5)
    return integral(function,lo,hi,stats)


def point_weights(qx,qy,m,sign,stats):
    """Suffix-first, suffix-last, special-prefix, each regular-prefix weights."""
    if m not in (2,3):raise ValueError('degree-two-point protocol covers T4/T5 only')
    n=m-1
    def allbelow(q):return G(q,sign)*q**n
    def oneabove(q):return (1-G(q,sign))*q**n+n*G(q,sign)*(1-q)*q**(n-1)
    # At equality use the limiting x>=y order consistently. Such ties have
    # zero continuous-input mass, but occur on empty integration intervals.
    first=np.where(qx>=qy,allbelow(qy),oneabove(qy))
    last=np.where(qx<qy,allbelow(qx),oneabove(qx))
    lo=np.minimum(qx,qy);hi=np.maximum(qx,qy)
    special=half_integral(lambda q:2*q**n,lo,hi,sign,stats)
    special+=half_integral(lambda q:2*n*(1-q)*q**(n-1),hi,np.ones_like(hi),sign,stats)
    regular=split_integral(lambda q:G(q,sign)*q**(n-1),lo,hi,stats)
    def remaining_one(q):
        value=(1-G(q,sign))*q**(n-1)
        if n>1:value+=(n-1)*G(q,sign)*(1-q)*q**(n-2)
        return value
    regular+=split_integral(remaining_one,hi,np.ones_like(hi),stats)
    defect=float(np.max(abs(first+last+special+n*regular-1)))
    stats['rank_partition_maximum_defect']=max(stats['rank_partition_maximum_defect'],defect)
    if defect>1e-11:raise ArithmeticError('conditional rank law does not partition')
    minimum=float(min(x.min() for x in (first,last,special,regular)))
    stats['minimum_rank_weight']=min(stats['minimum_rank_weight'],minimum)
    if minimum < -1e-13:raise ArithmeticError('negative conditional rank weight')
    stats['rank_point_queries']+=np.size(qx)
    return first,last,special,regular


def scalar_integrals(y,lo,hi,m,sign,stats):
    """Integrate conditional truth weights over a final-input normal interval."""
    def cdf_mass(a,b):
        stats['normal_cdf_values']+=4*a.size
        return np.where(a>0,ndtr(-a)-ndtr(-b),ndtr(b)-ndtr(a))
    qy=ndtr(y);stats['normal_cdf_values']+=len(y)
    zero=np.minimum(hi,np.maximum(lo,0.))
    order=np.minimum(hi,np.maximum(lo,y))
    edges=np.sort(np.stack((lo,zero,order,hi),axis=1),axis=1)
    stats['rank_four_endpoint_sorts']+=len(y)
    out=[np.zeros_like(y) for _ in range(4)]
    mass_total=np.zeros_like(y)
    for j in range(3):
        a,b=edges[:,j],edges[:,j+1]
        mass=cdf_mass(a,b);qa=ndtr(a);stats['normal_cdf_values']+=len(a)
        qx=qa[:,None]+mass[:,None]*NODES
        if np.any(qx<0) or np.any(qx>1):raise ArithmeticError('rank-CDF node outside support')
        weights=point_weights(qx,qy[:,None],m,sign,stats)
        for k,w in enumerate(weights):out[k]+=mass*np.mean(w,axis=1)
        mass_total+=mass
        stats['rank_outer_polynomial_nodes']+=qx.size
    return mass_total,*out
