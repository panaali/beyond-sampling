"""Exact scalar rank intervals for fixed two-input prefixes."""
import numpy as np
from scipy.special import ndtr

def normal_mass(lo,hi,stats):
    valid=hi>lo;right=np.maximum(lo,hi)
    stats['coupled_cdf_values']+=4*np.size(lo)
    value=np.where(lo>0,ndtr(-lo)-ndtr(-right),ndtr(right)-ndtr(lo))
    finite=valid&np.isfinite(lo)&np.isfinite(right)
    narrow=np.zeros_like(valid)
    narrow[finite]=(right[finite]-lo[finite])*(1+np.maximum(abs(lo[finite]),abs(right[finite])))<1e-5
    if np.any(narrow):
        a=lo[narrow];b=right[narrow];width=b-a
        density=lambda x:np.exp(-x*x/2)/np.sqrt(2*np.pi)
        lower=width*np.minimum(density(a),density(b))
        upper=width*density(np.minimum(b,np.maximum(a,0.)))
        value[narrow]=width*np.exp(-a*a/2-a*width/2-width*width/8)/np.sqrt(2*np.pi)
        allowance=16*np.finfo(float).eps*upper
        stats['narrow_gaussian_intervals']+=len(a)
        stats['narrow_gaussian_exp_calls']+=4*len(a)
        stats['narrow_gaussian_mass_upper_sum']+=float(np.sum(upper))
        stats['narrow_gaussian_uncertainty_proxy_sum']+=float(np.sum(upper-lower+allowance))
    if np.any(value<0):
        index=np.flatnonzero(value<0)[0]
        raise ArithmeticError(dict(message='negative Gaussian interval',lo=float(np.ravel(lo)[index]),hi=float(np.ravel(hi)[index]),right=float(np.ravel(right)[index]),value=float(np.ravel(value)[index]),cdf_lo=float(ndtr(np.ravel(lo)[index])),cdf_right=float(ndtr(np.ravel(right)[index])),sf_lo=float(ndtr(-np.ravel(lo)[index])),sf_right=float(ndtr(-np.ravel(right)[index]))))
    return np.where(valid,value,0.)

def rank_mass(y,lo,hi,prefix,label,stats):
    a,b=prefix
    if a==b:raise ArithmeticError('positive-weight prefix rank tie')
    if label==3:
        ordered=np.sort(np.stack((np.full_like(y,a),np.full_like(y,b),y)),axis=0)
        stats['coupled_three_value_sorts']+=len(y)
        return normal_mass(np.maximum(lo,ordered[1]),np.minimum(hi,ordered[2]),stats)
    if label==0:
        target=np.full_like(y,a);count=int(b>a)+(y>=a).astype(int)
    elif label==1:
        target=np.full_like(y,b);count=int(a>b)+(y>=b).astype(int)
    elif label==2:
        target=y;count=(a>y).astype(int)+(b>y).astype(int)
    else:raise ValueError(label)
    above=normal_mass(np.maximum(lo,target),hi,stats)
    below=normal_mass(lo,np.minimum(hi,target),stats)
    stats['coupled_rank_comparisons']+=4*len(y)
    return np.where(count==0,above,np.where(count==1,below,0.))

def panels(pieces,prefix,stats):
    result=[]
    for left,right,lower,upper,label in pieces:
        cuts=[left,right]
        for value in prefix:
            if left<value<right:cuts.append(float(value))
            for line in (lower,upper):
                stats['coupled_panel_intersection_tests']+=1
                if line[0]!=0:
                    root=(value-line[1])/line[0]
                    if left<root<right:cuts.append(float(root))
        for line in (lower,upper):
            stats['coupled_panel_intersection_tests']+=1
            if line[0]!=1:
                root=-line[1]/(line[0]-1)
                if left<root<right:cuts.append(float(root))
        cuts=np.unique(cuts);stats['coupled_panel_sort_items']+=len(cuts)
        result.extend((a,b,lower,upper,label) for a,b in zip(cuts[:-1],cuts[1:]))
    return result

def fixed_prefix_probabilities(prefix):
    a,b=prefix
    if a==b:raise ArithmeticError('prefix tie')
    qa,qb=ndtr(a),ndtr(b)
    first=qa*qa if a<b else 2*qa*(1-qa)
    second=qb*qb if b<a else 2*qb*(1-qb)
    other=(1-first-second)/2
    return np.array([first,second,other,other])
