"""Analytic reset-prefix ranking marginalization. No model forward evaluations.

m ordinary iid normal inputs, then two independent negative-truncated inputs,
then a fixed suffix. For exact algebra in Gaussian-CDF coordinates, use Fraction
inputs to discarded_target_probability; normal_probability evaluates Phi in float.
"""
from fractions import Fraction
from itertools import product
from math import comb
import numpy as np
from scipy.special import ndtr


def count_probability(v,a,b,k):
    """Probability k=0 or 1 of a normals+b negative-truncated normals exceed u."""
    if a<0 or b<0 or k not in (0,1):raise ValueError('invalid counts')
    g=min(1,2*v)
    if k==0:return v**a*g**b
    ans=0*v
    if a:ans+=a*(1-v)*v**(a-1)*g**b
    if b:ans+=b*(1-g)*g**(b-1)*v**a
    return ans


def monomial_integral(lo,hi,p):
    if p<0:raise ValueError('negative power')
    return (hi**(p+1)-lo**(p+1))/(p+1)


def count_integral(lo,hi,a,b,k):
    """Integral over v of count_probability, with exact rational option."""
    if not (0<=lo<=hi<=1):raise ValueError('CDF interval outside [0,1]')
    half=Fraction(1,2) if isinstance(lo,Fraction) or isinstance(hi,Fraction) else .5
    ans=0*lo
    end=min(hi,half)
    if lo<end:
        n=a+b
        if k==0:ans+=2**b*monomial_integral(lo,end,n)
        elif n:
            c1=a*2**b+(b*2**(b-1) if b else 0)
            c2=(a+b)*2**b
            ans+=c1*monomial_integral(lo,end,n-1)-c2*monomial_integral(lo,end,n)
    start=max(lo,half)
    if start<hi:
        if k==0:ans+=monomial_integral(start,hi,a)
        elif a:ans+=a*(monomial_integral(start,hi,a-1)-monomial_integral(start,hi,a))
    return ans


def discarded_target_probability(m,target,suffix_cdfs):
    """P(target is second-largest | fixed suffix and negative reset pair).

    target < m is ordinary; m or m+1 is in the reset pair. CDF values may be
    Fractions for exact tests. This integrates the target magnitude as well.
    """
    if not 0<=target<m+2:raise ValueError('target is not discarded')
    values=sorted(suffix_cdfs,reverse=True)
    rational=any(isinstance(v,Fraction) for v in values)
    zero=Fraction(0) if rational else 0.;one=zero+1
    largest=values[0] if values else zero
    second=values[1] if len(values)>1 else zero
    ordinary=target<m
    a=m-int(ordinary);b=2-int(not ordinary)
    cap=one if ordinary else one/2
    density=1 if ordinary else 2
    # c=0 for target above the suffix maximum; c=1 between the top two.
    p=zero
    if largest<cap:p+=count_integral(largest,cap,a,b,1)
    hi=min(largest,cap)
    if second<hi:p+=count_integral(second,hi,a,b,0)
    return density*p


def cdf_probability(m,target,values):
    """Conditional rank probability in CDF coordinates, preserving Fractions."""
    if target<m+2:return discarded_target_probability(m,target,values)
    j=target-(m+2)
    if not 0<=j<len(values):raise ValueError('target outside full sequence')
    v=values[j];c=sum(x>v for x in values)
    if c>1:return 0*v
    return count_probability(v,m,2,1-c)


def normal_probability(m,target,suffix):
    return float(cdf_probability(m,target,list(ndtr(suffix))))


def sign_rank_masses(T,requirements):
    masses=[Fraction(0) for _ in range(T)]
    for signs in product((0,1),repeat=T):
        if any(signs[i]!=v for i,v in requirements.items()):continue
        pos=[i for i,s in enumerate(signs) if s]
        group=pos if len(pos)>=2 else [i for i,s in enumerate(signs) if not s]
        for i in group:masses[i]+=Fraction(1,2**T*len(group))
    return masses


def last_reset_masses(T):
    """Exact sign masses including no-reset remainder, indexed by suffix length."""
    counts=[0]*(T-1);no_reset=0
    for signs in product((0,1),repeat=T):
        ends=[i for i in range(1,T) if signs[i-1:i+1]==(0,0)]
        if ends:counts[T-ends[-1]-1]+=1
        else:no_reset+=1
    return {'suffix_masses':[str(Fraction(n,2**T)) for n in counts],
            'no_reset_mass':str(Fraction(no_reset,2**T)),
            'total_mass':str(Fraction(sum(counts)+no_reset,2**T))}
