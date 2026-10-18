"""Directed rational Gaussian bin masses; no trained model calls."""
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'prefix_wedge_remainder'))
from rational_bounds import I,pi_interval,sqrt_interval,calc,COUNTS,accounting,serial
BOUNDARIES_V=(-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8)
Q=1<<80
DEGREE=96

@lru_cache(maxsize=None)
def cdf(x):
    x=F(x);assert -4<=x<=4
    COUNTS['local_cdf_unique_calls']+=1
    if x<0:return 1-cdf(-x)
    if x==0:return I.point(F(1,2))
    z=-calc('mul',x,x)/2;term=total=x;lower=None
    for n in range(1,DEGREE+1):
        term=calc('div',calc('mul',calc('mul',term,z),F(2*n-1)),F(n*(2*n+1)))
        total=calc('add',total,term)
        if n==DEGREE-1:lower=total
    assert lower is not None and lower<=total
    norm=1/sqrt_interval(2*pi_interval())
    return F(1,2)+norm*I(lower,total)

def dyadic(q):
    lo=q.lo.numerator*Q//q.lo.denominator
    hi=-((-q.hi.numerator*Q)//q.hi.denominator)
    assert F(lo,Q)<=q.lo<=q.hi<=F(hi,Q)
    return lo,hi

def setup():
    rows=[]
    for v in BOUNDARIES_V:
        p=cdf(F(v,2));l,u=dyadic(p);assert 0<=l<=u<=Q
        rows.append(dict(v=v,exact=p,dyadic=[l,u]))
    bins=[]
    for left,right in zip(rows,rows[1:]):
        l=right['dyadic'][0]-left['dyadic'][1];u=right['dyadic'][1]-left['dyadic'][0]
        assert 0<l<=u;bins.append((l,u))
    central=(rows[-1]['dyadic'][0]-rows[0]['dyadic'][1],rows[-1]['dyadic'][1]-rows[0]['dyadic'][0])
    tail=(Q**4-central[1]**4,Q**4-central[0]**4)
    assert 0<=tail[0]<=tail[1]<=Q**4
    return dict(cdf_rows=rows,bin_numerators=bins,bin_denominator=Q,box_denominator=Q**4,tail_numerators=tail)

def box_mass(bin_ids,bins):
    assert len(bin_ids)==4
    l=u=1
    for j in bin_ids:a,b=bins[j];l*=a;u*=b
    return l,u

def crossover(width_lower,cs,budget=10**7):
    w=F(width_lower);assert 0<=w<=1 and type(cs)is int and cs>0
    if not w:return dict(crossover=None,frozen_budget=budget,sufficient_tenfold=False)
    threshold=F(25)/w**2;n=threshold.numerator//threshold.denominator+1
    assert n*w*w>25 and (n-1)*w*w<=25
    return dict(minimum_samples=n,minimum_common_budget=n*cs,frozen_budget=budget,frozen_samples=budget//cs,sufficient_tenfold=(budget//cs)*w*w>25,scope='Conservative abstention-lower versus maximal sampling-SD comparison only; certificate setup granted free, no actual error/MSE or fullcost claim.')
