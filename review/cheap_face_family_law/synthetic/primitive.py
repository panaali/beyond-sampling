"""Exact-real rational interpolation prototype; never a NumPy parity claim."""
from bisect import bisect_right
from fractions import Fraction as F

CUTS=(F(0),F(1,4),F(1,2))

def rational(x):
    if type(x) is not F: raise TypeError('exact Fraction required')
    return x

class Grid:
    def __init__(self,x,c):
        self.x=tuple(rational(v) for v in x)
        self.c=tuple(rational(v) for v in c)
        if len(self.x)!=len(self.c) or len(self.x)<2: raise ValueError('matching nontrivial grid required')
        if any(a>=b for a,b in zip(self.x,self.x[1:])): raise ValueError('strictly increasing coordinates required')
        if self.c[0]!=0 or self.c[-1]!=1: raise ValueError('normalized endpoints required')
        if any(a>b for a,b in zip(self.c,self.c[1:])): raise ValueError('monotone CDF required')
    def cdf(self,u):
        u=rational(u)
        if u<=self.x[0]: return F(0)
        if u>=self.x[-1]: return F(1)
        j=bisect_right(self.x,u)-1
        return self.c[j]+(self.c[j+1]-self.c[j])*(u-self.x[j])/(self.x[j+1]-self.x[j])
    def inverse(self,q):
        q=rational(q)
        if q<0 or q>1: raise ValueError('CDF coordinate outside [0,1]')
        # Fixed explicit endpoint policy: rightmost knot at a repeated CDF value.
        j=bisect_right(self.c,q)-1
        if j==len(self.c)-1: return self.x[-1]
        return self.x[j]+(self.x[j+1]-self.x[j])*(q-self.c[j])/(self.c[j+1]-self.c[j])
    def bins(self):
        bounds=(F(0),*(self.cdf(b) for b in CUTS),F(1))
        return tuple(b-a for a,b in zip(bounds,bounds[1:]))
    def conditional(self,bin_index,v):
        if type(bin_index) is not int or bin_index<0 or bin_index>=4: raise ValueError('bin index outside 0..3')
        v=rational(v)
        if not 0<v<1: raise ValueError('conditional coordinate must be strictly interior')
        bounds=(F(0),*(self.cdf(b) for b in CUTS),F(1))
        lo,hi=bounds[bin_index:bin_index+2]
        if hi==lo: raise ValueError('zero probability stratum has no conditional law')
        q=lo+(hi-lo)*v
        return {'probability':hi-lo,'q':q,'u':self.inverse(q)}
