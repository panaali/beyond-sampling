"""Exact rational enclosures. No floating operation certifies an endpoint."""
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from math import isqrt

COUNTS=Counter()
PEAK=[0,0]
TERMS=64
SQRT_BITS=160

def observe(x):
    PEAK[0]=max(PEAK[0],abs(x.numerator).bit_length())
    PEAK[1]=max(PEAK[1],x.denominator.bit_length())
    return x

def calc(op,a,b):
    COUNTS['rational_'+op]+=1
    if op=='add':return observe(a+b)
    if op=='sub':return observe(a-b)
    if op=='mul':return observe(a*b)
    assert op=='div' and b
    return observe(a/b)

@dataclass(frozen=True)
class I:
    lo:F
    hi:F
    def __post_init__(self):
        assert isinstance(self.lo,F) and isinstance(self.hi,F) and self.lo<=self.hi
        observe(self.lo);observe(self.hi)
    @staticmethod
    def point(x):
        return I(F(x),F(x))
    def __add__(self,other):
        b=coerce(other)
        return I(calc('add',self.lo,b.lo),calc('add',self.hi,b.hi))
    __radd__=__add__
    def __neg__(self):
        COUNTS['rational_neg']+=2
        return I(-self.hi,-self.lo)
    def __sub__(self,other):return self+-coerce(other)
    def __rsub__(self,other):return coerce(other)+-self
    def __mul__(self,other):
        b=coerce(other)
        values=[calc('mul',a,c) for a in (self.lo,self.hi) for c in (b.lo,b.hi)]
        COUNTS['interval_minmax_4']+=1
        return I(min(values),max(values))
    __rmul__=__mul__
    def reciprocal(self):
        assert self.hi<0 or self.lo>0
        return I(calc('div',F(1),self.hi),calc('div',F(1),self.lo))
    def __truediv__(self,other):return self*coerce(other).reciprocal()
    def __rtruediv__(self,other):return coerce(other)*self.reciprocal()

def coerce(x):return x if isinstance(x,I) else I.point(x)

def atan_small(x):
    x=F(x);assert 0<=x<=F(1,2)
    COUNTS['atan_small_calls']+=1
    if not x:return I.point(0)
    xx=calc('mul',x,x);power=x;total=F(0)
    for j in range(TERMS):
        term=calc('div',power,F(2*j+1))
        total=calc('add' if j%2==0 else 'sub',total,term)
        power=calc('mul',power,xx)
    COUNTS['atan_series_terms']+=TERMS
    remainder=calc('div',power,F(2*TERMS+1))
    assert TERMS%2==0
    return I(total,calc('add',total,remainder))

@lru_cache(maxsize=1)
def pi_interval():
    return 16*atan_small(F(1,5))-4*atan_small(F(1,239))

def atan_point(x):
    x=F(x)
    if x<0:return -atan_point(-x)
    if x<=F(1,2):return atan_small(x)
    if x==1:return pi_interval()/4
    if x>1:return pi_interval()/2-atan_point(calc('div',F(1),x))
    # .5 < x < 1, so (1-x)/(1+x) is in(0,1/3).
    q=calc('div',calc('sub',F(1),x),calc('add',F(1),x))
    return pi_interval()/4-atan_small(q)

def atan_interval(x):
    x=coerce(x)
    return I(atan_point(x.lo).lo,atan_point(x.hi).hi)

def sqrt_point(x):
    x=F(x);assert x>=0
    COUNTS['sqrt_isqrt_calls']+=1
    scale=1<<SQRT_BITS
    n=isqrt((x.numerator*(scale*scale))//x.denominator)
    lo=F(n,scale)
    hi=lo if lo*lo==x else F(n+1,scale)
    assert lo*lo<=x<=hi*hi
    return I(lo,hi)

def sqrt_interval(x):
    x=coerce(x);assert x.lo>=0
    return I(sqrt_point(x.lo).lo,sqrt_point(x.hi).hi)

def event_bounds(k):
    k=F(k);assert k>0
    pi=pi_interval();rt=sqrt_point(F(2))
    angle=atan_interval(1/rt)
    lead=1/(8*rt*pi*k)+angle/(4*rt*pi*pi*k)
    remainder=1/(16*pi*k*k*k)
    bracket=I(max(F(0),(lead-remainder).lo),lead.hi)
    wedge=3*atan_point(1/k)/(16*pi)
    return {'k':k,'leading':lead,'cubic_remainder':remainder,
            'event_probability':bracket,'rank_factor_wedge_upper':wedge}

def serial(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,I):return {'lower':str(x.lo),'upper':str(x.hi),'lower_float_display':float(x.lo),'upper_float_display':float(x.hi)}
    if isinstance(x,dict):return {k:serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x

def accounting():
    return {'named_exact_operations':dict(COUNTS),'max_observed_numerator_bits':PEAK[0],
            'max_observed_denominator_bits':PEAK[1],
            'scope':'Named rational expressions and interval operations only; Fraction normalization, comparisons, integer isqrt/shift/division internals and serialization are not scalar-unit priced.'}
