"""Positive integer-power Chebyshev tails with directed Decimal enclosures.

This certifies arithmetic given valid radial/determinant inputs, not geometry.
"""
from collections import Counter,defaultdict
from dataclasses import dataclass
from decimal import (Decimal,Context,ROUND_FLOOR,ROUND_CEILING,InvalidOperation,
                     DivisionByZero,Overflow,Underflow,Clamped)
from fractions import Fraction
from math import comb,factorial,isfinite,nextafter,inf

@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal
    def __post_init__(self):
        if not self.lo.is_finite() or not self.hi.is_finite() or self.lo>self.hi:
            raise ValueError('finite ordered interval required')
    def floats(self):
        # Do not trust binary conversion direction; one extra ulp is harmless.
        lo=float(self.lo);hi=float(self.hi)
        if not isfinite(lo) or not isfinite(hi):raise OverflowError('float output overflow')
        if Decimal.from_float(lo)>self.lo:lo=nextafter(lo,-inf)
        if Decimal.from_float(hi)<self.hi:hi=nextafter(hi,inf)
        if not isfinite(lo) or not isfinite(hi):raise OverflowError('outward float endpoint overflow')
        return lo,hi

def exact(value):
    if isinstance(value,Decimal):return value
    if isinstance(value,float):return Decimal.from_float(value)
    if isinstance(value,int):return Decimal(value)
    if isinstance(value,str):return Decimal(value)
    raise TypeError('exact scalar expects int, float, str or Decimal')

def polynomial(nu):
    """Coefficients of P_nu(k,z), indexed by powers of k,z, exactly rational."""
    p={(0,0):Fraction(2)}
    for n in range(1,nu):
        new=defaultdict(Fraction)
        for (j,l),c in p.items():
            new[j+1,l]+=c/n
            new[j,l]+=c*(n+2*l)/n
            new[j,l+1]+=c*(n+l)/n
        p=dict(new)
    assert all(c>0 for c in p.values())
    return p

def eulerian(n):
    if n==0:return ()
    row=[1]
    for m in range(2,n+1):
        row=[(m-j)*(row[j-1] if j else 0)+(j+1)*(row[j] if j<len(row) else 0) for j in range(m)]
    return tuple(row)

def atan_fraction(q,n):
    total=Fraction(0)
    for j in range(n):total+=Fraction((-1)**j,(2*j+1)*q**(2*j+1))
    other=total+Fraction((-1)**n,(2*n+1)*q**(2*n+1))
    return min(total,other),max(total,other)

class Directed:
    def __init__(self,precision=50):
        if precision<20:raise ValueError('precision>=20 required')
        self.precision=precision;self.counts=Counter()
        self.down=Context(prec=precision,rounding=ROUND_FLOOR)
        self.up=Context(prec=precision,rounding=ROUND_CEILING)
        for context in (self.down,self.up):
            for signal in (InvalidOperation,DivisionByZero,Overflow,Underflow,Clamped):context.traps[signal]=True
    def point(self,value):
        x=exact(value);return Interval(x,x)
    def rational(self,value):
        return self.div(self.point(value.numerator),self.point(value.denominator))
    def add(self,a,b):
        self.counts['directed_add']+=2
        return Interval(self.down.add(a.lo,b.lo),self.up.add(a.hi,b.hi))
    def sub(self,a,b):
        self.counts['directed_subtract']+=2
        return Interval(self.down.subtract(a.lo,b.hi),self.up.subtract(a.hi,b.lo))
    def mul(self,a,b):
        if min(a.lo,b.lo)<0:raise ValueError('positive multiplication only')
        self.counts['directed_multiply']+=2
        return Interval(self.down.multiply(a.lo,b.lo),self.up.multiply(a.hi,b.hi))
    def div(self,a,b):
        if a.lo<0 or b.lo<=0:raise ValueError('positive division only')
        self.counts['directed_divide']+=2
        return Interval(self.down.divide(a.lo,b.hi),self.up.divide(a.hi,b.lo))
    def sqrt(self,a):
        if a.lo<0:raise ValueError('nonnegative square root required')
        self.counts['correctly_rounded_sqrt']+=2;self.counts['decimal_adjacent']+=2
        lo=self.down.sqrt(a.lo);hi=self.up.sqrt(a.hi)
        return Interval(self.down.next_minus(lo) if lo else lo,self.up.next_plus(hi))
    def power(self,a,n):
        if not isinstance(n,int) or n<0:raise ValueError('nonnegative integer exponent required')
        result=self.point(1)
        while n:
            if n&1:result=self.mul(result,a)
            n//=2
            if n:a=self.mul(a,a)
        return result
    def pi(self):
        n=self.precision+10;a=atan_fraction(5,n);b=atan_fraction(239,n)
        self.counts['rational_atan_series_terms']+=2*(n+1)
        lo=16*a[0]-4*b[1];hi=16*a[1]-4*b[0]
        return Interval(self.rational(lo).lo,self.rational(hi).hi)
    def tail(self,a,b,nu,degree):
        if not isinstance(nu,int) or not 1<=nu<=16:raise ValueError('integer nu1..16 required')
        if not isinstance(degree,int) or degree<0:raise ValueError('nonnegative integer degree required')
        a=self.point(a);b=self.point(b)
        if a.lo<=0 or b.lo<a.lo:raise ValueError('0<a<=b required')
        if a.lo==b.lo:return self.point(0)
        one=self.point(1);two=self.point(2);N=degree+1
        s=self.sqrt(self.mul(a,b));width=self.sub(b,a)
        denom=self.add(self.add(a,b),self.mul(two,s));r=self.div(width,denom)
        z=self.div(self.power(width,2),self.mul(self.mul(two,s),denom))
        omr=self.sub(one,r)
        if omr.lo<=0:raise ArithmeticError('precision cannot resolve 1-r; increase explicitly')
        rN=self.power(r,N);p=polynomial(nu);self.counts['rational_polynomial_terms']+=len(p)
        rp=[self.power(r,i) for i in range(nu)]
        zp=[self.power(z,i) for i in range(nu)]
        geometric=[]
        for j in range(nu):
            if j==0:g=self.div(one,omr)
            else:
                num=self.point(0)
                for l,c in enumerate(eulerian(j)):num=self.add(num,self.mul(self.point(c),rp[l+1]))
                g=self.div(num,self.power(omr,j+1))
            geometric.append(g)
        shifted=[]
        for j in range(nu):
            value=self.point(0)
            for l in range(j+1):value=self.add(value,self.mul(self.point(comb(j,l)*N**(j-l)),geometric[l]))
            shifted.append(value)
        value=self.point(0)
        for (j,l),c in p.items():value=self.add(value,self.mul(self.rational(c),self.mul(zp[l],shifted[j])))
        return self.div(self.mul(rN,value),self.power(s,nu))
    def simplex_bound(self,a,b,nu,degree,determinant):
        tail=self.tail(a,b,nu,degree)
        det=determinant if isinstance(determinant,Interval) else self.point(determinant)
        if det.lo<0:raise ValueError('absolute determinant enclosure required')
        num=self.mul(self.mul(det,tail),self.point(factorial(nu-1)))
        den=self.mul(self.power(self.pi(),nu),self.point(factorial(2*nu-1)))
        return self.div(num,den)
    def sum(self,intervals):
        result=self.point(0)
        for value in intervals:result=self.add(result,value)
        return result
