"""Exact represented-hull cap bound; conservative resource/domain fallbacks."""
from collections import Counter
from fractions import Fraction as F
import math

class Limit(Exception):pass
class Ledger:
    def __init__(self,bits=8192,steps=128000):
        self.bits=bits;self.steps=steps;self.counts=Counter();self.maxbits=0;self.total=0
    def charge(self,key,n=1):
        self.counts[key]+=n;self.total+=n
        if self.total>self.steps:raise Limit('step_limit')
    def check(self,x):
        q=F(x);b=max(abs(q.numerator).bit_length(),q.denominator.bit_length());self.maxbits=max(self.maxbits,b)
        if b>self.bits:raise Limit('bit_limit')
        return q
    def value(self,x):self.charge('binary64_to_fraction');return self.check(F(x))
    def add(self,a,b):self.charge('rational_add');return self.check(a+b)
    def sub(self,a,b):self.charge('rational_sub');return self.check(a-b)
    def mul(self,a,b):self.charge('rational_mul');return self.check(a*b)
    def div(self,a,b):self.charge('rational_div');return self.check(a/b)
    def dot(self,a,b):
        out=F(0)
        for x,y in zip(a,b,strict=True):out=self.add(out,self.mul(x,y))
        return out
    def report(self):return dict(counts=dict(self.counts),counted_steps=self.total,max_fraction_bits=self.maxbits,bit_limit=self.bits,step_limit=self.steps)

def upper_sqrt(q,L,k=40):
    assert 0<=q<=1 and k==40
    L.charge('integer_shift');scaled=q.numerator<<(2*k)
    L.check(scaled)
    L.charge('integer_divmod');n,rem=divmod(scaled,q.denominator)
    L.charge('integer_compare');n+=int(rem!=0)
    L.charge('integer_isqrt');m=math.isqrt(n)
    L.charge('integer_square_compare');m+=int(m*m<n)
    L.charge('rational_construct');u=L.check(F(m,1<<k))
    L.charge('rational_square_compare');assert u*u>=q
    return u

def outward_float(q):
    assert isinstance(q,F) and 0<=q<=1
    value=float(q);steps=0
    if not math.isfinite(value):return 1.,{'fallback':'conversion_nonfinite','nextafter':steps}
    if F(value)<q:value=math.nextafter(value,math.inf);steps=1
    if not math.isfinite(value) or not q<=F(value)<=1:return 1.,{'fallback':'conversion_check','nextafter':steps}
    return value,{'fallback':None,'nextafter':steps}

def bound(rays,*,maxbits=8192,maxsteps=128000):
    L=Ledger(maxbits,maxsteps);details={};reason=None;answer=F(1)
    try:
        if not isinstance(rays,(list,tuple)) or not 1<=len(rays)<=64:raise Limit('ray_count_or_type')
        if not isinstance(rays[0],(list,tuple)):raise Limit('row_type')
        d=len(rays[0]);details['dimension']=d;details['rays']=len(rays)
        if d not in (2,3,4,5):raise Limit('unsupported_dimension')
        if any(not isinstance(row,(list,tuple)) or len(row)!=d for row in rays):raise Limit('shape')
        if any(type(v) is not float or not math.isfinite(v) for row in rays for v in row):raise Limit('nonfinite_or_type')
        R=[[L.value(v) for v in row] for row in rays]
        axis=[F(0) for _ in range(d)]
        for row in R:axis=[L.add(x,y) for x,y in zip(axis,row,strict=True)]
        norm=L.dot(axis,axis);L.charge('rational_compare')
        if norm==0:raise Limit('zero_axis')
        q=F(1)
        for row in R:
            n=L.dot(row,row);dot=L.dot(axis,row);L.charge('rational_compare',2)
            if n==0:raise Limit('zero_ray')
            if dot<=0:raise Limit('nonpositive_axis_dot')
            ratio=L.div(L.mul(dot,dot),L.mul(norm,n));L.charge('rational_compare',2)
            if not 0<ratio<=1:raise Limit('cosine_invariant')
            L.charge('rational_compare');q=min(q,ratio)
        delta=L.sub(F(1),q)
        if d==2:answer=L.div(upper_sqrt(delta,L),F(2))
        elif d==3:answer=L.div(delta,F(2))
        elif d==4:answer=L.mul(F(2,3),L.mul(delta,upper_sqrt(delta,L)))
        else:answer=L.mul(F(3,4),L.mul(delta,delta))
        L.charge('rational_compare');answer=min(answer,F(1,2))
        details.update(q=str(q),delta=str(delta),axis=[str(v) for v in axis])
    except Limit as e:answer=F(1);reason=str(e)
    outward,conversion=outward_float(answer)
    assert 0<=answer<=F(outward)<=1
    return dict(bound=str(answer),outward=outward,outward_hex=outward.hex(),fallback=reason,conversion=conversion,details=details,ledger=L.report())
