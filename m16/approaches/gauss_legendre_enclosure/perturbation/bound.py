"""Exact upper bound on original4D Duffy source-constant perturbation."""
from fractions import Fraction as F
from collections import Counter
import math

SQRT_BITS=160
class Ledger:
    def __init__(self):self.count=Counter();self.peak=[0,0]
    def see(self,x):
        self.peak[0]=max(self.peak[0],abs(x.numerator).bit_length());self.peak[1]=max(self.peak[1],x.denominator.bit_length());return x
    def op(self,name,x,y):
        self.count['rational_'+name]+=1
        return self.see({'mul':lambda:x*y,'div':lambda:x/y,'add':lambda:x+y,'sub':lambda:x-y}[name]())
    def comp(self,name,x,y):
        self.count['comparison_'+name]+=1
        return {'lt':lambda:x<y,'le':lambda:x<=y}[name]()

def sqrt_floor(x,L):
    assert x>=0
    shifted=x.numerator<<(2*SQRT_BITS);quotient=shifted//x.denominator;k=math.isqrt(quotient)
    L.count.update(integer_shift=2,integer_divide=1,integer_isqrt=1,integer_add=1)
    den=1<<SQRT_BITS;lo=L.see(F(k,den));hi=L.see(F(k+1,den))
    return lo,hi

def bound(a,M2,det,epsilon,s,delta):
    L=Ledger()
    for x in [a,M2,det,epsilon,s,delta]:L.see(x)
    result=dict(status='unknown',counts=L.count,peak_bits=L.peak)
    if not L.comp('lt',F(0),a):result['status']='unknown_nonpositive_radial_lower';return result
    if any(L.comp('lt',x,F(0)) for x in [M2,epsilon,s,delta]):result['status']='invalid_negative_parameter';return result
    low,high=sqrt_floor(a,L);result['sqrt_interval']=(low,high)
    if not L.comp('lt',F(0),low):result['status']='unknown_fixed_sqrt_precision';return result
    q5=L.op('div',F(1),L.op('mul',L.op('mul',a,a),low));q7=L.op('div',q5,a)
    lipschitz=L.op('add',L.op('mul',F(6),q5),L.op('mul',L.op('mul',F(40),M2),q7))
    s2=L.op('mul',s,s);s3=L.op('mul',s2,s);s4=L.op('mul',s3,s)
    node=L.op('mul',L.op('mul',s4,epsilon),lipschitz)
    polynomial=L.op('add',L.op('add',L.op('add',s3,s2),s),F(1))
    weight=L.op('mul',L.op('mul',delta,polynomial),q5)
    total=L.op('add',node,weight);L.count['rational_abs']+=1
    # pi >3 gives3/(8*pi^2)<1/24. This deliberately conservative rational
    # coefficient avoids treating a rounded pi value as directed arithmetic.
    prefactor=L.op('div',abs(det),F(24));probability=L.op('mul',prefactor,total)
    result.update(status='bounded',q5_upper=q5,q7_upper=q7,derivative_sum_upper=lipschitz,node_upper=node,weight_upper=weight,integral_upper=total,probability_upper=probability,prefactor_upper=prefactor)
    return result

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
