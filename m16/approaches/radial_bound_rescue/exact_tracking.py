"""Exact Fraction operation inventory; not a scalar-FLOP conversion."""
from fractions import Fraction
from collections import Counter
import operator

class Tracker:
    def __init__(self):
        self.counts=Counter();self.max_numerator_bits=0;self.max_denominator_bits=0
    def observe(self,v):
        self.max_numerator_bits=max(self.max_numerator_bits,abs(v.numerator).bit_length())
        self.max_denominator_bits=max(self.max_denominator_bits,v.denominator.bit_length())
    def report(self):return dict(operations=dict(self.counts),max_numerator_bits=self.max_numerator_bits,max_denominator_bits=self.max_denominator_bits,big_integer_internal_cost_unmeasured=True)

class Tracked:
    current=None
    def __init__(self,v):
        self.value=v.value if isinstance(v,Tracked) else Fraction(v)
        self.current.counts['construct_'+type(v).__name__]+=1;self.current.observe(self.value)
    def other(self,x):return x.value if isinstance(x,Tracked) else Fraction(x)
    def op(self,name,x):
        y=self.other(x);self.current.observe(y);self.current.counts[name]+=1
        return Tracked(getattr(operator,name)(self.value,y))
    def compare(self,name,x):
        y=self.other(x);self.current.observe(y);self.current.counts['compare_'+name]+=1
        return getattr(operator,name)(self.value,y)
    def __add__(self,x):return self.op('add',x)
    __radd__=__add__
    def __sub__(self,x):return self.op('sub',x)
    def __rsub__(self,x):return Tracked(x).op('sub',self)
    def __mul__(self,x):return self.op('mul',x)
    __rmul__=__mul__
    def __truediv__(self,x):return self.op('truediv',x)
    def __rtruediv__(self,x):return Tracked(x).op('truediv',self)
    def __neg__(self):self.current.counts['neg']+=1;return Tracked(-self.value)
    def __abs__(self):self.current.counts['abs']+=1;return Tracked(abs(self.value))
    def __bool__(self):self.current.counts['bool']+=1;return bool(self.value)
    def __eq__(self,x):return self.compare('eq',x)
    def __lt__(self,x):return self.compare('lt',x)
    def __le__(self,x):return self.compare('le',x)
    def __gt__(self,x):return self.compare('gt',x)
    def __ge__(self,x):return self.compare('ge',x)
    def __str__(self):return str(self.value)

def acute_certificate(R,radial):
    A=[[Tracked(float(v)) for v in row] for row in R]
    G=[[sum(x*y for x,y in zip(a,b)) for b in A] for a in A]
    acute=all(G[i][j]>=0 for i in range(4) for j in range(i+1,4))
    norms=[G[i][i] for i in range(4)];det=radial.determinant(A)
    return dict(a=str(min(norms)/4),b=str(max(norms)),det=str(det),gram=[[str(x) for x in row] for row in G],acute=acute)
