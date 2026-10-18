"""A47-scalar determinant, with optional separately counted outward intervals."""
import math
PAIRS=((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))

class Counts:
    def __init__(self):
        self.arithmetic=0;self.comparisons=0;self.finite_checks=0;self.nextafter=0
    def dictionary(self):return vars(self).copy()

class Point:
    def __init__(self,value,counts):self.value=float(value);self.counts=counts
    def __add__(self,other):
        self.counts.arithmetic+=1
        return Point(self.value+other.value,self.counts)
    def __sub__(self,other):
        self.counts.arithmetic+=1
        return Point(self.value-other.value,self.counts)
    def __mul__(self,other):
        self.counts.arithmetic+=1
        return Point(self.value*other.value,self.counts)

class Interval:
    def __init__(self,lo,hi,counts):
        self.lo=float(lo);self.hi=float(hi);self.counts=counts
        counts.comparisons+=1
        if not self.lo<=self.hi:raise ArithmeticError('unordered interval')
        for x in (self.lo,self.hi):
            counts.finite_checks+=1
            if not math.isfinite(x):raise ArithmeticError('nonfinite interval endpoint')
    def outward(self,lo,hi):
        c=self.counts
        for x in (lo,hi):
            c.finite_checks+=1
            if not math.isfinite(x):raise ArithmeticError('nonfinite rounded operation')
        c.nextafter+=2
        return Interval(math.nextafter(lo,-math.inf),math.nextafter(hi,math.inf),c)
    def __add__(self,other):
        self.counts.arithmetic+=2
        return self.outward(self.lo+other.lo,self.hi+other.hi)
    def __sub__(self,other):
        self.counts.arithmetic+=2
        return self.outward(self.lo-other.hi,self.hi-other.lo)
    def __mul__(self,other):
        c=self.counts;c.arithmetic+=4
        values=(self.lo*other.lo,self.lo*other.hi,self.hi*other.lo,self.hi*other.hi)
        for x in values:
            c.finite_checks+=1
            if not math.isfinite(x):raise ArithmeticError('nonfinite endpoint product')
        lo=hi=values[0]
        for v in values[1:]:
            c.comparisons+=2
            if v<lo:lo=v
            if v>hi:hi=v
        return self.outward(lo,hi)

def determinant47(A):
    s=[A[0][i]*A[1][j]-A[0][j]*A[1][i] for i,j in PAIRS]
    c=[A[2][i]*A[3][j]-A[2][j]*A[3][i] for i,j in PAIRS]
    terms=[s[i]*c[5-i] for i in range(6)]
    return terms[0]-terms[1]+terms[2]+terms[3]-terms[4]+terms[5]

def point_matrix(A,counts):return [[Point(v,counts) for v in row] for row in A]
def interval_matrix(A,counts):return [[Interval(v,v,counts) for v in row] for row in A]
