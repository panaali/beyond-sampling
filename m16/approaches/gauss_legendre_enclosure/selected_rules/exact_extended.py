"""Exact, bounded-degree Legendre root and Gaussian-weight certificates."""
from fractions import Fraction as F
from collections import Counter

ROOT_BITS=160
WEIGHT_BITS=128

class Arithmetic:
    def __init__(self):self.count=Counter();self.peak=[0,0]
    def see(self,x):
        self.peak[0]=max(self.peak[0],abs(x.numerator).bit_length())
        self.peak[1]=max(self.peak[1],x.denominator.bit_length());return x
    def op(self,name,x,y):
        self.count['rational_'+name]+=1
        return self.see({'add':lambda:x+y,'sub':lambda:x-y,'mul':lambda:x*y,'div':lambda:x/y}[name]())
    def compare(self,name,x,y):
        self.count['comparison_'+name]+=1
        return {'lt':lambda:x<y,'le':lambda:x<=y,'eq':lambda:x==y}[name]()
    def sign(self,x):
        if self.compare('lt',x,F(0)):return -1
        if self.compare('eq',x,F(0)):return 0
        return 1

def sequence(n,x,A):
    out=[F(1),x]
    if n==0:return out[:1]
    for k in range(2,n+1):
        t=A.op('mul',A.op('mul',F(2*k-1),x),out[-1])
        out.append(A.op('div',A.op('sub',t,A.op('mul',F(k-1),out[-2])),F(k)))
    return out

def signs_and_variations(n,x,A):
    seq=sequence(n,x,A);signs=[A.sign(v) for v in seq];nz=[s for s in signs if s]
    A.count['integer_sign_comparisons']+=max(len(nz)-1,0)
    return signs,sum(a!=b for a,b in zip(nz,nz[1:]))

def isolate(n,A,bits=ROOT_BITS):
    assert n in [2,6,10,12,16,22,28,30] and bits>0
    _,vl=signs_and_variations(n,F(0),A);_,vr=signs_and_variations(n,F(1),A)
    stack=[(F(0),F(1),0,vl,vr)];roots=[];unresolved=[];visited=0
    while stack:
        lo,hi,depth,vl,vr=stack.pop();visited+=1;count=vl-vr;A.count['integer_count_subtractions']+=1
        if count==0:continue
        if depth==bits:
            if count==1:roots.append((lo,hi))
            else:unresolved.append(dict(reason='multiple_roots_at_fixed_precision',count=count,lo=lo,hi=hi))
            continue
        mid=A.op('div',A.op('add',lo,hi),F(2));signs,vm=signs_and_variations(n,mid,A)
        if signs[-1]==0:
            # Do not silently perturb a rational root or endpoint convention.
            unresolved.append(dict(reason='exact_midpoint_root',count=count,lo=lo,hi=hi,mid=mid));continue
        stack.extend([(mid,hi,depth+1,vm,vr),(lo,mid,depth+1,vl,vm)])
    return dict(positive_roots=sorted(roots),unresolved=unresolved,visited=visited,expected=n//2)

def coefficients(n,A):
    p0=[F(1)];p1=[F(0),F(1)]
    if n==0:return p0
    for k in range(2,n+1):
        p=[F(0)]*(k+1)
        for i,c in enumerate(p1):p[i+1]=A.op('mul',F(2*k-1),c)
        for i,c in enumerate(p0):p[i]=A.op('sub',p[i],A.op('mul',F(k-1),c))
        p=[A.op('div',c,F(k)) for c in p];p0,p1=p1,p
    return p1

def imul(x,y,A):
    products=[A.op('mul',a,b) for a in x for b in y]
    A.count['interval_extremum_comparisons']+=6
    return min(products),max(products)

def square(x,A):
    lo,hi=x
    a,b=A.op('mul',lo,lo),A.op('mul',hi,hi)
    A.count['interval_extremum_comparisons']+=2
    if A.compare('le',lo,F(0)) and A.compare('le',F(0),hi):return F(0),max(a,b)
    return min(a,b),max(a,b)

def horner(coeff,x,A):
    v=(F(0),F(0))
    for c in reversed(coeff):
        v=imul(v,x,A);v=(A.op('add',v[0],c),A.op('add',v[1],c))
    return v

def outward_dyadic(x,A):
    lo,hi=x
    q,r=divmod(lo.numerator<<WEIGHT_BITS,lo.denominator)
    qh,rh=divmod(hi.numerator<<WEIGHT_BITS,hi.denominator)
    A.count.update(integer_shift=2,integer_divmod=2,integer_comparison=1,integer_add=1)
    return F(q,1<<WEIGHT_BITS),F(qh+int(rh!=0),1<<WEIGHT_BITS)

def weight_interval(n,root,A):
    c=coefficients(n,A)
    derivative=[A.op('mul',F(i),c[i]) for i in range(1,len(c))]
    dp=horner(derivative,root,A);dp2=square(dp,A);x2=square(root,A)
    one_minus=(A.op('sub',F(1),x2[1]),A.op('sub',F(1),x2[0]))
    den=imul(one_minus,dp2,A)
    if not A.compare('lt',F(0),den[0]):return dict(status='unknown_nonpositive_denominator',derivative=dp,denominator=den)
    raw=(A.op('div',F(2),den[1]),A.op('div',F(2),den[0]))
    return dict(status='bounded',derivative=dp,denominator=den,raw=raw,dyadic=outward_dyadic(raw,A))

def positive_power(x,k,A):
    assert x[0]>=0 and k>=0
    lo=hi=F(1)
    for _ in range(k):lo=A.op('mul',lo,x[0]);hi=A.op('mul',hi,x[1])
    return lo,hi

def moments(n,roots,weights,A):
    rows=[]
    for k in range(2*n):
        if k%2:
            rows.append(dict(degree=k,interval=(F(0),F(0)),truth=F(0),contains=True,reason='exact reflection cancellation'));continue
        lo=hi=F(0)
        for root,weight in zip(roots,weights):
            term=imul(positive_power(root,k,A),weight,A)
            lo=A.op('add',lo,term[0]);hi=A.op('add',hi,term[1])
        lo=A.op('mul',F(2),lo);hi=A.op('mul',F(2),hi);truth=F(2,k+1)
        contains=A.compare('le',lo,truth) and A.compare('le',truth,hi)
        rows.append(dict(degree=k,interval=(lo,hi),truth=truth,contains=contains))
    return rows

def pack(x):
    if isinstance(x,F):return [str(x.numerator),str(x.denominator)]
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
