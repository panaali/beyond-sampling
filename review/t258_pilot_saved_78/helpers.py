"""Exact unchanged scalar helpers from independently reviewed0427df32."""
from fractions import Fraction as F
import math,struct

def design(row,hrow,order,t,f):
    # Independent scalar polynomial cell ID and column construction. Sorting
    # itself shares the explicitly pinned NumPy quicksort primitive.
    ranked=[row[k] for k in order]+[0.]*(10-t)
    ages=[t-1-k for k in order]+[t]*(10-t)
    nr=[0,0,1,2,3,3,4,4,3,3,3,2][f]
    code=sum(ages[k]*(t+1)**(nr-k-1) for k in range(nr))
    if f>=5:code=4*code+min(sum(v>0 for v in row[:t]),3)
    take=3 if f<4 else 10 if f==7 else 5
    v=[row[t-1]]+ranked[:take]
    if f==1:
        theta=2*math.pi/8.42
        for k in range(3):v.extend([ranked[k]*math.cos(theta*ages[k]),ranked[k]*math.sin(theta*ages[k])])
    if f in (5,6,8,9,10,11):
        tail=[F.from_float(z) for z in ranked[5:t]]
        v.extend([float(sum((z for z in tail if z>0),F(0))),float(sum((z for z in tail if z<0),F(0)))])
    if f>=8:
        lag={8:1,9:3,10:9,11:9}[f]
        for s in range(t-1,max(0,t-lag-1),-1):v.extend(hrow[s])
    return code,v

def nearest_even(q):
    n,r=divmod(q.numerator,q.denominator)
    return n+int(2*r>q.denominator or (2*r==q.denominator and n%2))

def binary32(v):
    # Integer round-to-nearest-even, including subnormals. No NumPy/struct cast
    # from the input float. struct only decodes the independently assembled bits.
    sign=struct.unpack('>Q',struct.pack('>d',v))[0]>>63;q=abs(F.from_float(v))
    if not q:bits=sign<<31
    else:
        e=q.numerator.bit_length()-q.denominator.bit_length()
        two=lambda n:F(2**n) if n>=0 else F(1,2**(-n))
        if q<two(e):e-=1
        if e < -126:
            m=nearest_even(q*2**149);assert m<=2**23;bits=(sign<<31)+m
        else:
            m=nearest_even(q/two(e-23))
            if m==2**24:m//=2;e+=1
            assert -126<=e<=127 and 2**23<=m<2**24
            bits=(sign<<31)+((e+127)<<23)+(m-2**23)
    return float(struct.unpack('>f',struct.pack('>I',bits))[0])
