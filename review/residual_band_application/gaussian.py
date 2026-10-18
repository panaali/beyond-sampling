"""Gaussian interval masses in δ, with standard-normal coordinate z=sqrt(2)δ.
Floating evaluations are diagnostics, not outward enclosures.
"""
import math

def mass(a,b):
    a=float(a);b=float(b)
    if not math.isfinite(a+b) or not a<=b:raise ValueError('finite ordered endpoints')
    if a>=0:return (math.erfc(a)-math.erfc(b))/2
    if b<=0:return (math.erfc(-b)-math.erfc(-a))/2
    return (math.erf(b)-math.erf(a))/2

def correction(bands):
    first=[];second=[];exact=[];bound1=[];bound2=[];detail=[]
    for aa,bb,sign in bands:
        if sign not in (-1,1) or not aa<bb or aa<0<bb:raise ValueError('one-sided signed band')
        # Form displacement exactly before conversion; avoid subtracting rounded endpoints.
        c=aa if aa>=0 else bb;delta=bb-aa if aa>=0 else aa-bb;q=1 if delta>0 else -1
        cf=float(c);df=float(delta);psi=math.exp(-cf*cf)/math.sqrt(math.pi)
        one=q*psi*df;two=q*psi*(df-cf*df*df);value=mass(aa,bb)
        e1=math.exp(-.5)/math.sqrt(2*math.pi)*df*df
        e2=abs(df)**3/(3*math.sqrt(math.pi))
        first.append(sign*one);second.append(sign*two);exact.append(sign*value);bound1.append(e1);bound2.append(e2)
        detail.append({'first':one,'second':two,'mass':value,'remainder1':e1,'remainder2':e2})
    return {'first':math.fsum(first),'second':math.fsum(second),'direct':math.fsum(exact),'remainder1':math.fsum(bound1),'remainder2':math.fsum(bound2),'detail':detail}
