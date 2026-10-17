"""Exact-rational uniform-state reset coefficient and overlap-aware masses."""
from fractions import Fraction as F
import math

def rational(value):return value if isinstance(value,F) else F.from_float(float(value))

def coefficient(wi,wh):
    w1,w2=map(rational,wi);a,b=-rational(wh[0][0]),-rational(wh[0][1]);c,d=rational(wh[1][0]),-rational(wh[1][1])
    if min(w1,w2)<=0 or min(a,b,c,d)<0:raise ValueError('Reset theorem sign assumptions unsupported')
    gamma=c*w1 if d==0 else max(F(0),c*w1-d*w2,c*max(w1-b*w2/d,F(0)))
    return dict(kappa=gamma/w2,gamma=gamma,parameters=(w1,w2,a,b,c,d),
        rational_operation_proxy=30,max_integer_bits=max(abs(x.numerator).bit_length()+x.denominator.bit_length() for x in (w1,w2,a,b,c,d,gamma,gamma/w2)))

def masses(kappa,T):
    if T<1:raise ValueError('Positive integer length required')
    if kappa<0:raise ValueError('Nonnegative reset slope required')
    k=float(kappa)
    # Complement form retains the improvement over two-negative resets at large k.
    q=math.atan(k)/(2*math.pi) if k<=1 else .25-math.atan(1/k)/(2*math.pi)
    r=.5-q;s=[1.,1.];u=[1.,.5]
    for n in range(2,T+1):s.append(.5*s[-1]+q*s[-2]);u.append(.5*u[-1]+q*u[-2])
    last={j:r*u[T-j] for j in range(2,T+1)}
    return dict(pair_reset_mass=r,q=q,no_reset_mass=s[T],last_reset_masses=last,
        total_mass=s[T]+sum(last.values()),scalar_operation_proxy=50+8*T,
        special_functions={'atan':1},roundoff_certified=False)
