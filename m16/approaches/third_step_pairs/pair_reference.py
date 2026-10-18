"""Independent cylindrical raw ReLU-pair integral; no cone mass or facet code."""
from fractions import Fraction as F
from collections import Counter
import mpmath as mp

def q(x):return mp.mpf(x.numerator)/x.denominator

def integrate(sector,v,w,dps=80):
    v=tuple(map(F,v));w=tuple(map(F,w));lo=sector['lo'];hi=sector['hi']
    if not any(v) or not any(w):return dict(value='0.0',evaluations=0,status='zero_product',domains={})
    # Exact opposite-proportional products vanish; independent of cone rank.
    cross=[v[j]*w[k]-v[k]*w[j] for j,k in [(1,2),(2,0),(0,1)]]
    if not any(cross) and sum(a*b for a,b in zip(v,w))<0:
        return dict(value='0.0',evaluations=0,status='opposite_product',domains={})
    rays={}
    # Exact zero/threshold harmonic coefficients; only interior halfplane rays.
    for a,b in [(v[0],v[1]),(w[0],w[1]),(v[2]*w[0]-w[2]*v[0],v[2]*w[1]-w[2]*v[1])]:
        if not b:continue # cosine-only zeros occur at halfplane endpoints.
        ray=(b,-a) if b>0 else (-b,a)
        if lo[0]*ray[1]-lo[1]*ray[0]>0 and ray[0]*hi[1]-ray[1]*hi[0]>0:rays[ray[1]/ray[0]]=ray
    cuts=[lo]+[rays[s] for s in sorted(rays)]+[hi]
    with mp.workdps(dps):
        angles=[mp.atan2(q(r[1]),q(r[0])) for r in cuts];assert all(a<b for a,b in zip(angles,angles[1:]))
        b=q(v[2]);d=q(w[2]);calls=0;domains=Counter()
        def primitive(t):
            if mp.isinf(t):
                sign=mp.mpf(1) if t>0 else mp.mpf(-1)
                return (2*sign/3,mp.mpf(0),sign/3)
            root=mp.sqrt(1+t*t);s=t/root
            return (s-s**3/3,-1/(3*root**3),s**3/3)
        def f(theta):
            nonlocal calls
            calls+=1;co=mp.cos(theta);si=mp.sin(theta)
            a=q(v[0])*co+q(v[1])*si;c=q(w[0])*co+q(w[1])*si
            lower=-mp.inf;upper=mp.inf
            for intercept,slope in [(a,b),(c,d)]:
                if slope>0:lower=max(lower,-intercept/slope)
                elif slope<0:upper=min(upper,-intercept/slope)
                elif intercept<=0:domains['constant_nonpositive']+=1;return mp.mpf(0)
                else:domains['constant_positive']+=1
            if lower>=upper:domains['empty_or_tied']+=1;return mp.mpf(0)
            domains['unbounded' if mp.isinf(lower) or mp.isinf(upper) else 'bounded']+=1
            left=primitive(lower);right=primitive(upper);delta=[y-x for x,y in zip(left,right)]
            return a*c*delta[0]+(a*d+c*b)*delta[1]+b*d*delta[2]
        result=3*mp.quad(f,angles)/(4*mp.pi)
        return dict(value=mp.nstr(result,dps),evaluations=calls,status='integrated',domains=dict(domains),
                    exact_cuts=cuts,angles=[mp.nstr(t,dps) for t in angles])
