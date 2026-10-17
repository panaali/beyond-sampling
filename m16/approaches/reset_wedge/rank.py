"""Positive pair rank masses; scalar numerical primitive, not accuracy estimator."""
import math
from scipy.integrate import quad
from scipy.special import ndtr

PHI0=1/math.sqrt(2*math.pi)

def phi(x):return PHI0*math.exp(-.5*x*x)

def pair_rank(kappa,t):
    k,t=float(kappa),float(t)
    if math.isnan(k) or k<0 or not math.isfinite(t):raise ValueError('Nonnegative slope and finite threshold required')
    stats=dict(callbacks=0,quadrature_error_estimate=0.,tail_allowance=0.,roundoff_certified=False)
    def integral(fn,a,b):
        # Positive integrands bounded by phi(x); split to avoid missing a narrow
        # relevant interval when the formal endpoint -t/k is enormous.
        b=min(b,8.)
        if a>=b:
            if a>=8:stats['tail_allowance']+=float(ndtr(-a))
            return 0.
        if b==8:stats['tail_allowance']+=float(ndtr(-8.))
        total=0.
        cuts=sorted(set([a,b]+[v for v in (1.,3.) if a<v<b]))
        def counted(x):stats['callbacks']+=1;return fn(x)
        for lo,hi in zip(cuts,cuts[1:]):
            value,error=quad(counted,lo,hi,epsabs=2e-13,epsrel=2e-13,limit=100)
            total+=value;stats['quadrature_error_estimate']+=error
        return total
    u=float(ndtr(t))
    if k==0:
        v=float(ndtr(min(t,0.)));above=float(ndtr(-t));rest=.5-v
        g=(u*v,above*v+u*rest,above*rest)
    elif math.isinf(k):
        v=float(ndtr(min(t,0.)));rest=.5-v;g=(v*v,2*v*rest,rest*rest)
    elif t>=0:
        f=lambda x:phi(x)*float(ndtr(-k*x))
        g=(.25+integral(f,0,t),integral(f,t,math.inf),0.)
    else:
        a=-t/k;v=.5-u
        g2=v*v+integral(lambda x:phi(x)*max(0.,float(ndtr(-k*x))-u),0,a)
        g1=2*u*v+u*(float(ndtr(a))-.5)+integral(lambda x:phi(x)*float(ndtr(-k*x)),a,math.inf)
        g=(u*u,g1,g2)
    return g,stats

def prefix_counts(kappa,t,older_count):
    if not isinstance(older_count,int) or older_count<0:raise ValueError('Nonnegative integer older count required')
    g,stats=pair_rank(kappa,t);u=float(ndtr(t));v=float(ndtr(-t));out=[0.]*(older_count+3)
    for n in range(older_count+1):
        b=math.comb(older_count,n)*v**n*u**(older_count-n)
        for l in range(3):out[n+l]+=b*g[l]
    stats['binomial_terms']=older_count+1
    return out,stats
