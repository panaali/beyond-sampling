"""Two terminal last-reset strata, with independent order-statistic references."""
import math
from fractions import Fraction as F
from scipy.integrate import quad
from scipy.special import ndtr,betaincc,beta
from .rank import pair_rank,phi
from .reset import masses

class Integrals:
    def __init__(self):self.stats=dict(callbacks=0,rank_callbacks=0,quad_error_estimate=0.,inner_error_estimate_max=0.,tail_allowance=0.,roundoff_certified=False)
    def integrate(self,f,lo=-8.,hi=8.):
        self.stats['tail_allowance']+=float(ndtr(-8.))*(int(lo==-8.)+int(hi==8.))
        def counted(x):self.stats['callbacks']+=1;return f(x)
        cuts=sorted(set([lo,hi]+[x for x in (-3.,-1.,0.,1.,3.) if lo<x<hi]));s=0.
        for a,b in zip(cuts,cuts[1:]):
            v,e=quad(counted,a,b,epsabs=2e-12,epsrel=2e-12,limit=100);s+=v;self.stats['quad_error_estimate']+=e
        return s
    def rank(self,k,t):
        g,s=pair_rank(k,t);self.stats['rank_callbacks']+=s['callbacks'];self.stats['inner_error_estimate_max']=max(self.stats['inner_error_estimate_max'],s['quadrature_error_estimate']+s['tail_allowance']);return g

def ray_label(wi,wo):
    q=lambda x:F.from_float(float(x))
    scores=[sum(q(a)*q(b) for a,b in zip(row,wi)) for row in wo]
    return max(range(len(scores)),key=lambda i:scores[i])

def calculate(wi,wo,k):
    T=len(wo);N=T-2
    if N<2:raise ValueError('This bounded terminal implementation requires T>=4')
    k=float(k);r=masses(k,T)['pair_reset_mass'];a=Integrals()
    def pair_labeled(t,label):
        u=float(ndtr(t));v=float(ndtr(-t))
        if label=='X':b=-k*max(t,0.)
        else:
            if t>0:return 0.
            b=math.inf if k==0 else -t/k
        below=float(ndtr(min(t,b)));above=max(0.,float(ndtr(b))-u)
        return phi(t)*(below*N*v*u**(N-1)+above*u**N)
    px=a.integrate(lambda t:pair_labeled(t,'X'));py=a.integrate(lambda t:pair_labeled(t,'Y'),hi=0.)
    final_pair_identity=(r-px-py)/N
    def marker(t):
        u=float(ndtr(t));v=float(ndtr(-t));g=a.rank(k,t)
        return phi(t)*((N-1)*v*u**(N-2)*g[0]+u**(N-1)*g[1])
    final_pair_marker=a.integrate(marker)
    label=ray_label(wi,wo);out=dict(T=T,ray_label=label,last_T=dict(estimate=final_pair_marker,independent_identity=final_pair_identity,pair_X_second=px,pair_Y_second=py,event_mass=r),stats=a.stats)
    if label!=T-1:
        out['last_T_minus_1']=dict(unsupported='One-step output is not the final input label');return out
    n=T-3
    def suffix(v):
        u=float(ndtr(v));g=a.rank(k,v)
        return phi(v)*(n*float(ndtr(-v))*u**(n-1)*g[0]+u**n*g[1])
    one=a.integrate(suffix,lo=0.)
    # Reverse integration order, no pair_rank calls. J is a positive beta tail.
    J=lambda u:n*float(beta(n,2))*float(betaincc(n,2,u))
    def reverse(x):
        u=float(ndtr(x));inner=(u**(n+1)-.5**(n+1))/(n+1)+J(u)
        return phi(x)*float(ndtr(-k*x))*inner
    rev=.25*J(.5)+a.integrate(reverse,lo=0.)
    out['last_T_minus_1']=dict(estimate=one,independent_reversed_beta=rev,event_mass=.5*r)
    out['partial_accuracy']=final_pair_marker+one
    out['unresolved_event_mass']=1-1.5*r
    out['formal_float_enclosure']=False
    return out
