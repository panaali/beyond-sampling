"""Rank integration given a fixed surviving suffix; reset mass is unnormalized."""
import math
from scipy.integrate import quad
from scipy.special import ndtr
from .rank import phi,pair_rank,prefix_counts

def conditional_rank(kappa,older_count,suffix,label):
    k=float(kappa);N=older_count;suffix=tuple(map(float,suffix));T=N+2+len(suffix)
    if not isinstance(N,int) or N<0 or not isinstance(label,int) or not 0<=label<T:raise ValueError('Invalid dimensions/label')
    if math.isnan(k) or k<0 or any(not math.isfinite(x) for x in suffix) or len(set(suffix))!=len(suffix):raise ValueError('Invalid slope or tied/nonfinite fixed suffix')
    stats=dict(outer_callbacks=0,pair_callbacks=0,quad_error_estimate=0.,inner_error_estimate_max=0.,tail_allowance=0.,roundoff_certified=False)
    def binomial(n,c,t):
        if c<0 or c>n:return 0.
        return math.comb(n,c)*float(ndtr(-t))**c*float(ndtr(t))**(n-c)
    if label>=N+2:
        t=suffix[label-N-2];wanted=1-sum(x>t for x in suffix);g,s=prefix_counts(k,t,N);stats['pair_callbacks']=s['callbacks'];stats['inner_error_estimate_max']=s['quadrature_error_estimate']+s['tail_allowance'];return (g[wanted] if 0<=wanted<len(g) else 0.),stats
    def f(t):
        stats['outer_callbacks']+=1;wanted=1-sum(x>t for x in suffix)
        if wanted<0:return 0.
        if label<N:
            g,s=pair_rank(k,t);stats['pair_callbacks']+=s['callbacks'];stats['inner_error_estimate_max']=max(stats['inner_error_estimate_max'],s['quadrature_error_estimate']+s['tail_allowance'])
            return phi(t)*sum(g[l]*binomial(N-1,wanted-l,t) for l in range(3))
        if label==N:
            b=0. if t<=0 or k==0 else -k*t
        else:
            if t>0:return 0.
            b=math.inf if k==0 else (0. if math.isinf(k) else -t/k)
        below=float(ndtr(min(t,b)));above=max(0.,float(ndtr(b))-float(ndtr(t)))
        return phi(t)*(below*binomial(N,wanted,t)+above*binomial(N,wanted-1,t))
    cuts=sorted(set([-8.,-3.,-1.,0.,1.,3.,8.]+[x for x in suffix if -8<x<8]));total=0.
    for a,b in zip(cuts,cuts[1:]):
        value,error=quad(f,a,b,epsabs=2e-12,epsrel=2e-12,limit=100);total+=value;stats['quad_error_estimate']+=error
    stats['tail_allowance']=2*float(ndtr(-8.))
    return total,stats
