"""Two constrained pairs, all target ranks; positive bounded quadrature."""
import math,time
import numpy as np
from scipy.special import ndtr
from rank_mass import mul,power,cdf_difference
from direct_gaussian import LIMIT,phi

def integrate(T,kappa,k,csign,outer=64,inner=32):
    if T<4 or min(kappa,k)<0 or csign not in ('+','-'):raise ValueError('domain')
    start=time.process_time();x,w=np.polynomial.legendre.leggauss(outer);y,v=np.polynomial.legendre.leggauss(inner);nodes=(y+1)/2;weights=v/2
    stats={'rule_generations':2,'rule_nodes':outer+inner,'outer_nodes':outer,'inner_nodes':0,'cdf_values':0,'exp_values':0}
    def Q(t):stats['cdf_values']+=np.asarray(t).size;return ndtr(-np.asarray(t))
    def delta(hi,lo):stats['cdf_values']+=2*max(np.asarray(hi).size,np.asarray(lo).size);return cdf_difference(hi,lo)
    def integral(lo,hi,fun):
        if hi<=lo:return 0.
        pts=lo+(hi-lo)*nodes;stats['inner_nodes']+=inner;stats['exp_values']+=inner
        return (hi-lo)*float(weights@(phi(pts)*fun(pts)))
    def pair(u,tail,threshold,sign):
        cutoff=u/threshold if threshold else math.inf
        top=min(u,cutoff) if sign=='+' else min(LIMIT,cutoff)
        p0=integral(0,top,lambda z:delta(u,threshold*z))
        if sign=='-':
            p1=integral(0,top,lambda z:np.full_like(z,tail))+integral(top,LIMIT,lambda z:Q(threshold*z))
        else:
            p1=integral(0,min(u,cutoff),lambda z:np.full_like(z,tail))
            p1+=integral(min(u,cutoff),u,lambda z:Q(threshold*z))
            p1+=integral(u,min(cutoff,LIMIT),lambda z:delta(u,threshold*z))
        if min(p0,p1)<0:raise ArithmeticError('negative pair')
        return p0,p1
    total=np.zeros(4)
    for u,wt in zip((x+1)*LIMIT/2,w*LIMIT/2):
        tail=float(Q(u));F=1-tail;ordinary=power(F,tail,T-4);first=pair(u,tail,kappa,'-');second=pair(u,tail,k,csign)
        prefix=mul(mul(power(F,tail,T-5),first),second)[1] if T>4 else 0.
        partnerA=float(delta(u/kappa if kappa else math.inf,0))
        B=mul(ordinary,second)[1]*partnerA
        if csign=='-':C=0.
        else:
            conditional=(float(delta(u,k*u)) if k<=1 else 0.,float(Q(max(u,k*u))))
            C=mul(mul(ordinary,first),conditional)[1]
        cutoff=u/k if k else math.inf
        conditional=(float(delta(cutoff,0)),0.) if csign=='-' else (float(delta(min(u,cutoff),0)),float(delta(cutoff,u)) if cutoff>u else 0.)
        D=mul(mul(ordinary,first),conditional)[1]
        stats['exp_values']+=1;total+=wt*float(phi(u))*np.array([prefix,B,C,D])
    vector=[total[0]]*(T-4)+[0.,total[1],total[2],total[3]];event=math.atan2(1,kappa)*math.atan2(1,k)/(4*math.pi**2)
    return {'T':T,'kappa':kappa,'k':k,'csign':csign,'outer':outer,'inner':inner,'classes':total.tolist(),'joint_by_target':[float(z) for z in vector],'event_mass':event,'partition_error':float(sum(vector)-event),'tail_bound_per_class':3*float(ndtr(-LIMIT)),'stats':stats,'cpu_seconds':time.process_time()-start}
