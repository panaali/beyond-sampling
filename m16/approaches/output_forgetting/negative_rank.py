"""Signed target sectors for A-,B+,C-,D- with both pair constraints."""
import math,time
import numpy as np
from scipy.special import ndtr,erf
from rank_mass import mul,power,cdf_difference
from direct_gaussian import phi,LIMIT

def integrate(T,kappa,k,outer=64,inner=32):
    if T<4 or min(kappa,k)<0:raise ValueError('domain')
    start=time.process_time();x,w=np.polynomial.legendre.leggauss(outer);y,v=np.polynomial.legendre.leggauss(inner);nodes=(y+1)/2;weights=v/2
    stats={'rule_generations':2,'rule_nodes':outer+inner,'outer_nodes':outer,'inner_nodes':0,'cdf_values':0,'exp_values':0,'erf_values':0}
    def Q(t):stats['cdf_values']+=np.asarray(t).size;return ndtr(-np.asarray(t))
    def G(t):stats['erf_values']+=np.asarray(t).size;return .5*erf(np.asarray(t)/math.sqrt(2))
    def delta(hi,lo):stats['cdf_values']+=2*max(np.asarray(hi).size,np.asarray(lo).size);return cdf_difference(hi,lo)
    def integral(lo,hi,fun):
        if hi<=lo:return 0.
        z=lo+(hi-lo)*nodes;stats['inner_nodes']+=inner;stats['exp_values']+=inner
        return (hi-lo)*float(weights@(phi(z)*fun(z)))
    m=T-4;pair2=math.atan2(1,k)/(2*math.pi);pos=np.zeros(5);neg=np.zeros(5)
    for s,wt in zip((x+1)*LIMIT/2,w*LIMIT/2):
        tail=float(Q(s));F=1-tail
        if kappa:
            first=(integral(0,s,lambda b:G(b/kappa)),integral(s,LIMIT,lambda b:G(b/kappa)))
            I=integral(s,LIMIT,lambda a:Q(kappa*a));partnerA=float(G(s/kappa))
        else:first=(.5*float(G(s)),.5*tail);I=.5*tail;partnerA=.5
        if k:
            top=min(LIMIT,s/k) if k<1 else s
            J=tail*float(delta(top,s))+integral(top,LIMIT,lambda c:Q(k*c))
        else:J=tail*tail
        Pplus=mul(power(F,tail,m-1),first)[1]*pair2 if m else 0.
        Bplus=(m*F**(m-1)*tail if m else 0.)*partnerA*pair2
        Pminus=tail**(m-1)*I*J if m else 0.
        Aminus=tail**m*float(Q(kappa*s))*J
        Cminus=tail**m*I*float(Q(max(s,k*s)))
        Dminus=tail**m*I*float(delta(s/k if k else math.inf,s)) if k<1 else 0.
        factor=wt*float(phi(s));stats['exp_values']+=1
        pos+=factor*np.array([Pplus,0.,Bplus,0.,0.]);neg+=factor*np.array([Pminus,Aminus,0.,Cminus,Dminus])
    total=pos+neg;vector=[total[0]]*m+total[1:].tolist();event=math.atan2(1,kappa)*math.atan2(1,k)/(4*math.pi**2)
    if min(total)<0:raise ArithmeticError('negative target contribution')
    return {'T':T,'kappa':kappa,'k':k,'outer':outer,'inner':inner,'classes':total.tolist(),'positive_classes':pos.tolist(),'negative_classes':neg.tolist(),'joint_by_target':[float(z) for z in vector],'event_mass':event,'partition_error':float(sum(vector)-event),'tail_bound_per_class':3*float(ndtr(-LIMIT)),'stats':stats,'cpu_seconds':time.process_time()-start}
