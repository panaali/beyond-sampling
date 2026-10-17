"""Positive rank integration on four disjoint signed output events."""
import math,time
import numpy as np
from scipy.special import ndtr
from rank_mass import mul,power,cdf_difference
from direct_gaussian import phi,LIMIT

def integrate(T,k,signs,outer=64,inner=32):
    if T<4 or k<0 or signs not in ('++','+-','-+','--'):raise ValueError('domain')
    start=time.process_time();x,w=np.polynomial.legendre.leggauss(outer);y,v=np.polynomial.legendre.leggauss(inner)
    nodes=(y+1)/2;weights=v/2
    stats={'rule_generations':2,'rule_nodes':outer+inner,'outer_nodes':outer,'inner_nodes':0,'cdf_values':0,'exp_values':0}
    pair_event=math.atan2(1.,k)/(2*math.pi);total=np.zeros(4);minimum=math.inf
    def Q(z):stats['cdf_values']+=np.asarray(z).size;return ndtr(-np.asarray(z))
    def delta(hi,lo):stats['cdf_values']+=2*max(np.asarray(hi).size,np.asarray(lo).size);return cdf_difference(hi,lo)
    def integral(lo,hi,f):
        if hi<=lo:return 0.
        pts=lo+(hi-lo)*nodes;stats['inner_nodes']+=inner;stats['exp_values']+=inner
        return (hi-lo)*float(weights@(phi(pts)*f(pts)))
    for u,wt in zip((x+1)*LIMIT/2,w*LIMIT/2):
        F=1-float(Q(u));G=F-.5;tail=1-F;cut=u/k if k else math.inf
        if signs=='++':
            p0=integral(0,min(u,cut),lambda c:delta(u,k*c))
            p1=integral(0,min(u,cut),lambda c:np.full_like(c,tail))
            p1+=integral(min(u,cut),u,lambda c:Q(k*c))
            p1+=integral(u,min(cut,LIMIT),lambda c:delta(u,k*c))
        elif signs=='+-':
            p0=integral(0,u,lambda c:Q(k*c));p1=integral(u,LIMIT,lambda c:Q(k*c))
        elif signs=='-+':
            p0=integral(0,min(cut,LIMIT),lambda c:delta(u,k*c))
            p1=integral(0,min(cut,LIMIT),lambda c:np.full_like(c,tail))
            p1+=integral(min(cut,LIMIT),LIMIT,lambda c:Q(k*c))
        else:p0,p1=pair_event,0.
        if min(p0,p1)<0:raise ArithmeticError(('negative pair',T,k,signs,u,p0,p1))
        minimum=min(minimum,p0,p1);m=T-4;ordinary=power(F,tail,m);positive=power(G,tail,2)
        A=mul(mul(power(F,tail,m-1),positive),(p0,p1))[1] if m else 0.
        B=mul(mul(ordinary,(G,tail)),(p0,p1))[1]
        common=mul(ordinary,positive)
        if signs[0]=='-':C=0.
        elif signs[1]=='-':C=mul(common,(float(Q(k*u)),0.))[1]
        else:C=mul(common,(float(delta(u,k*u)) if k<=1 else 0.,float(Q(max(u,k*u)))))[1]
        if signs[1]=='-':D=0.
        elif signs[0]=='-':D=mul(common,(float(delta(cut,0)),0.))[1]
        else:D=mul(common,(float(delta(min(u,cut),0)),float(delta(cut,u)) if cut>u else 0.))[1]
        stats['exp_values']+=1;total+=wt*float(phi(u))*np.array([A,B,C,D])
    vector=[total[0]]*(T-4)+[total[1],total[1],total[2],total[3]]
    event=pair_event/4
    return {'T':T,'k':k,'signs':signs,'outer':outer,'inner':inner,'classes':total.tolist(),'joint_by_target':[float(z) for z in vector],'event_mass':event,'partition_error':float(sum(vector)-event),'minimum_pair_mass':minimum,'tail_bound_per_class':2*float(ndtr(-LIMIT)),'stats':stats,'cpu_seconds':time.process_time()-start}
