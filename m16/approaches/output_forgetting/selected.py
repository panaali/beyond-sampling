"""Selected classes only; positive polynomial preintegration, not full accuracy."""
import math,time
import numpy as np
from scipy.special import ndtr
from fractions import Fraction
from controls import coefficient
from direct_gaussian import LIMIT,phi
from rank_mass import cdf_difference

def selected(T,k,signs,outer=64):
    if T<4 or k<0 or signs not in ('++','-+','+-','--'):raise ValueError('domain')
    start=time.process_time();m=T-4
    stats={'rule_generations':0,'rule_nodes':0,'outer_nodes':0,'polynomial_nodes':0,'cdf_values':0,'exp_values':0,'atan_calls':0,'fraction_polynomial_degree':0}
    if signs=='--':
        exact=coefficient(m,1);stats['fraction_polynomial_degree']=m+1;stats['atan_calls']=1
        answer=float(exact)*math.atan2(1.,k)/(2*math.pi);tail=0.
    else:
        x,w=np.polynomial.legendre.leggauss(outer);stats.update(rule_generations=1,rule_nodes=outer,outer_nodes=outer)
        u=(x+1)*LIMIT/2;Q=ndtr(-u);F=1-Q;G=F-.5;stats['cdf_values']+=outer
        if signs=='+-':
            n=(m+3)//2;y,v=np.polynomial.legendre.leggauss(n);nodes=(y+1)/2;weights=v/2
            stats['rule_generations']+=1;stats['rule_nodes']+=n;stats['polynomial_nodes']=2*outer*n
            lo=.5+G[:,None]*nodes;hi=F[:,None]+Q[:,None]*nodes
            f0=lo**m*(lo-.5)
            f1=(1-hi)*(m*hi**(m-1)*(hi-.5)+hi**m) if m else 1-hi
            H=G*(f0@weights)+Q*(f1@weights)
            values=ndtr(-k*u)*H;stats['cdf_values']+=outer
        else:
            common0=F**m*G**2
            common1=m*F**(m-1)*Q*G**2+2*F**m*G*Q if m else 2*G*Q
            cutoff=u/k if k else np.full_like(u,np.inf)
            if signs=='-+':values=common1*cdf_difference(cutoff,0);stats['cdf_values']+=2*outer
            else:
                below=cdf_difference(np.minimum(u,cutoff),0)
                above=cdf_difference(cutoff,u) if k<1 else np.zeros_like(u)
                values=common1*below+common0*above;stats['cdf_values']+=2*outer+(2*outer if k<1 else 0)
        if np.any(values<0):raise ArithmeticError('negative selected integrand')
        stats['exp_values']+=outer;answer=float((w*LIMIT/2)@(phi(u)*values));tail=float(ndtr(-LIMIT))
    return {'T':T,'k':k,'signs':signs,'selected_class':'D' if signs[1]=='+' else 'B','outer':outer,'correct_mass':answer,'tail_bound':tail,'stats':stats,'cpu_seconds':time.process_time()-start}
