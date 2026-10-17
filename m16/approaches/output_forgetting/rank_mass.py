"""Positive joint rank integrals for the four-positive output event."""
import math,time
import numpy as np
from scipy.special import ndtr,ndtri,owens_t

def mul(p,q):return (p[0]*q[0],p[1]*q[0]+p[0]*q[1])
def power(a,b,n):
    if n<0:raise ValueError('negative factor count')
    return (a**n,n*a**(n-1)*b if n else 0.)

def cdf_difference(hi,lo):
    if np.any(np.asarray(hi)<np.asarray(lo)):raise ArithmeticError('reversed CDF interval')
    result=ndtr(-np.asarray(lo))-ndtr(-np.asarray(hi))
    if np.any(result<0):raise ArithmeticError('negative CDF mass')
    return result

def pair_positive(u,F,d,nodes,weights,stats):
    G=F-.5;Q=1-F
    if d==0:return G*G,2*G*Q
    v=ndtri(.5+G*nodes)
    g0=G*float(weights@cdf_difference(u,d*v))
    delta=float(cdf_difference(u/d,u))
    if delta:
        v=ndtri(F+delta*nodes)
        g1=G*Q+delta*float(weights@cdf_difference(u,d*v))
        stats['inverse_cdf_values']+=2*len(nodes);stats['cdf_values']+=4*len(nodes)+2
    else:
        g1=G*Q;stats['inverse_cdf_values']+=len(nodes);stats['cdf_values']+=2*len(nodes)+2
    stats['inner_nodes']+=len(nodes)*(1+int(delta>0))
    return g0,g1

def pair_owen(u,F,d):
    G=F-.5;Q=1-F
    if d==0:return G*G,2*G*Q
    H=lambda x:.5*(float(ndtr(x))-.5)+math.atan(d)/(2*math.pi)-float(owens_t(x,d))
    H1,H2=H(u),H(u/d);F2=float(ndtr(u/d))
    return F*G-H1,G*Q+F*(F2-F)-H2+H1

def coefficients(T,d,u,F,pair):
    m=T-4;Q=1-F;G=F-.5
    ordinary=power(F,Q,m);positive=power(G,Q,2)
    A=mul(mul(power(F,Q,m-1),positive),pair)[1] if m else 0.
    B=mul(mul(ordinary,(G,Q)),pair)[1]
    common=mul(ordinary,positive)
    C=mul(common,(float(cdf_difference(u,d*u)),Q))[1]
    D=mul(common,(G,float(cdf_difference(u/d,u)) if d else Q))[1]
    return np.array([A,B,C,D])

def joint_mass(T,d,outer=32,inner=32,pair_method='positive'):
    if T<4 or not 0<=d<=1:raise ValueError('T>=4 and0<=d<=1 required')
    start=time.process_time();x,w=np.polynomial.legendre.leggauss(outer);ix,iw=np.polynomial.legendre.leggauss(inner)
    Fnodes=.75+.25*x;outerweights=w/4;nodes=(ix+1)/2;weights=iw/2
    stats={'rule_generations':2,'rule_generation_nodes':outer+inner,'outer_nodes':outer,'inner_nodes':0,'inverse_cdf_values':outer,'cdf_values':0,'owen_calls':0}
    total=np.zeros(4);min_pair=math.inf;negative_pairs=0
    for F,weight in zip(Fnodes,outerweights):
        u=float(ndtri(F))
        if pair_method=='positive':pair=pair_positive(u,F,d,nodes,weights,stats)
        elif pair_method=='owen':pair=pair_owen(u,F,d);stats['owen_calls']+=2 if d else 0;stats['cdf_values']+=3 if d else 0
        else:raise ValueError('unknown pair method')
        min_pair=min(min_pair,*pair);negative_pairs+=int(min(pair)<0)
        if min(pair)<0:raise ArithmeticError(('negative pair mass',u,pair))
        total+=weight*coefficients(T,d,u,F,pair);stats['cdf_values']+=4 if d else 2
    vector=np.array([total[0]]*(T-4)+[total[1],total[1],total[2],total[3]])
    event=.25*(.25-math.atan(d)/(2*math.pi))
    return {'T':T,'d':d,'outer':outer,'inner':inner,'pair_method':pair_method,'classes':total.tolist(),'joint_by_target':vector.tolist(),'event_mass':event,'partition_error':float(vector.sum()-event),'minimum_pair_mass':min_pair,'negative_pairs':negative_pairs,'stats':stats,'cpu_seconds':time.process_time()-start}
