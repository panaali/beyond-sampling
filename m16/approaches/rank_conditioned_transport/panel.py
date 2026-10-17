"""Streaming outer integration with charged complete breakpoint panels."""
from collections import defaultdict
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr,betainc,betaincc
from engine import fixed_target
from breakpoints import prepare


def density_nodes(T,limit,target_limit,order,edges):
    nodes,base=leggauss(order)
    lo,hi=edges
    mid=(lo+hi)/2;half=(hi-lo)/2
    u=(mid[:,None]+half[:,None]*nodes).ravel()
    alpha=ndtr(-limit);z=1-2*alpha
    q=(ndtr(u)-alpha)/z
    weights=(half[:,None]*base).ravel()*T*(T-1)*q**(T-2)*(1-q)*np.exp(-u*u/2)/(np.sqrt(2*np.pi)*z)
    qm,qp=(ndtr(np.array([-target_limit,target_limit]))-alpha)/z
    omitted=float(betainc(T-1,2,qm)+betaincc(T-1,2,qp))
    return q,weights,omitted


def stream(data,q,weights,deadline):
    stats=data['stats'];model=data['model'];T=len(model[2])
    contributions={j:0. for j in data['labels']}
    transition_bound=0.
    for qq,weight in zip(q,weights):
        operator_cache={}
        for label in data['labels']:
            value,bound=fixed_target(model,label,float(qq),data['axes'],data['states'],8,
                data['limit'],stats,deadline,'polynomial_piece_cached',operator_cache,
                data['piece_cache'],data['terminal_cache'])
            contributions[label]+=float(weight)*value/T
            transition_bound+=float(weight)*bound/T
        size=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes
                 for matrices,_ in operator_cache.values() for m in matrices)
        stats['peak_streaming_conditional_kernel_bytes']=max(stats['peak_streaming_conditional_kernel_bytes'],size)
        stats['streaming_target_nodes']+=1
    return sum(contributions.values()),contributions,transition_bound


def estimate_panel(model,hidden_order=9,order=2,scheme='complete',seconds=100.):
    start=time.process_time();deadline=start+seconds
    data=prepare(model,hidden_order,seconds=seconds)
    if len(data['points'])-1>2000:
        raise ValueError('complete panel count exceeds frozen pilot cap')
    lo,hi=data['points'][:-1],data['points'][1:]
    unresolved=hi-lo<=8*np.finfo(float).eps*np.maximum(1.,abs((lo+hi)/2))
    omitted_width=float((hi[unresolved]-lo[unresolved]).sum())
    lo,hi=lo[~unresolved],hi[~unresolved]
    if scheme=='uniform':
        edges=np.linspace(-data['target_limit'],data['target_limit'],len(lo)+1)
        lo,hi=edges[:-1],edges[1:]
        omitted_width=0.
    elif scheme!='complete':raise ValueError(scheme)
    T=len(model[2]);mode=(T-2)/(T-1)
    density_upper=T*(T-1)*mode**(T-2)*(1-mode)/(np.sqrt(2*np.pi)*(1-2*ndtr(-data['limit'])))
    omitted_panel_bound=density_upper*omitted_width
    q,weights,omitted_target=density_nodes(T,data['limit'],data['target_limit'],order,(lo,hi))
    data['stats']['target_density_cdf_values']+=len(q)
    data['stats']['target_density_exponentials']+=len(q)
    answer,contributions,bound=stream(data,q,weights,deadline)
    box_mass=float((1-2*ndtr(-data['limit']))**T)
    return answer,dict(hidden_order=hidden_order,order=order,scheme=scheme,
        complete_panels=len(data['points'])-1,used_panels=len(lo),target_nodes=len(q),
        omitted_panel_mass_upper_bound=omitted_panel_bound,omitted_target_mass_bound=omitted_target,
        omitted_input_mass_bound=1-box_mass,raw_target_rule_mass=float(weights.sum()),
        retained_mass_defect_upper=abs(float(weights.sum())-(1-omitted_target))+omitted_panel_bound,
        transition_quadrature_only_bound=bound,contributions=contributions,
        cpu_seconds=time.process_time()-start,work=dict(data['stats']),
        total_error_certified=False,classification='complete-breakpoint panel quadrature of finite-grid interpolant')
