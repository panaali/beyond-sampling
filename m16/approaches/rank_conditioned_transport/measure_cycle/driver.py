"""Same cached recurrence, two explicitly normalized target coordinates."""
from pathlib import Path
import importlib.util,sys,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr,ndtri,betainc,betaincc
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'penultimate_cycle'))
spec=importlib.util.spec_from_file_location('measure_private_penultimate',HERE.parent/'penultimate_cycle/driver.py')
pen=importlib.util.module_from_spec(spec);spec.loader.exec_module(pen)
from breakpoints import prepare
from polynomial import fallback_rule
from engine import fixed_target


def rule(points,T,order,coordinate,limit=6.,target_limit=4.):
    lo,hi=np.asarray(points[:-1]),np.asarray(points[1:])
    tiny=hi-lo<=8*np.finfo(float).eps*np.maximum(1.,abs((lo+hi)/2))
    omitted_width=float((hi[tiny]-lo[tiny]).sum());lo,hi=lo[~tiny],hi[~tiny]
    alpha=ndtr(-limit);z=1-2*alpha
    qlo=(ndtr(lo)-alpha)/z;qhi=(ndtr(hi)-alpha)/z
    collapsed=qhi<=qlo;omitted_width+=float((hi[collapsed]-lo[collapsed]).sum())
    lo,hi,qlo,qhi=(a[~collapsed] for a in (lo,hi,qlo,qhi))
    x,w=leggauss(order)
    if coordinate=='q':
        half=(qhi-qlo)/2;q=((qhi+qlo)[:,None]/2+half[:,None]*x).ravel()
        weights=(half[:,None]*w).ravel()*T*(T-1)*q**(T-2)*(1-q)
    elif coordinate=='u':
        half=(hi-lo)/2;u=((hi+lo)[:,None]/2+half[:,None]*x).ravel()
        q=(ndtr(u)-alpha)/z
        weights=(half[:,None]*w).ravel()*T*(T-1)*q**(T-2)*(1-q)*np.exp(-u*u/2)/(np.sqrt(2*np.pi)*z)
    else:raise ValueError('unknown coordinate')
    qm,qp=(ndtr(np.array([-target_limit,target_limit]))-alpha)/z
    target_omission=float(betainc(T-1,2,qm)+betaincc(T-1,2,qp))
    beta_mode=(T-2)/(T-1);density_upper=T*(T-1)*beta_mode**(T-2)*(1-beta_mode)/(np.sqrt(2*np.pi)*z)
    omission=density_upper*omitted_width
    return q,weights,dict(coordinate=coordinate,complete_panels=len(points)-1,retained_panels=len(lo),target_nodes=len(q),tiny_panels=int(tiny.sum()),collapsed_q_panels=int(collapsed.sum()),omitted_panel_mass_bound=omission,omitted_target_mass_bound=target_omission,omitted_input_mass_bound=float(1-z**T),raw_target_mass=float(weights.sum()),raw_mass_defect_upper=abs(float(weights.sum())-(1-target_omission))+omission)


def setup(model,h,seconds):
    start=time.process_time();data=prepare(model,h,seconds=seconds)
    cache=pen.PenultimateCache(model,data['states'][len(model[2])-2],data['limit'],data['target_limit'],seconds-(time.process_time()-start))
    data['penultimate']=cache;data['points']=np.unique(np.r_[data['points'],cache.knots])
    if len(data['points'])-1>2000:raise ValueError('complete panels exceed cap2000')
    return data


def estimate(model,h=9,order=2,coordinate='q',seconds=100.):
    start=time.process_time();deadline=start+seconds;fallback_rule.cache_clear();fallback_rule()
    data=setup(model,h,seconds-(time.process_time()-start));stats=data['stats'];T=len(model[2])
    q,weights,out=rule(data['points'],T,order,coordinate,data['limit'],data['target_limit'])
    stats['fallback_gauss12_rules_constructed']=1;stats['target_legendre_rules_constructed']=1
    stats['target_density_cdf_values']+=2*out['retained_panels']+3+(len(q) if coordinate=='u' else 0)
    stats['target_beta_density_polynomial_nodes']+=len(q)
    if coordinate=='u':stats['target_normal_density_exp_values']+=len(q)
    contributions={j:0. for j in data['labels']};total_bound=0.
    for qq,weight in zip(q,weights):
        if time.process_time()>deadline:raise TimeoutError('measure cycle CPU cap')
        operators={}
        for label in data['labels']:
            if label==T-2:value,bound=pen.cached_label(data,float(qq),operators,deadline)
            else:value,bound=fixed_target(model,label,float(qq),data['axes'],data['states'],8,data['limit'],stats,deadline,'polynomial_piece_cached',operators,data['piece_cache'],data['terminal_cache'])
            contributions[label]+=float(weight)*value/T;total_bound+=float(weight)*bound/T
        size=sum(m.data.nbytes+m.indices.nbytes+m.indptr.nbytes for matrices,_ in operators.values() for m in matrices)
        stats['peak_streaming_conditional_kernel_bytes']=max(stats['peak_streaming_conditional_kernel_bytes'],size)
        stats['streaming_target_nodes']+=1
    out.update(h=h,order=order,contributions=contributions,transition_quadrature_only_bound=total_bound,penultimate_work=dict(data['penultimate'].stats),work=dict(stats),cpu_seconds=time.process_time()-start,total_error_certified=False,portable_flops_certified=False)
    return sum(contributions.values()),out
