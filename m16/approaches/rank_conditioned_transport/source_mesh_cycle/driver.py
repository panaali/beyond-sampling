"""Existing fixed-interpolant recurrence; complete topology with q-panel rule."""
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import ndtr,betainc,betaincc
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from breakpoints import prepare
from panel import stream
from mesh import source_mesh


def estimate(model,h=9,order=2,mode='baseline',seconds=100.):
    start=time.process_time();deadline=start+seconds
    data=prepare(model,h,seconds=seconds,mesh_builder=None if mode=='baseline' else source_mesh(mode))
    if len(data['points'])-1>2000:raise ValueError('complete panels exceed frozen cap2000')
    lo,hi=data['points'][:-1],data['points'][1:]
    tiny=hi-lo<=8*np.finfo(float).eps*np.maximum(1.,abs((lo+hi)/2))
    omitted_width=float((hi[tiny]-lo[tiny]).sum());lo,hi=lo[~tiny],hi[~tiny]
    limit=data['limit'];target_limit=data['target_limit'];T=len(model[2])
    alpha=ndtr(-limit);z=1-2*alpha
    qlo=(ndtr(lo)-alpha)/z;qhi=(ndtr(hi)-alpha)/z
    collapsed=qhi<=qlo
    omitted_width+=float((hi[collapsed]-lo[collapsed]).sum())
    qlo,qhi=qlo[~collapsed],qhi[~collapsed]
    x,w=leggauss(order);half=(qhi-qlo)/2
    q=((qhi+qlo)[:,None]/2+half[:,None]*x).ravel()
    weights=(half[:,None]*w).ravel()*T*(T-1)*q**(T-2)*(1-q)
    qm,qp=(ndtr(np.array([-target_limit,target_limit]))-alpha)/z
    target_omission=float(betainc(T-1,2,qm)+betaincc(T-1,2,qp))
    beta_mode=(T-2)/(T-1)
    density_upper=T*(T-1)*beta_mode**(T-2)*(1-beta_mode)/(np.sqrt(2*np.pi)*z)
    omission=density_upper*omitted_width
    data['stats']['target_density_cdf_values']+=2*len(lo)+3
    data['stats']['target_legendre_rules_constructed']+=1
    data['stats']['target_beta_density_polynomial_nodes']+=len(q)
    answer,contributions,bound=stream(data,q,weights,deadline)
    return answer,dict(mode=mode,h=h,order=order,complete_panels=len(data['points'])-1,
        retained_panels=len(qlo),target_nodes=len(q),tiny_panels=int(tiny.sum()),
        collapsed_q_panels=int(collapsed.sum()),omitted_panel_mass_bound=omission,
        omitted_target_mass_bound=target_omission,omitted_input_mass_bound=float(1-z**T),
        raw_target_mass=float(weights.sum()),
        raw_mass_defect_upper=abs(float(weights.sum())-(1-target_omission))+omission,
        transition_quadrature_only_bound=bound,contributions=contributions,
        work=dict(data['stats']),cpu_seconds=time.process_time()-start,
        total_error_certified=False,portable_flops_certified=False)
