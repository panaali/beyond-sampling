"""Identical rank-conditioned reachable probes for terminal mesh comparisons."""
from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri
from scipy.interpolate import RegularGridInterpolator

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from engine import grids, possible_labels, batched_terminal
from spatial import terminal_mesh
sys.path.pop(0)


def evaluate(model, prefix_order=8, q_order=8, hidden_orders=(9,17),
             modes=('baseline','curvature','gate'), seconds=100.):
    start=time.process_time();deadline=start+seconds
    wi,wh,wo=model
    if wi.shape!=(2,) or wo.shape!=(4,2):raise ValueError('frozen H2T4 scope')
    labels=possible_labels(wo)
    if 3 in labels:raise ValueError('target-last label not covered')
    stats=defaultdict(float);limit=6.;alpha=ndtr(-limit);z=1-2*alpha
    vn,vw=leggauss(prefix_order);vn=(vn+1)/2;vw=vw/2
    v1,v2=np.meshgrid(vn,vn,indexing='ij')
    pairweight=np.outer(vw,vw).ravel();v1=v1.ravel();v2=v2.ravel()
    qn,qw=leggauss(q_order)
    q=np.r_[(qn+1)/4,.5+(qn+1)/4]
    weights=np.tile(qw/4,2)*12*q*q*(1-q)
    stats['legendre_rules_constructed']=2
    variants=[]
    for hidden_order in hidden_orders:
        for mode in modes:
            setup_start=time.process_time();work=defaultdict(float)
            axes,states,_=grids(model,hidden_order,limit,work,2.)
            metadata=None
            if mode!='baseline':
                axes,states,metadata=terminal_mesh(mode)(model,axes,states,limit,work)
            variants.append(dict(hidden_order=hidden_order,mode=mode,
                axis=axes[3],states=states[3],cache={},work=work,
                setup_cpu=time.process_time()-setup_start,metadata=metadata,
                signed=0.,absolute=0.,squared=0.,maximum=0.,by_label={},
                first_cell_mass=np.zeros(2),probe_mass=0.))
    exact=0.;by_label={};max_bound_violation=0.;min_probability=1.;max_probability=0.
    for qq,weight in zip(q,weights):
        u=float(ndtri(alpha+z*qq));stats['target_inverse_cdf_values']+=1
        below=[ndtri(alpha+z*qq*v) for v in (v1,v2)]
        above=[ndtri(alpha+z*(qq+(1-qq)*v)) for v in (v1,v2)]
        stats['prefix_inverse_cdf_values']+=4*len(v1)
        for label in labels:
            hidden=[];need=[]
            for above_slot in range(3):
                h=np.zeros((len(v1),2));slot=0
                for step in range(3):
                    if step==label:x=np.full(len(v1),u)
                    else:
                        x=above[slot] if slot==above_slot else below[slot]
                        slot+=1
                    h=np.maximum(h@wh.T+x[:,None]*wi,0)
                    stats['recurrence_multiplications']+=len(v1)*(4+2)
                    stats['recurrence_additions']+=len(v1)*4
                    stats['recurrence_relu_comparisons']+=len(v1)*2
                hidden.append(h);need.extend([int(above_slot==2)]*len(v1))
            h=np.concatenate(hidden);need=np.asarray(need)
            mass=np.tile(pairweight,3)*float(weight)/12
            truth=batched_terminal(model,h,u,float(qq),label,limit,stats,{})
            truth=truth[np.arange(len(h)),need]
            min_probability=min(min_probability,float(truth.min()))
            max_probability=max(max_probability,float(truth.max()))
            contribution=float(mass@truth);exact+=contribution
            by_label[label]=by_label.get(label,0.)+contribution
            for item in variants:
                axis=item['axis'];work=item['work']
                violation=max(float(np.max(-h)),
                    float(np.max(h-np.array([a[-1] for a in axis]))))
                max_bound_violation=max(max_bound_violation,violation)
                if violation>1e-12:raise ArithmeticError('reachable state outside stage box')
                values=batched_terminal(model,item['states'],u,float(qq),label,
                                        limit,work,item['cache'])
                gridvalues=values.reshape(len(axis[0]),len(axis[1]),2)
                interp=RegularGridInterpolator(axis,gridvalues,bounds_error=True)
                approximate=interp(h)[np.arange(len(h)),need]
                if approximate.min() < -1e-12 or approximate.max()>1+1e-10:
                    raise ArithmeticError('interpolated probability out of bounds')
                delta=approximate-truth
                signed=float(mass@delta);absolute=float(mass@abs(delta))
                item['signed']+=signed;item['absolute']+=absolute
                item['squared']+=float(mass@(delta*delta))
                item['maximum']=max(item['maximum'],float(abs(delta).max()))
                entry=item['by_label'].setdefault(label,dict(signed=0.,absolute=0.))
                entry['signed']+=signed;entry['absolute']+=absolute
                item['first_cell_mass']+=mass@(h<=np.array([a[1] for a in axis]))
                item['probe_mass']+=float(mass.sum())
                work['bilinear_queries']+=len(h)
                work['bilinear_interpolator_constructions']+=1
            if time.process_time()>deadline:raise TimeoutError('frozen diagnostic deadline')
    rows=[]
    for item in variants:
        rows.append({k:v for k,v in item.items()
                     if k not in ('axis','states','cache','work','first_cell_mass')})
        rows[-1].update(axes=[a.tolist() for a in item['axis']],
                       work=dict(item['work']),
                       first_cell_mass=item['first_cell_mass'].tolist())
    return dict(prefix_order=prefix_order,q_order_per_half=q_order,
                raw_target_rule_mass=float(weights.sum()),
                omitted_input_mass_bound=float(1-z**4),
                source_quadrature_estimate=exact,by_label=by_label,rows=rows,
                min_probability=min_probability,max_probability=max_probability,
                maximum_stage_box_violation=max_bound_violation,
                shared_probe_work=dict(stats),cpu_seconds=time.process_time()-start,
                source_sha256={f:hashlib.sha256((HERE.parent/f).read_bytes()).hexdigest()
                               for f in ('engine.py','spatial.py')},
                total_error_certified=False,portable_flops_certified=False)
