"""T=4 angular quadrature using T55's analytic weighted two-input tail.

Generic hidden width. No random inputs in estimate; numerical outer integration.
"""
from pathlib import Path
import sys,time
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'feynman_fundamentals'))
from tail import conditional_probability,prefix_summary


def prefix_breaks(model):
    """Known two-step ReLU and prefix-order transitions, derived from weights."""
    wi,wh,_=model
    points=list(np.arange(9)*np.pi/4)
    for sign in (-1.,1.):
        gates=(wi*sign>0)
        a=wh@(wi*gates)
        for ai,bi in zip(a,wi):
            if ai==0 and bi==0:continue
            theta=float(np.arctan2(-ai,bi)%(2*np.pi))
            for angle in (theta,(theta+np.pi)%(2*np.pi)):
                if sign*np.cos(angle)>1e-13:points.append(angle)
    points=sorted(points)
    unique=[points[0]]
    for x in points[1:]:
        if x-unique[-1]>1e-13:unique.append(x)
    return np.array(unique)


def angular_estimate(model, order=8, offset=0., max_cpu=110., boundary_mode='octants', subdivision=1):
    if model[2].shape[0]!=4:
        raise ValueError('This outer rule requires sequence length4')
    start=time.process_time();wall=time.perf_counter()
    z,w=leggauss(order)
    if boundary_mode=='prefix':
        if offset:raise ValueError('Keep known prefix boundaries fixed; use subdivision instead of offset')
        boundaries=prefix_breaks(model)
    elif boundary_mode=='octants':boundaries=offset+np.arange(9)*np.pi/4
    else:raise ValueError('Unknown boundary_mode')
    if subdivision<1:raise ValueError('subdivision must be positive')
    boundaries=np.array([a+(b-a)*j/subdivision for a,b in zip(boundaries[:-1],boundaries[1:]) for j in range(subdivision)]+[boundaries[-1]])
    nodes=(len(boundaries)-1)*order
    stats=dict(order=order,angular_nodes=nodes,deterministic_prefix_steps=2*nodes,boundary_mode=boundary_mode,subdivision=subdivision,boundaries=boundaries.tolist(),
               offset=offset,clips=0,affine_products=0,moment_calls=0,
               gate_cells=0,correct_pieces=0,high_precision_fallbacks=0,
               maximum_partition_error=0.)
    estimate=0.;values=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        theta=(left+right)/2+z*(right-left)/2
        for angle,weight in zip(theta,w):
            prefix=np.array([np.cos(angle),np.sin(angle)])
            # A prefix tie is a null event; avoid evaluating exactly on it.
            if prefix[0]==prefix[1]:
                angle=np.nextafter(angle,np.inf);prefix=np.array([np.cos(angle),np.sin(angle)])
            h,top=prefix_summary(model,prefix)
            p,s=conditional_probability(model,h,top,2)
            estimate+=weight*p*(right-left)/(4*np.pi)
            values.append([float(angle),float(p)])
            for key in ('clips','affine_products','moment_calls','gate_cells','correct_pieces','high_precision_fallbacks'):
                stats[key]+=s[key]
            stats['maximum_partition_error']=max(stats['maximum_partition_error'],abs(s['partition_mass']-1))
            if time.process_time()-start>max_cpu:
                raise TimeoutError(f'angular rule exceeded CPU cap after{len(values)}nodes')
    stats.update(cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall,
                 values=values)
    return float(estimate),stats


def forward(model,x):
    wi,wh,wo=model
    h=np.zeros((len(x),len(wi)))
    for t in range(x.shape[1]):h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
    return h@wo.T


def est(model,budget=None,rng=None):
    """Fixed GL8; budget advisory, rng unused. Cost is CPU forward equivalents.

    Deterministic timing inputs do not influence estimate. Calibration CPU charged.
    This hardware-dependent cost is explicitly not a portable scalar FLOP count.
    """
    p,s=angular_estimate(model,order=8)
    x=np.zeros((32768,4));start=time.process_time();forward(model,x)
    calibration=time.process_time()-start
    return p,(s['cpu_seconds']+calibration)/(calibration/len(x))
