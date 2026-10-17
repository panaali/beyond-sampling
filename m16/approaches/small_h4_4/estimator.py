"""T=4 angular quadrature using T55's analytic weighted two-input tail.

Generic hidden width. No random inputs in estimate; numerical outer integration.
"""
from pathlib import Path
import sys,time
import numpy as np
from numpy.polynomial.legendre import leggauss
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'feynman_fundamentals'))
from tail import conditional_probability,prefix_summary


def angular_estimate(model, order=8, offset=0., max_cpu=110.):
    if model[2].shape[0]!=4:
        raise ValueError('This outer rule requires sequence length4')
    start=time.process_time();wall=time.perf_counter()
    z,w=leggauss(order)
    stats=dict(order=order,angular_nodes=8*order,deterministic_prefix_steps=16*order,
               offset=offset,clips=0,affine_products=0,moment_calls=0,
               gate_cells=0,correct_pieces=0,high_precision_fallbacks=0,
               maximum_partition_error=0.)
    estimate=0.;values=[]
    for segment in range(8):
        theta=offset+(segment+.5)*np.pi/4+z*np.pi/8
        for angle,weight in zip(theta,w):
            prefix=np.array([np.cos(angle),np.sin(angle)])
            # A prefix tie is a null event; avoid evaluating exactly on it.
            if prefix[0]==prefix[1]:
                angle=np.nextafter(angle,np.inf);prefix=np.array([np.cos(angle),np.sin(angle)])
            h,top=prefix_summary(model,prefix)
            p,s=conditional_probability(model,h,top,2)
            estimate+=weight*p/16
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
