"""T3 full accuracy from exactly two analytic two-input tails.
No random model inputs; imports T55's shared numerical geometry.
"""
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'feynman_fundamentals'))
from tail import conditional_probability,prefix_summary

def calculate(model):
    if len(model[2])!=3:
        raise ValueError('This estimator implements T3 only')
    start=time.process_time();total=0.;counts={};details=[]
    for sign in (-1.,1.):
        hidden,top=prefix_summary(model,np.array([sign]))
        value,stats=conditional_probability(model,hidden,top,1)
        total+=value/2
        details.append(dict(prefix_sign=sign,probability=value,**stats))
        for key in ('clips','affine_products','moment_calls','gate_cells','correct_pieces'):
            counts[key]=counts.get(key,0)+stats[key]
    H=len(model[0]);V=2*H+7
    operation_bound=(1000*(V+3)*(counts['clips']+counts['moment_calls'])
                     +100*(H+1)**2*counts['affine_products']
                     +1000*(H+1)*counts['gate_cells']+1000*(H+1)**2)
    counts.update(prefix_steps=2,cpu_seconds=time.process_time()-start,details=details,
                  scalar_operation_upper_bound=operation_bound,
                  forward_scalar_operation_proxy=2*(3*H*(H+1)+3*H),
                  bound_convention='loose arithmetic/special-function unit-cost bound; excludes Python/memory overhead')
    return total,counts

def forward(model,x):
    wi,wh,wo=model;h=np.zeros((len(x),len(wi)))
    for t in range(x.shape[1]):h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
    return h@wo.T

def est(model,budget,rng):
    """Indivisible analytic computation; returns proposed conservative arithmetic FPE.
    Budget is advisory; callers must compare actual returned cost, not requested budget.
    """
    p,stats=calculate(model)
    return p,stats['scalar_operation_upper_bound']/stats['forward_scalar_operation_proxy']
