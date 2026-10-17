"""Exact rank-conditioned first-input marginal, used only for mesh placement."""
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import betainc, ndtr
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from engine import possible_labels


def mixture_cdf(q,T,labels):
    q=np.asarray(q);g=betainc(T-1,2,q)
    if 0 not in labels:return (T*q-g)/(T-1)
    K=len(labels)
    return ((T-K)*g+T*(K-1)*q)/(K*(T-1))


def positive_cdf(s,sign,T,labels,limit,stats=None):
    alpha=ndtr(-limit);z=1-2*alpha
    q=(ndtr(sign*np.asarray(s))-alpha)/z
    half=float(mixture_cdf(.5,T,labels))
    value=mixture_cdf(q,T,labels)
    if stats is not None:
        stats['source_mesh_normal_cdf_values']+=np.size(s)+1
        stats['source_mesh_beta_cdf_values']+=np.size(s)+1
    return (value-half)/(1-half) if sign>0 else (half-value)/half


def source_mesh(mode):
    if mode not in ('physical','reversed'):raise ValueError(mode)
    def build(model,axes,states,limit,stats):
        start=time.process_time();wi,wh,wo=model
        if np.shape(wi)!=(2,) or wi[0]*wi[1]>=0:
            raise ValueError('prototype requires two nonzero mixed-sign input weights')
        labels=possible_labels(wo);T=len(wo);newaxes=[]
        for d in range(2):
            def cdf(s):
                physical=positive_cdf(s,np.sign(wi[d]),T,labels,limit,stats)
                if mode=='reversed':
                    physical=1-positive_cdf(limit-s,np.sign(wi[d]),T,labels,limit,stats)
                return .5*np.sqrt(s/limit)+.5*physical
            # Same tiny outer bound padding as canonical grids, preserving coverage.
            quantiles=np.linspace(0.,1.,len(axes[1][d]));nodes=[0.]
            for p in quantiles[1:-1]:
                nodes.append(brentq(lambda s:cdf(s)-p,0.,limit,xtol=1e-14,rtol=1e-14))
                stats['source_mesh_scalar_root_solves']+=1
            nodes.append(limit)
            axis=abs(wi[d])*(1+1e-12)*np.array(nodes)
            if np.any(np.diff(axis)<=0):raise ArithmeticError('source axis not ordered')
            newaxes.append(axis)
        axes=list(axes);states=list(states);axes[1]=tuple(newaxes)
        states[1]=np.stack(np.meshgrid(*newaxes,indexing='ij'),axis=-1).reshape(-1,2)
        stats['hidden_grid_points_constructed']+=len(states[1])
        stats['source_mesh_setup_cpu_seconds']+=time.process_time()-start
        return axes,states,dict(mode=mode,labels=labels,changed_stage=1,
            axes=[a.tolist() for a in newaxes],conditional_first_sign_mass=float(mixture_cdf(.5,T,labels)))
    return build
