"""Original finite-grid last-two-step continuation on fixed absolute states."""
from collections import defaultdict
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import ndtr
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport'))
from engine import grids,batched_terminal
from polynomial import StagePieces,fallback_rule


class InterpolatedPair:
    def __init__(self,model,states,h=9,limit=6.,seconds=15.):
        start=time.process_time();self.stats=defaultdict(float);self.model=model;self.limit=limit;self.deadline=start+seconds
        self.states=np.asarray(states);T=len(model[2])
        if T!=4:raise ValueError('frozen comparison is T4')
        self.axes,self.grid_states,self.bounds=grids(model,h,limit,self.stats,2.)
        maxima=np.array([a[-1] for a in self.axes[2]])
        if np.any(self.states<0) or np.any(self.states>maxima):raise ValueError('source state outside stage2 grid domain')
        endpoint_states=np.maximum(self.states[:,None,:]@model[1].T+np.array([-limit,limit])[None,:,None]*model[0],0.)
        nextmax=np.array([a[-1] for a in self.axes[3]])
        if np.any(endpoint_states<0) or np.any(endpoint_states>nextmax):raise ValueError('next scalar support extends outside stage3 grid domain')
        self.domain=dict(source_states_inside=True,next_support_inside=True,stage2_maxima=maxima.tolist(),stage3_maxima=nextmax.tolist(),next_endpoint_maxima=endpoint_states.max(axis=(0,1)).tolist(),clamping=False,extrapolation=False)
        fallback_rule.cache_clear();fallback_rule();self.stats['fallback_gauss12_rules_constructed']=1
        self.pieces=StagePieces(model,self.states,self.axes[3],limit,self.stats,self.deadline)
        self.terminal_cache={};self.operators={};self.stats['construction_cpu_seconds']=time.process_time()-start

    def values(self,u,label):
        start=time.process_time();z=1-2*ndtr(-self.limit);q=float((ndtr(u)-ndtr(-self.limit))/z)
        if u not in self.operators:self.operators[u]=self.pieces.matrices(u,q,self.stats,self.deadline)
        matrices,error=self.operators[u]
        grid=batched_terminal(self.model,self.grid_states[3],u,q,label,self.limit,self.stats,self.terminal_cache)
        low,high=(m@grid for m in matrices)
        values=np.stack((low[:,0],(low[:,1]+high[:,0])/2),axis=1)
        self.stats['sparse_value_products']+=2*sum(m.nnz for m in matrices)
        self.stats['query_cpu_seconds']+=time.process_time()-start
        return values,error
