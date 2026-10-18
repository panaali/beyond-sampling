"""Original stage1-to-stage2 interpolated conditional transition."""
from collections import defaultdict
from pathlib import Path
import importlib.util
import sys,time
import numpy as np
from scipy.special import ndtr
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'rank_conditioned_transport'))
from engine import grids
from polynomial import StagePieces,fallback_rule
_path=HERE.parent/'rank_conditioned_transport/penultimate_cycle/cache.py'
_spec=importlib.util.spec_from_file_location('middle_old_penultimate',_path)
pen=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(pen)


class InterpolatedMiddle:
    def __init__(self,model,states,h=3,limit=6.,seconds=15.):
        start=time.process_time();self.stats=defaultdict(float);self.limit=limit
        self.deadline=start+seconds;self.model=model;self.states=np.asarray(states)
        if len(model[2])!=4:raise ValueError('frozen T4 scope')
        self.axes,self.grid_states,self.bounds=grids(model,h,limit,self.stats,2.)
        firstmax=np.array([a[-1] for a in self.axes[1]])
        secondmax=np.array([a[-1] for a in self.axes[2]])
        if np.any(self.states<0) or np.any(self.states>firstmax):raise ValueError('source outside stage1 grid')
        endpoints=np.maximum(self.states[:,None,:]@model[1].T+np.array([-limit,limit])[None,:,None]*model[0],0.)
        if np.any(endpoints>secondmax):raise ValueError('full image support outside stage2 grid')
        self.domain=dict(source_states_inside=True,next_support_inside=True,
                         stage1_maxima=firstmax.tolist(),stage2_maxima=secondmax.tolist(),
                         endpoint_maxima=endpoints.max(axis=(0,1)).tolist(),clamping=False,extrapolation=False)
        fallback_rule.cache_clear();fallback_rule();self.stats['fallback_gauss12_rules_constructed']=1
        self.pieces=StagePieces(model,self.states,self.axes[2],limit,self.stats,self.deadline)
        self.penultimate=pen.PenultimateCache(model,self.grid_states[2],limit,4.,seconds)
        self.stats['construction_cpu_seconds']=time.process_time()-start

    def values(self,u):
        start=time.process_time();q=float((ndtr(u)-ndtr(-self.limit))/(1-2*ndtr(-self.limit)))
        matrices,error=self.pieces.matrices(u,q,self.stats,self.deadline)
        grid=self.penultimate.values(u,q)
        low,high=(m@grid for m in matrices)
        out=np.stack((low[:,0],(low[:,1]+high[:,0])/2),axis=1)
        self.stats['sparse_value_products']+=2*sum(m.nnz for m in matrices)
        self.stats['query_cpu_seconds']+=time.process_time()-start
        return out,error
