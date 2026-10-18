"""Positive two-Gaussian polygons for x, fixed target u, y chronology."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import importlib.util
import math
import time
import numpy as np

_path=Path(__file__).resolve().parent.parent/'gaussian_pair_terminal/terminal.py'
_spec=importlib.util.spec_from_file_location('verified_pair_kernel',_path)
kernel=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(kernel)


class MiddleTarget(kernel.PairTerminal):
    def __init__(self,model,hidden,u,label=None,limit=6.,diagnostic=False):
        start=time.process_time();self.stats=defaultdict(float)
        wi,wh,wo=(np.array(a,dtype=float,copy=True) for a in model)
        h=np.asarray(hidden,dtype=float);T=len(wo)
        label=T-2 if label is None else label
        if wi.shape!=(2,) or wh.shape!=(2,2) or wo.shape!=(T,2) or h.shape!=(2,) or np.any(h<0) or not all(np.all(np.isfinite(a)) for a in (wi,wh,wo,h)):
            raise ValueError('finite H2 model and nonnegative hidden state required')
        if not 0<=label<T or int(label)!=label or (label!=T-2 and not diagnostic):
            raise ValueError('penultimate target required; other labels diagnostic only')
        if not math.isfinite(limit) or limit<=0 or not math.isfinite(u) or not -limit<u<limit:
            raise ValueError('finite target inside positive truncated support required')
        self.limit=limit;self.model=wi,wh,wo;self.hidden=h;self.label=label;self.target=u;self.cells=[]
        offset=wh@h;edges=[-limit,limit];self.stats['first_map_scalar_products']+=4
        if not np.all(np.isfinite(offset)):raise ArithmeticError('nonfinite first affine map')
        for a,b in zip(wi,offset):
            if a:
                root=-b/a;self.stats['first_gate_roots']+=1
                if -limit<root<limit:edges.append(root)
        edges=np.unique(edges)
        for first_left,first_right in zip(edges[:-1],edges[1:]):
            active=offset+wi*((first_left+first_right)/2)>0
            middle_a=wh@(wi*active);middle_b=wh@(offset*active)+wi*u
            self.stats['middle_map_scalar_products']+=8
            if not np.all(np.isfinite(np.r_[middle_a,middle_b])):raise ArithmeticError('nonfinite middle affine map')
            middle_edges=[first_left,first_right]
            for a,b in zip(middle_a,middle_b):
                if a:
                    root=-b/a;self.stats['middle_gate_roots']+=1
                    if first_left<root<first_right:middle_edges.append(root)
            middle_edges=np.unique(middle_edges)
            for left,right in zip(middle_edges[:-1],middle_edges[1:]):
                middle_mask=middle_a*((left+right)/2)+middle_b>0
                cx=wh@(middle_a*middle_mask);cc=wh@(middle_b*middle_mask)
                self.stats['final_map_scalar_products']+=8
                if not np.all(np.isfinite(np.r_[cx,cc])):raise ArithmeticError('nonfinite final affine map')
                zero=(cx==0)&(wi==0)&(cc==0)
                for pattern in product((False,True),repeat=2):
                    mask=np.asarray(pattern);self.stats['final_masks_tested']+=1
                    if np.any(mask&zero):continue
                    logits=np.stack((wo@(cx*mask),wo@(wi*mask),wo@(cc*mask)),axis=1)
                    self.stats['readout_scalar_products']+=6*T
                    if not np.all(np.isfinite(logits)):raise ArithmeticError('nonfinite affine readout')
                    constraints=[(np.array([1.,0.,-left]),False),(np.array([-1.,0.,right]),False)]
                    constraints.extend((np.array([cx[i],wi[i],cc[i]])*(1 if mask[i] else -1),bool(mask[i])) for i in range(2) if not zero[i])
                    for other in range(T):
                        if other!=label:
                            constraints.append((logits[label]-logits[other],other<label));self.stats['readout_margins']+=1
                    self.cells.append(constraints)
        self.pieces={False:[p for c in self.cells for p in kernel.vertical(c,limit,self.stats)],True:None}
        self.stats['vertical_pieces']=len(self.pieces[False]);self.stats['construction_cpu_seconds']=time.process_time()-start

    def probabilities(self,u=None,swap=False,tolerance=1e-11):
        if u is not None and u!=self.target:raise ValueError('target differs from frozen geometry')
        return super().probabilities(self.target,swap,tolerance)
