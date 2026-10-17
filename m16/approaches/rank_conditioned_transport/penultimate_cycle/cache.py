"""Original-halfspace cache: fixed target u first, fresh scalar x last."""
from collections import defaultdict
from itertools import product
from pathlib import Path
import importlib.util,time
import numpy as np
path=Path(__file__).resolve().parent.parent/'target_last_cycle/terminal_cache.py'
spec=importlib.util.spec_from_file_location('penultimate_vertical_base',path)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class PenultimateCache(base.TerminalCache):
    def __init__(self,model,states,limit=6.,target_limit=4.,seconds=100.):
        start=time.process_time();self.stats=defaultdict(float)
        wi,wh,wo=model;H=len(wi);label=len(wo)-2
        if H!=2:raise ValueError('bounded cache requires H2')
        self.limit=limit;self.target_limit=target_limit;self.count=len(states);self.model=model
        rows=[];sides=[];pieces=[]
        for row,hidden in enumerate(states):
            offset=wh@hidden;edges=[-target_limit,target_limit]
            self.stats['first_map_scalar_products']+=H*H
            for a,b in zip(wi,offset):
                if a!=0:
                    root=-b/a;self.stats['first_gate_roots']+=1
                    if -target_limit<root<target_limit:edges.append(root)
            edges=np.unique(edges)
            for index,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
                active=offset+wi*((left+right)/2)>0
                cu=wh@(wi*active);cc=wh@(offset*active)
                self.stats['second_map_scalar_products']+=2*H*H
                zero=(cu==0)&(wi==0)&(cc==0)
                for pattern in product((False,True),repeat=H):
                    mask=np.asarray(pattern);self.stats['second_masks_tested']+=1
                    if np.any(mask&zero):continue
                    logits=np.stack((wo@(cu*mask),wo@(wi*mask),wo@(cc*mask)),axis=1)
                    self.stats['readout_scalar_products']+=3*len(wo)*H
                    constraints=[(np.array([1.,0.,-left]),False),
                                 (np.array([-1.,0.,right]),index<len(edges)-2)]
                    constraints.extend((np.array([cu[i],wi[i],cc[i]])*(1 if mask[i] else -1),bool(mask[i])) for i in range(H) if not zero[i])
                    reject=False
                    for other in range(len(wo)):
                        if other==label:continue
                        margin=logits[label]-logits[other];strict=other<label
                        self.stats['readout_margins']+=1
                        if not np.any(margin):
                            if strict:reject=True;break
                            continue
                        constraints.append((margin,strict))
                    if reject:continue
                    for side in (0,1):
                        new=base.vertical(constraints,(-limit,limit),limit,target_limit,side,self.stats)
                        rows.extend([row]*len(new));sides.extend([side]*len(new));pieces.extend(new)
            if time.process_time()-start>seconds:raise TimeoutError('penultimate cache construction cap')
        self.rows=np.asarray(rows,dtype=int);self.sides=np.asarray(sides,dtype=int)
        self.left=np.array([p[0] for p in pieces]);self.right=np.array([p[1] for p in pieces])
        self.lower=np.array([p[2] for p in pieces]).reshape(-1,2);self.upper=np.array([p[3] for p in pieces]).reshape(-1,2)
        self.left_closed=np.array([p[4] for p in pieces],dtype=bool);self.right_closed=np.array([p[5] for p in pieces],dtype=bool)
        self.knots=np.unique(np.r_[-target_limit,0.,target_limit,self.left,self.right])
        self.stats['terminal_pieces']=len(pieces);self.stats['terminal_knots']=len(self.knots)
        self.stats['construction_cpu_seconds']=time.process_time()-start
        self.stats['cache_array_bytes']=sum(x.nbytes for x in (self.rows,self.sides,self.left,self.right,self.lower,self.upper,self.left_closed,self.right_closed,self.knots))
