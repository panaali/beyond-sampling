"""Original-halfspace verticalization for fresh x followed by target u."""
from collections import defaultdict
from itertools import product
import time
import numpy as np
from scipy.special import ndtr


def vertical(constraints,first_interval,limit,target_limit,side,stats):
    lo,hi=-target_limit,target_limit;left_closed=right_closed=True
    lower=[np.array([0.,first_interval[0]])]
    upper=[np.array([0.,first_interval[1]])]
    (lower if side else upper).append(np.array([1.,0.]))
    for raw,strict in constraints:
        norm=float(np.max(abs(raw)));stats['plane_normalizations']+=1
        if norm==0:
            if strict:return []
            continue
        a,b,c=raw/norm
        if b>0:lower.append(np.array([-a/b,-c/b]))
        elif b<0:upper.append(np.array([-a/b,-c/b]))
        elif a>0:
            root=-c/a
            if root>lo:lo=root;left_closed=not strict
            elif root==lo:left_closed &= not strict
        elif a<0:
            root=-c/a
            if root<hi:hi=root;right_closed=not strict
            elif root==hi:right_closed &= not strict
        elif c<0 or (strict and c==0):return []
    if hi<=lo:return []
    lines=lower+upper;knots=[lo,hi]
    for i in range(len(lines)):
        for j in range(i):
            stats['line_intersection_tests']+=1
            slope=lines[i][0]-lines[j][0]
            if slope!=0:
                root=(lines[j][1]-lines[i][1])/slope
                if lo<root<hi:knots.append(root)
    stats['sort_input_items']+=len(knots)
    knots=np.unique(knots);pieces=[]
    for index,(left,right) in enumerate(zip(knots[:-1],knots[1:])):
        mid=(left+right)/2
        low=max(lower,key=lambda z:z[0]*mid+z[1])
        high=min(upper,key=lambda z:z[0]*mid+z[1])
        stats['envelope_line_evaluations']+=len(lines)
        if high[0]*mid+high[1]<=low[0]*mid+low[1]:continue
        lc=left_closed if index==0 else True
        rc=right_closed if index==len(knots)-2 else False
        pieces.append((left,right,low,high,lc,rc))
    return pieces


class TerminalCache:
    def __init__(self,model,states,limit=6.,target_limit=4.,seconds=100.):
        start=time.process_time();self.stats=defaultdict(float)
        wi,wh,wo=model;H=len(wi);label=len(wo)-1
        if H!=2:raise ValueError('bounded prototype supports H2')
        self.limit=limit;self.target_limit=target_limit;self.count=len(states)
        self.model=model;rows=[];sides=[];pieces=[]
        for row,hidden in enumerate(states):
            base=wh@hidden;edges=[-limit,limit]
            self.stats['first_map_scalar_products']+=H*H
            for a,b in zip(wi,base):
                if a!=0:
                    root=-b/a;self.stats['first_gate_roots']+=1
                    if -limit<root<limit:edges.append(root)
            edges=np.unique(edges)
            for left,right in zip(edges[:-1],edges[1:]):
                active=base+wi*((left+right)/2)>0
                cx=wh@(wi*active);cc=wh@(base*active)
                self.stats['second_map_scalar_products']+=2*H*H
                zero=(wi==0)&(cx==0)&(cc==0)
                for pattern in product((False,True),repeat=H):
                    mask=np.asarray(pattern)
                    self.stats['second_masks_tested']+=1
                    if np.any(mask&zero):continue
                    logits=np.stack((wo@(wi*mask),wo@(cx*mask),wo@(cc*mask)),axis=1)
                    self.stats['readout_scalar_products']+=3*len(wo)*H
                    constraints=[(np.array([wi[i],cx[i],cc[i]])*(1 if mask[i] else -1),bool(mask[i]))
                                 for i in range(H) if not zero[i]]
                    reject=False
                    for other in range(len(wo)-1):
                        margin=logits[label]-logits[other]
                        self.stats['readout_margins']+=1
                        if not np.any(margin):reject=True;break
                        constraints.append((margin,True))
                    if reject:continue
                    for side in (0,1):
                        new=vertical(constraints,(left,right),limit,target_limit,side,self.stats)
                        rows.extend([row]*len(new));sides.extend([side]*len(new));pieces.extend(new)
            if time.process_time()-start>seconds:raise TimeoutError('terminal topology construction cap')
        self.rows=np.asarray(rows,dtype=int);self.sides=np.asarray(sides,dtype=int)
        self.left=np.array([p[0] for p in pieces]);self.right=np.array([p[1] for p in pieces])
        self.lower=np.array([p[2] for p in pieces]).reshape(-1,2)
        self.upper=np.array([p[3] for p in pieces]).reshape(-1,2)
        self.left_closed=np.array([p[4] for p in pieces],dtype=bool)
        self.right_closed=np.array([p[5] for p in pieces],dtype=bool)
        self.knots=np.unique(np.r_[-target_limit,0.,target_limit,self.left,self.right])
        self.stats['terminal_pieces']=len(pieces)
        self.stats['terminal_knots']=len(self.knots)
        self.stats['construction_cpu_seconds']=time.process_time()-start
        self.stats['cache_array_bytes']=sum(x.nbytes for x in (
            self.rows,self.sides,self.left,self.right,self.lower,self.upper,
            self.left_closed,self.right_closed,self.knots))

    def values(self,u,q=None):
        if not -self.target_limit<=u<=self.target_limit:raise ValueError('target outside cached domain')
        z=1-2*ndtr(-self.limit)
        if q is None:q=float((ndtr(u)-ndtr(-self.limit))/z)
        valid=((u>self.left)|((u==self.left)&self.left_closed)) & (
            (u<self.right)|((u==self.right)&self.right_closed))
        lo=self.lower[valid,0]*u+self.lower[valid,1]
        hi=self.upper[valid,0]*u+self.upper[valid,1]
        mass=np.zeros(len(lo));positive=(lo>=0)&(hi>lo);negative=(lo<0)&(hi>lo)
        mass[positive]=ndtr(-lo[positive])-ndtr(-hi[positive])
        mass[negative]=ndtr(hi[negative])-ndtr(lo[negative])
        out=np.zeros((self.count,2))
        np.add.at(out,(self.rows[valid],self.sides[valid]),mass)
        out/=z*np.array([q,1-q])[None,:]
        self.stats['piece_query_comparisons']+=8*len(self.left)
        self.stats['query_cdf_values']+=2*int(positive.sum()+negative.sum())+3
        self.stats['query_bound_scalar_ops']+=4*len(lo)
        self.stats['queries']+=1
        if out.min() < -1e-12 or out.max()>1+1e-10:
            raise ArithmeticError('target-last cached probability outside bounds')
        return out
