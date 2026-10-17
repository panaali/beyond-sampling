"""Fixed absolute hidden-state, positive two-Gaussian polygon terminal."""
from collections import defaultdict
from itertools import product
import math,time
import numpy as np
from scipy.integrate import quad

SQRT2=math.sqrt(2.);SQRT2PI=math.sqrt(2*math.pi)


def mass(lo,hi,stats):
    if hi<=lo:return 0.
    stats['normal_cdf_values']+=2
    if lo>=0:return .5*(math.erfc(lo/SQRT2)-math.erfc(hi/SQRT2))
    return .5*(math.erfc(-hi/SQRT2)-math.erfc(-lo/SQRT2))


def vertical(constraints,limit,stats,swap=False):
    lo,hi=-limit,limit;lower=[(0.,-limit)];upper=[(0.,limit)]
    for raw,strict in constraints:
        norm=float(np.max(abs(raw)));stats['plane_normalizations']+=1
        if norm==0:
            if strict:return []
            continue
        a,b,c=raw/norm
        if swap:a,b=b,a
        if b>0:lower.append((-a/b,-c/b))
        elif b<0:upper.append((-a/b,-c/b))
        elif a>0:lo=max(lo,-c/a)
        elif a<0:hi=min(hi,-c/a)
        elif c<0 or (strict and c==0):return []
    if hi<=lo:return []
    lines=lower+upper;knots=[lo,hi]
    for i in range(len(lines)):
        for j in range(i):
            stats['line_intersection_tests']+=1
            slope=lines[i][0]-lines[j][0]
            if slope:
                root=(lines[j][1]-lines[i][1])/slope
                if lo<root<hi:knots.append(root)
    stats['sort_input_items']+=len(knots);knots=np.unique(knots);out=[]
    for left,right in zip(knots[:-1],knots[1:]):
        mid=(left+right)/2
        low=max(lower,key=lambda a:a[0]*mid+a[1]);high=min(upper,key=lambda a:a[0]*mid+a[1])
        stats['envelope_line_evaluations']+=len(lines)
        if high[0]*mid+high[1]>low[0]*mid+low[1]:out.append((left,right,low,high))
    return out


class PairTerminal:
    def __init__(self,model,hidden,label,limit=6.,diagnostic=False):
        start=time.process_time();self.stats=defaultdict(float)
        wi,wh,wo=(np.array(a,dtype=float,copy=True) for a in model)
        h=np.asarray(hidden,dtype=float);T=len(wo)
        if wi.shape!=(2,) or wh.shape!=(2,2) or wo.shape!=(T,2) or h.shape!=(2,) or np.any(h<0) or not all(np.all(np.isfinite(a)) for a in (wi,wh,wo,h)):
            raise ValueError('finite H2 model and nonnegative hidden state required')
        if not 0<=label<T or int(label)!=label or (label>T-3 and not diagnostic):raise ValueError('target must already be inserted; later labels diagnostic only')
        if not math.isfinite(limit) or limit<=0:raise ValueError('positive input limit required')
        self.limit=limit;self.model=wi,wh,wo;self.hidden=h;self.label=label;self.cells=[]
        offset=wh@h;edges=[-limit,limit];self.stats['first_map_scalar_products']+=4
        if not np.all(np.isfinite(offset)):raise ArithmeticError('nonfinite first hidden affine map')
        for a,b in zip(wi,offset):
            if a:
                root=-b/a;self.stats['first_gate_roots']+=1
                if -limit<root<limit:edges.append(root)
        edges=np.unique(edges)
        for left,right in zip(edges[:-1],edges[1:]):
            active=offset+wi*((left+right)/2)>0
            cx=wh@(wi*active);cc=wh@(offset*active);self.stats['second_map_scalar_products']+=8
            if not np.all(np.isfinite(np.r_[cx,cc])):raise ArithmeticError('nonfinite second affine map')
            zero=(cx==0)&(wi==0)&(cc==0)
            for pattern in product((False,True),repeat=2):
                mask=np.asarray(pattern);self.stats['second_masks_tested']+=1
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
        self.pieces={False:[p for c in self.cells for p in vertical(c,limit,self.stats)],True:None}
        self.stats['vertical_pieces']=len(self.pieces[False]);self.stats['construction_cpu_seconds']=time.process_time()-start

    def probabilities(self,u,swap=False,tolerance=1e-11):
        start=time.process_time();limit=self.limit
        if not -limit<u<limit or not math.isfinite(u):raise ValueError('target inside truncated support required')
        if self.pieces[swap] is None:
            self.pieces[swap]=[p for c in self.cells for p in vertical(c,limit,self.stats,True)]
            self.stats['swapped_vertical_pieces']=len(self.pieces[swap])
        lowmass=mass(-limit,u,self.stats);highmass=mass(u,limit,self.stats)
        denominators=np.array([[lowmass*lowmass,lowmass*highmass],[highmass*lowmass,highmass*highmass]])
        if not np.all(np.isfinite(denominators)) or np.any(denominators<=0):
            raise ArithmeticError('rank normalization underflow or nonfinite mass')
        raw=np.zeros((2,2));errors=np.zeros((2,2));pieces=self.pieces[swap]
        for first,second in product((0,1),repeat=2):
            for left,right,lower,upper in pieces:
                left=max(left,u) if first else left;right=right if first else min(right,u)
                if right<=left:continue
                knots=[left,right]
                for slope,intercept in (lower,upper):
                    self.stats['rank_line_intersections']+=1
                    if slope:
                        root=(u-intercept)/slope
                        if left<root<right:knots.append(root)
                knots=np.unique(knots)
                for a,b in zip(knots[:-1],knots[1:]):
                    def integrand(x):
                        self.stats['quadrature_callbacks']+=1
                        lo=lower[0]*x+lower[1];hi=upper[0]*x+upper[1]
                        if second:lo=max(lo,u)
                        else:hi=min(hi,u)
                        return math.exp(-x*x/2)/SQRT2PI*mass(lo,hi,self.stats)
                    result=quad(integrand,a,b,epsabs=max(1e-300,tolerance*denominators[first,second]/max(1,4*len(pieces))),epsrel=tolerance,limit=100,full_output=1)
                    if len(result)!=3:raise ArithmeticError('quadrature did not converge: '+repr(dict(message=result[3],interval=[a,b],u=u,rank=[first,second],lower=lower,upper=upper,hidden=self.hidden.tolist(),label=self.label,quad_value=result[0],quad_error=result[1])))
                    value,error,info=result
                    if not np.isfinite(value) or not np.isfinite(error) or value<0 or error<0:raise ArithmeticError('invalid positive polygon mass or error')
                    raw[first,second]+=value;errors[first,second]+=error;self.stats['quad_calls']+=1
                    self.stats['quad_subintervals']+=info['last']
        if swap:raw=raw.T;errors=errors.T
        conditional=raw/denominators
        if not np.all(np.isfinite(conditional)) or not np.all(np.isfinite(errors)) or np.any(conditional < -1e-12) or np.any(conditional>1+1e-10):raise ArithmeticError('conditional probability outside unit interval')
        values=np.array([conditional[0,0],(conditional[0,1]+conditional[1,0])/2])
        bounds=np.array([errors[0,0]/denominators[0,0],(errors[0,1]+errors[1,0])/(2*denominators[0,1])])
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(bounds)):raise ArithmeticError('nonfinite normalized probability or error')
        self.stats['queries']+=1;self.stats['query_cpu_seconds']+=time.process_time()-start
        return values,dict(raw_rank_masses=raw.tolist(),conditional_by_sides=conditional.tolist(),quadrature_error_estimates=bounds.tolist(),quadrature_error_certified=False)
