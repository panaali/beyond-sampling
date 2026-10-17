"""Vectorized q^-3 tail enclosures under explicit IEEE correct-rounding assumptions."""
from collections import Counter
import numpy as np
from directed import Directed

class Binary:
    def __init__(self):self.counts=Counter()
    def point(self,x):
        x=np.asarray(x,dtype=np.float64)
        if not np.isfinite(x).all() or (x<0).any():raise ValueError('finite nonnegative exact binary64 point required')
        return x,x
    def expand(self,lo,hi,kind):
        lo,hi=np.broadcast_arrays(lo,hi);self.counts[kind]+=2*lo.size;self.counts['nextafter']+=2*lo.size
        # Nonnegative analytic operands make [0,+inf] safe for unresolved operations.
        bad=np.isnan(lo)|np.isnan(hi);self.counts['unresolved_intermediate']+=int(bad.sum())
        lower=np.maximum(0.,np.nextafter(lo,-np.inf));upper=np.nextafter(hi,np.inf)
        return np.where(bad,0.,lower),np.where(bad,np.inf,upper)
    def add(self,a,b):return self.expand(a[0]+b[0],a[1]+b[1],'binary_add')
    def sub_positive(self,a,b):return self.expand(a[0]-b[1],a[1]-b[0],'binary_subtract')
    def mul(self,a,b):return self.expand(a[0]*b[0],a[1]*b[1],'binary_multiply')
    def div(self,a,b):
        low,high=self.expand(a[0]/b[1],a[1]/b[0],'binary_divide')
        # A denominator enclosure touching zero gives no finite upper information.
        return low,np.where(b[0]<=0,np.inf,high)
    def sqrt(self,a):return self.expand(np.sqrt(a[0]),np.sqrt(a[1]),'correctly_rounded_binary_sqrt')
    def power(self,a,n):
        result=self.point(1)
        while n:
            if n&1:result=self.mul(result,a)
            n//=2
            if n:a=self.mul(a,a)
        return result
    def tail(self,a,b,degree):
        if not isinstance(degree,int) or not 0<=degree<2**26:raise ValueError('integer degree0..2^26-1 required')
        a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
        if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<=0).any() or (b<a).any():raise ValueError('finite 0<a<=b required')
        constant=a==b;a=self.point(a);b=self.point(b);N=degree+1
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            s=self.sqrt(self.mul(a,b));width=self.sub_positive(b,a)
            denom=self.add(self.add(a,b),self.mul(self.point(2),s));r=self.div(width,denom)
            z=self.div(self.power(width,2),self.mul(self.mul(self.point(2),s),denom))
            alpha=self.add(self.point(1),z);omr=self.sub_positive(self.point(1),r)
            # P3=N²+3*N*alpha+2+3*(alpha²-1), with alpha²-1=z*(z+2).
            p=self.add(self.add(self.point(N*N+2),self.mul(self.point(3*N),alpha)),self.mul(self.point(3),self.mul(z,self.add(z,self.point(2)))))
            inner=self.add(self.div(p,omr),self.div(self.mul(self.add(self.point(2*N),self.mul(self.point(3),alpha)),r),self.power(omr,2)))
            inner=self.add(inner,self.div(self.mul(r,self.add(self.point(1),r)),self.power(omr,3)))
            value=self.div(self.mul(self.power(r,N),inner),self.power(s,3))
        return tuple(np.where(constant,0.,x) for x in value)
    def bound(self,a,b,degree,det_lo,det_hi):
        lo,hi=np.broadcast_arrays(np.asarray(det_lo,float),np.asarray(det_hi,float))
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo<0).any() or (hi<lo).any():raise ValueError('finite positive determinant enclosure required')
        # Setup is deliberately cold for each invocation, and its ledger is exposed.
        ctx=Directed(50);pi=ctx.pi().floats();self.counts.update({'decimal_setup_'+k:v for k,v in ctx.counts.items()})
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            v=self.div(self.mul((lo,hi),self.tail(a,b,degree)),self.mul(self.point(60),self.power(pi,3)))
        zero=(hi==0)|(np.asarray(a)==np.asarray(b))
        v=tuple(np.where(zero,0.,x) for x in v)
        self.counts['infinite_output_bounds']+=int(np.isinf(v[1]).sum())
        return v
    def sum(self,interval):
        """Pairwise outward positive sum; no ordinary floating reduction."""
        lo,hi=np.broadcast_arrays(np.asarray(interval[0],float),np.asarray(interval[1],float));lo=lo.ravel();hi=hi.ravel()
        if np.isnan(lo).any() or np.isnan(hi).any() or (lo<0).any() or (hi<lo).any():raise ValueError('ordered nonnegative intervals required')
        if not len(lo):return self.point(0)
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            while len(lo)>1:
                n=len(lo)//2;a,b=self.add((lo[:2*n:2],hi[:2*n:2]),(lo[1:2*n:2],hi[1:2*n:2]))
                if len(lo)%2:a=np.append(a,lo[-1]);b=np.append(b,hi[-1])
                lo,hi=a,b
        return lo[0],hi[0]
