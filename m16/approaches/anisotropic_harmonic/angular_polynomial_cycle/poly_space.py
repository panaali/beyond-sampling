"""Nonperiodic angular Legendre basis with exact-degree positive Gram rules."""
from pathlib import Path
import sys
import numpy as np
from numpy.polynomial.legendre import leggauss

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'operator_cycle'))
from operator_space import Space as RealFourier,PHI0,WIDTH
sys.path.pop(0)


class Space(RealFourier):
    def __init__(self,normal_degree=8,tangent_degree=4,quadrature_order=64,optimized=True):
        super().__init__(normal_degree,tangent_degree,quadrature_order,optimized=True)
        self.angular_degree=2*self.M
        self.angular_nodes,self.angular_weights=leggauss(self.nang)
        self.work['angular_legendre_rule_constructions']+=1
        self.work['angular_legendre_rule_order']=self.nang

    def angular_basis(self,t):
        B=np.empty(t.shape+(self.nang,));B[...,0]=1.
        if self.angular_degree:B[...,1]=t
        for n in range(1,self.angular_degree):
            B[...,n+1]=((2*n+1)*t*B[...,n]-n*B[...,n-1])/(n+1)
            self.work['angular_basis_recurrence_real_ops']+=6*t.size
        norm=np.sqrt(2*np.arange(self.nang)+1)
        self.work['angular_basis_sqrt_calls']+=self.nang
        self.work['angular_basis_normalization_real_ops']+=self.nang*(2+t.size)
        return B*norm

    def interval_gram(self,lo,hi):
        lo=np.asarray(lo);hi=np.asarray(hi)
        half=(hi-lo)/2
        phi=(hi+lo)[:,None]/2+half[:,None]*self.angular_nodes
        t=2*(phi-PHI0)/WIDTH-1
        weights=half[:,None]*self.angular_weights/WIDTH
        B=self.angular_basis(t)
        result=B.transpose(0,2,1)@(weights[...,None]*B)
        self.work['angular_node_mapping_real_ops']+=6*t.size+3*len(lo)
        self.work['angular_gram_multiply_adds']+=len(lo)*self.nang**3
        self.work['angular_gram_weights_real_ops']+=len(lo)*self.nang**2
        self.work['angular_positive_nodes']+=t.size
        return result

    def arcs(self,z,v):
        """Return real angular Gram matrices instead of Fourier moments."""
        r=np.hypot(v[0],v[1]);self.work['cut_hypot_calls']+=1
        result=np.zeros((len(z),self.nang,self.nang))
        if r==0:
            result[v[2]*z>=0]=np.eye(self.nang)
            self.work['arc_axis_comparisons']+=len(z)
            return result
        delta=np.arctan2(v[1],v[0]);self.work['cut_atan2_calls']+=1
        denominator=r*np.sqrt(np.maximum(0.,1-z*z))
        ratio=np.copysign(np.full_like(z,np.inf),-v[2]*z)
        np.divide(-v[2]*z,denominator,out=ratio,where=denominator>0)
        self.work['rounded_pole_nodes']+=int(np.count_nonzero(denominator==0))
        half=np.arccos(np.clip(ratio,-1.,1.))
        self.work['arc_sqrt_calls']+=len(z);self.work['arc_acos_calls']+=len(z)
        self.work['arc_ratio_real_ops']+=6*len(z)
        for wrap in (-1,0,1):
            center=delta+2*np.pi*wrap
            lo=np.maximum(PHI0,center-half);hi=np.minimum(PHI0+WIDTH,center+half)
            valid=hi>lo
            self.work['arc_interval_real_ops']+=4*len(z)
            self.work['arc_interval_comparisons']+=3*len(z)
            if not np.any(valid):
                self.work['empty_arc_wraps_skipped']+=1
                continue
            result[valid]+=self.interval_gram(lo[valid],hi[valid])
            self.work['angular_gram_accumulation_real_ops']+=int(valid.sum())*self.nang**2
        return result

    def angular(self,gram):
        return gram
