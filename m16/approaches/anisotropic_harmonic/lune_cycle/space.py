"""Legendre/Fourier projection supported on an exact half-order lune."""
from collections import defaultdict
import numpy as np
from numpy.polynomial.legendre import leggauss

PHI0=np.pi/3
WIDTH=np.pi/6


def unit(vector):
    value=np.asarray(vector,dtype=float);scale=max(abs(value))
    if scale==0:raise ValueError('zero cut requires explicit tie ownership')
    value=value/scale
    return value/np.sqrt(value@value)


class Space:
    def __init__(self,normal_degree=8,tangent_degree=4,quadrature_order=64):
        self.N=int(normal_degree);self.M=int(tangent_degree)
        self.nrad=self.N+1;self.nang=2*self.M+1;self.size=self.nrad*self.nang
        self.constant=self.M;self.order=quadrature_order or 64
        self.work=defaultdict(float)
        nodes,weights=leggauss(self.order)
        self.theta=nodes*np.pi/2;self.base=weights*np.pi/2
        self.work['legendre_rule_constructions']+=1
        self.work['legendre_rule_order']=self.order
        self.work['rule_scaling_real_ops']+=2*self.order
        self.eye=np.eye(self.size,dtype=complex)

    def one(self):
        result=np.zeros(self.size,dtype=complex);result[self.constant]=1.
        return result

    def mass(self,c):
        if abs(c[self.constant].imag)>1e-10:raise ArithmeticError('complex mass')
        self.work['mass_real_divisions']+=1
        return float(c[self.constant].real/12)

    def basis(self,z):
        P=np.empty((len(z),self.nrad));P[:,0]=1
        if self.N:P[:,1]=z
        for n in range(1,self.N):
            P[:,n+1]=((2*n+1)*z*P[:,n]-n*P[:,n-1])/(n+1)
            self.work['basis_recurrence_real_ops']+=6*len(z)
        norm=np.sqrt(2*np.arange(self.nrad)+1)
        self.work['basis_sqrt_calls']+=self.nrad
        self.work['basis_normalization_real_ops']+=self.nrad*(2+len(z))
        return P*norm

    def arcs(self,z,v):
        """Conditional Fourier integrals for nonnegative mode differences."""
        r=np.hypot(v[0],v[1]);self.work['cut_hypot_calls']+=1
        out=np.zeros((len(z),2*self.M+1),dtype=complex)
        if r==0:
            out[:,0]=(v[2]*z>=0)
            self.work['arc_axis_comparisons']+=len(z)
            return out
        delta=np.arctan2(v[1],v[0]);self.work['cut_atan2_calls']+=1
        denominator=r*np.sqrt(np.maximum(0.,1-z*z))
        # Very thin panels can round a mapped node to a pole. Use the
        # exact limiting empty/full arc rather than divide by zero.
        ratio=np.copysign(np.full_like(z,np.inf),-v[2]*z)
        np.divide(-v[2]*z,denominator,out=ratio,where=denominator>0)
        self.work['rounded_pole_nodes']+=int(np.count_nonzero(denominator==0))
        half=np.arccos(np.clip(ratio,-1.,1.))
        self.work['arc_sqrt_calls']+=len(z)
        self.work['arc_acos_calls']+=len(z)
        self.work['arc_ratio_real_ops']+=6*len(z)
        for wrap in (-1,0,1):
            center=delta+2*np.pi*wrap
            lo=np.maximum(PHI0,center-half)
            hi=np.minimum(PHI0+WIDTH,center+half)
            valid=hi>lo;length=np.where(valid,hi-lo,0.)
            middle=(lo+hi)/2-PHI0
            out[:,0]+=length/WIDTH
            self.work['arc_interval_real_ops']+=8*len(z)
            self.work['arc_interval_comparisons']+=3*len(z)
            for k in range(1,2*self.M+1):
                omega=2*np.pi*k/WIDTH
                contribution=2*np.sin(omega*length/2)/(omega*WIDTH)*np.exp(1j*omega*middle)
                out[:,k]+=contribution
                self.work['arc_sin_calls']+=len(z)
                self.work['arc_complex_exp_calls']+=len(z)
                self.work['arc_mode_real_ops']+=12*len(z)+3
        return out

    def matrix(self,normal):
        v=unit(normal);self.work['cut_normalizations']+=1
        self.work['normalization_real_ops']+=13
        self.work['normalization_sqrt_calls']+=1
        if v[2]<0 or (v[2]==0 and (v[0]<0 or (v[0]==0 and v[1]<0))):
            self.work['matrix_complement_real_ops']+=2*self.size**2
            return self.eye-self.matrix(-v)
        if v[2]==0:
            arc=self.arcs(np.array([0.]),v)[0]
            block=np.empty((self.nang,self.nang),complex)
            for i in range(self.nang):
                for j in range(self.nang):
                    k=j-i;block[i,j]=arc[k] if k>=0 else np.conj(arc[-k])
            self.work['azimuth_only_matrices']+=1
            return np.kron(np.eye(self.nrad),block)
        r=np.hypot(v[0],v[1]);self.work['cut_hypot_calls']+=1
        points=[-1.,1.,-r,r]
        for phi in (PHI0,PHI0+WIDTH):
            h=v[0]*np.cos(phi)+v[1]*np.sin(phi)
            points.append(-h/np.hypot(h,v[2]))
            self.work['topology_trig_calls']+=2
            self.work['cut_hypot_calls']+=1
            self.work['topology_real_ops']+=4
        points=np.unique(points)
        blocks=[np.zeros((self.nrad,self.nrad),complex) for _ in range(2*self.M+1)]
        for lo,hi in zip(points[:-1],points[1:]):
            if hi<=lo:continue
            # Sine mapping makes square-root arc endpoints smooth.
            z=(hi+lo)/2+(hi-lo)/2*np.sin(self.theta)
            weights=self.base*(hi-lo)/2*np.cos(self.theta)/2
            P=self.basis(z);arcs=self.arcs(z,v)
            for k in range(2*self.M+1):
                blocks[k]+=P.T@((weights*arcs[:,k])[:,None]*P)
            self.work['mapping_trig_calls']+=2*self.order
            self.work['mapping_real_ops']+=7*self.order
            self.work['complex_gram_multiply_adds']+=(2*self.M+1)*self.order*self.nrad**2
            self.work['complex_gram_weight_products']+=(2*self.M+1)*self.order*(1+self.nrad)
            self.work['complex_gram_block_additions']+=(2*self.M+1)*self.nrad**2
            self.work['cut_radial_panels']+=1
            self.work['cut_quadrature_nodes']+=self.order
        result=np.empty_like(self.eye)
        for i in range(self.nang):
            for j in range(self.nang):
                k=j-i
                result[i::self.nang,j::self.nang]=blocks[k] if k>=0 else np.conj(blocks[-k])
        self.work['generic_cut_matrices']+=1
        return result

    def cut(self,c,normal):
        matrix=self.matrix(normal)
        self.work['complex_matvec_multiply_adds']+=self.size**2
        return matrix@c
