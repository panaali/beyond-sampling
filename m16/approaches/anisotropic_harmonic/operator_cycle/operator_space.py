"""Real tensor contraction of the unchanged supported cut quadrature."""
import importlib.util
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parent.parent/'lune_cycle'/'space.py'
spec=importlib.util.spec_from_file_location('operator_dense_space',SOURCE)
dense=importlib.util.module_from_spec(spec);spec.loader.exec_module(dense)
PHI0,WIDTH,unit=dense.PHI0,dense.WIDTH,dense.unit


def to_complex(c,M):
    c=np.asarray(c).reshape(-1,2*M+1)
    a=np.zeros_like(c,dtype=complex);a[:,M]=c[:,0]
    for k in range(1,M+1):
        a[:,M+k]=(c[:,2*k-1]-1j*c[:,2*k])/np.sqrt(2.)
        a[:,M-k]=np.conj(a[:,M+k])
    return a.ravel()


def to_real(a,M):
    a=np.asarray(a).reshape(-1,2*M+1)
    if np.max(abs(a[:,M].imag))>1e-10:raise ArithmeticError('complex constant radial mode')
    c=np.zeros(a.shape,dtype=float);c[:,0]=a[:,M].real
    for k in range(1,M+1):
        if np.max(abs(a[:,M-k]-np.conj(a[:,M+k])))>1e-10:
            raise ArithmeticError('lost real-function conjugacy')
        c[:,2*k-1]=np.sqrt(2.)*a[:,M+k].real
        c[:,2*k]=-np.sqrt(2.)*a[:,M+k].imag
    return c.ravel()


class Space(dense.Space):
    def __init__(self,normal_degree=8,tangent_degree=4,quadrature_order=64,optimized=True):
        super().__init__(normal_degree,tangent_degree,quadrature_order,optimized=True)
        self.constant=0
        # The parent's small identity allocation is included in cold setup;
        # it is unused by this matrix-free implementation.
        self.eye=None

    def one(self):
        c=np.zeros(self.size);c[0]=1.
        return c

    def mass(self,c):
        self.work['mass_real_divisions']+=1
        return float(c[0]/12)

    def angular(self,arcs):
        R=np.empty((len(arcs),self.nang,self.nang))
        R[:,0,0]=arcs[:,0].real
        root2=np.sqrt(2.)
        self.work['angular_normalization_sqrt_calls']+=1
        for k in range(1,self.M+1):
            ci,si=2*k-1,2*k
            R[:,0,ci]=R[:,ci,0]=root2*arcs[:,k].real
            R[:,0,si]=R[:,si,0]=root2*arcs[:,k].imag
            for l in range(1,self.M+1):
                cj,sj=2*l-1,2*l
                real_difference=arcs[:,abs(k-l)].real
                imag_l_minus_k=np.sign(l-k)*arcs[:,abs(k-l)].imag
                total=arcs[:,k+l]
                R[:,ci,cj]=real_difference+total.real
                R[:,si,sj]=real_difference-total.real
                R[:,ci,sj]=total.imag+imag_l_minus_k
                R[:,si,cj]=total.imag-imag_l_minus_k
        # Includes each sign multiplication and all adds; assignments do not
        # add floating arithmetic. Loop index arithmetic is tracked separately.
        self.work['angular_assembly_real_ops']+=len(arcs)*(5*self.M**2+2*self.M)
        self.work['angular_index_iterations']+=self.M**2+self.M
        self.work['angular_real_matrices']+=len(arcs)
        return R

    def cut(self,c,normal):
        c=np.asarray(c,dtype=float)
        if c.shape!=(self.size,):raise ValueError('coefficient size')
        v=unit(normal);self.work['cut_normalizations']+=1
        self.work['normalization_real_ops']+=13
        self.work['normalization_sqrt_calls']+=1
        if v[2]<0 or (v[2]==0 and (v[0]<0 or (v[0]==0 and v[1]<0))):
            self.work['operator_complement_real_ops']+=self.size
            return c-self.cut(c,-v)
        C=c.reshape(self.nrad,self.nang)
        if v[2]==0:
            R=self.angular(self.arcs(np.array([0.]),v))[0]
            self.work['real_operator_multiply_adds']+=self.nrad*self.nang**2
            self.work['azimuth_only_operators']+=1
            return (C@R.T).ravel()
        r=np.hypot(v[0],v[1]);self.work['cut_hypot_calls']+=1
        points=[-1.,1.,-r,r]
        for phi in (PHI0,PHI0+WIDTH):
            h=v[0]*np.cos(phi)+v[1]*np.sin(phi)
            points.append(-h/np.hypot(h,v[2]))
            self.work['topology_trig_calls']+=2
            self.work['cut_hypot_calls']+=1
            self.work['topology_real_ops']+=4
        points=np.unique(points)
        result=np.zeros((self.nrad,self.nang))
        for lo,hi in zip(points[:-1],points[1:]):
            if hi<=lo:continue
            z=(hi+lo)/2+(hi-lo)/2*np.sin(self.theta)
            weights=self.base*(hi-lo)/2*np.cos(self.theta)/2
            P=self.basis(z);R=self.angular(self.arcs(z,v))
            F=P@C
            G=np.einsum('kij,kj->ki',R,F,optimize=False)
            result+=P.T@(weights[:,None]*G)
            self.work['real_operator_multiply_adds']+=self.order*(2*self.nrad*self.nang+self.nang**2)
            self.work['operator_weights_real_ops']+=self.order*self.nang
            self.work['operator_accumulation_real_ops']+=self.size
            self.work['mapping_trig_calls']+=2*self.order
            self.work['mapping_real_ops']+=7*self.order
            self.work['cut_radial_panels']+=1
            self.work['cut_quadrature_nodes']+=self.order
        self.work['generic_cut_operators']+=1
        return result.ravel()
