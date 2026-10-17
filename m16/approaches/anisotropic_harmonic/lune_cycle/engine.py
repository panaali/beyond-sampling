"""Exact-root support driver; original cone geometry with conservative audit."""
from collections import defaultdict
import hashlib
import importlib.util
from pathlib import Path
import time
import numpy as np
from space import Space,unit

GEOMETRY=Path('/workspace/clean-neurons/m16/approaches/harmonic_sphere/cones.py')
spec=importlib.util.spec_from_file_location('lune_common_geometry',GEOMETRY)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Cone:
    def __init__(self,work,inner=None):
        self.work=work;self.inner=module.Cone(3) if inner is None else inner
        work['geometry_objects']+=1
    def copy(self):
        self.work['geometry_mask_copies']+=len(self.inner.Z)
        return Cone(self.work,self.inner.copy())
    def classify(self,a):
        l,r=len(self.inner.L),len(self.inner.R)
        # Covers both dot products even when the lineality test returns early.
        self.work['geometry_scalar_upper_ops']+=6*(l+r)+4*l+5*r+10
        self.work['geometry_classifications']+=1
        return self.inner.classify(a)
    def add(self,a):
        l,r=len(self.inner.L),len(self.inner.R);pairs=r*r//4;n=3
        # Sum of lineality and ray-clip branch envelopes, though only one
        # executes. Includes row normalization and every opposite-sign pair.
        self.work['geometry_scalar_upper_ops']+=(3*n+4*n*(l+r)+4*l+4*r+
            (l+r)*(1+2*n)+3*n*(r+1)+pairs*6*n+50*(l+r+pairs+1))
        self.work['geometry_sqrt_upper_calls']+=r+2+pairs
        # Python bit masks have variable width; retain bit-work separately.
        self.work['geometry_bit_upper_ops']+=(pairs*(r+2)+r+1)*(self.inner.nc+1)
        self.work['geometry_additions']+=1
        self.inner.add(a)


def cells(work):
    eye=np.eye(3)
    for maximum in range(3):
        for target in range(3):
            if maximum==target:continue
            minimum=3-maximum-target
            first=eye[maximum]-eye[target];second=eye[target]-eye[minimum]
            for near,far in ((first,second),(second,first)):
                a=unit(near);bisector=unit(far-near)
                b=unit(bisector-(bisector@a)*a)
                frame=np.stack((a,b,np.cross(a,b)),axis=1)
                work['root_frame_scalar_ops']+=100
                work['root_frame_sqrt_calls']+=3
                yield target,a,bisector,frame


def estimate(model,normal_degree=8,tangent_degree=4,quadrature_order=64,seconds=100.):
    start=time.process_time();work=defaultdict(float)
    wi,wh,wo=(np.asarray(a,dtype=float) for a in model);H=len(wi)
    if wi.shape!=(H,) or wh.shape!=(H,H) or wo.shape!=(3,H):raise ValueError('T3 only')
    space=Space(normal_degree,tangent_degree,quadrature_order)
    root=space.one();root_total=12*space.mass(root)
    contribution=np.zeros(3);gate_partition=0.;leaves=0
    def deadline():
        if time.process_time()-start>seconds:raise TimeoutError('lune CPU cap')
    def normalize(a):
        work['plane_normalization_scalar_ops']+=13
        work['plane_normalization_sqrt_calls']+=1
        return unit(a)
    for target,a,bisector,frame in cells(work):
        cone=Cone(work);cone.add(a);cone.add(bisector)
        branches=[(cone,root.copy(),np.zeros((H,3)))]
        for t in range(3):
            after=[]
            for cone,c,A in branches:
                deadline();W=wh@A;W[:,t]+=wi
                work['gate_map_scalar_ops']+=6*H*H+H
                live=[(cone,c,np.zeros(H,bool))]
                for j in range(H):
                    work['zero_gate_comparisons']+=3
                    if not np.any(W[j]):continue
                    normal=normalize(W[j]);nextlive=[]
                    for K,g,mask in live:
                        side=K.classify(normal)
                        if side:
                            copied=mask.copy();copied[j]=side>0
                            nextlive.append((K,g,copied))
                        else:
                            gp=space.cut(g,frame.T@normal)
                            work['frame_transform_scalar_ops']+=18
                            Kp,Km=K.copy(),K.copy();Kp.add(normal);Km.add(-normal)
                            mp,mm=mask.copy(),mask.copy();mp[j]=True
                            nextlive.extend(((Kp,gp,mp),(Km,g-gp,mm)))
                            work['branch_complement_scalar_ops']+=2*space.size+3
                        work['mask_copy_entries']+=H*(1 if side else 2)
                    live=nextlive
                for K,g,mask in live:
                    after.append((K,g,W*mask[:,None]));work['gate_mask_scalar_ops']+=3*H
            branches=after
        leaves+=len(branches)
        gate_partition+=sum(space.mass(c) for _,c,_ in branches)
        work['mass_accumulation_scalar_ops']+=len(branches)+1
        for K,c,A in branches:
            deadline();L=wo@A;work['readout_map_scalar_ops']+=18*H
            accepted=True
            for other in range(3):
                if other==target:continue
                margin=L[target]-L[other]
                work['readout_margin_scalar_ops']+=3;work['zero_readout_comparisons']+=3
                if not np.any(margin):
                    work['zero_readout_margins']+=1
                    if other<target:accepted=False;break
                    continue
                normal=normalize(margin);side=K.classify(normal)
                if side<0:accepted=False;break
                if side==0:
                    c=space.cut(c,frame.T@normal);work['frame_transform_scalar_ops']+=18
                    K.add(normal)
            if accepted:
                contribution[target]+=space.mass(c);work['mass_accumulation_scalar_ops']+=1
    if abs(gate_partition-root_total)>1e-10:raise ArithmeticError('gate mass partition')
    sw=dict(space.work)
    scalar=sum(v for k,v in work.items() if k.endswith('_scalar_ops') or k=='geometry_scalar_upper_ops')
    scalar+=sum(v for k,v in sw.items() if k.endswith('_real_ops') or k=='mass_real_divisions')
    scalar+=8*sw.get('complex_gram_multiply_adds',0)+8*sw.get('complex_matvec_multiply_adds',0)
    scalar+=6*sw.get('complex_gram_weight_products',0)+2*sw.get('complex_gram_block_additions',0)
    return float(contribution.sum()),dict(N=normal_degree,M=tangent_degree,order=quadrature_order,
        coefficients=space.size,root_mass=root_total,gate_partition=gate_partition,leaves=leaves,
        contributions=contribution.tolist(),driver_work=dict(work),cut_work=sw,
        accounted_scalar_arithmetic_upper=scalar,
        opaque_kernels=[dict(name='Gauss-Legendre setup',order=space.order,calls=1),
                        dict(name='elementary real/complex functions',calls='see named counters')],
        portable_total_flops_certified=False,total_error_certified=False,
        geometry_sha256=hashlib.sha256(GEOMETRY.read_bytes()).hexdigest(),cpu_seconds=time.process_time()-start)
