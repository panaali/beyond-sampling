"""Deterministic TT-cross quadrature of both complete exact error-face laws."""
import importlib.util
import os
from pathlib import Path
import numpy as np
from scipy.linalg import qr

_spec=importlib.util.spec_from_file_location('_tensor_cross_walk',Path(__file__).parents[1]/'sci_tao_v2/core.py')
C=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)
_backend=Path(os.environ.get('TENSOR_CROSS_BACKEND',str(Path(__file__).parents[1]/'sci_tao_v2/compiled.py')))
WALK=C.walk_mass
if _backend.exists() and _backend.with_name('_walker.so').exists():
    _compiled_spec=importlib.util.spec_from_file_location('_tensor_cross_compiled',_backend)
    _compiled=importlib.util.module_from_spec(_compiled_spec); _compiled_spec.loader.exec_module(_compiled)
    WALK=_compiled.walk_mass
CLASS='unreviewed' 
DEDUCTIVE=False
ADVICE=None
SUPPORTS=[(16,10)]
PRECOMPUTE_FLOPS=30_000_000
FIXED_COST=True

class TensorCross:
    def __init__(self, evaluate, shape, rank=4, sweeps=2):
        self.evaluate=evaluate; self.shape=shape; self.d=len(shape)
        self.r=[1]+[min(rank,int(np.prod(shape[:k])),int(np.prod(shape[k:]))) for k in range(1,self.d)]+[1]
        self.I=[None]*(self.d+1); self.I[0]=[()]
        self.J=[None]*(self.d+1); self.J[-1]=[()]
        self.flops=0; self.conditions=[]; self.cores=[]
        for k in range(self.d-1,-1,-1):
            candidates=[(i,)+s for i in range(shape[k]) for s in self.J[k+1]]
            ix=np.linspace(0,len(candidates)-1,self.r[k],dtype=int)
            self.J[k]=[candidates[i] for i in ix]
        for _ in range(sweeps):
            self.forward(False)
            self.backward()
        self.forward(True)

    def block(self,k):
        points=[a+(i,)+b for a in self.I[k] for i in range(self.shape[k]) for b in self.J[k+1]]
        return self.evaluate(points).reshape(len(self.I[k]),self.shape[k],len(self.J[k+1]))

    def pivots(self,m, count):
        # Stable QR basis before volume heuristic; pivoted QR chooses row subset.
        q,_=np.linalg.qr(m,mode='reduced')
        ids=qr(q.T,mode='economic',pivoting=True)[2][:count]
        a,b=m.shape
        self.flops+=12*a*b*b+12*b*b*b
        return ids

    def forward(self, save):
        cores=[]
        for k in range(self.d):
            a=self.block(k); left,n,right=a.shape
            m=a.reshape(left*n,right)
            if k==self.d-1:
                cores.append(a)
                break
            piv=self.pivots(m,self.r[k+1])
            combinations=[p+(i,) for p in self.I[k] for i in range(n)]
            self.I[k+1]=[combinations[i] for i in piv]
            if save:
                cross=m[piv,:]
                s=np.linalg.svd(cross,compute_uv=False)
                self.conditions.append(dict(k=k,largest=float(s[0]),smallest=float(s[-1]),
                    effective_rank=int(np.sum(s>1e-10*s[0])) if s[0]>0 else 0,
                    condition=float(s[0]/s[-1]) if s[-1]>0 else None))
                g=m @ np.linalg.pinv(cross,rcond=1e-10)
                self.flops+=80*right**3+2*left*n*right**2
                cores.append(g.reshape(left,n,right))
        if save: self.cores=cores

    def backward(self):
        for k in range(self.d-1,0,-1):
            a=self.block(k); left,n,right=a.shape
            piv=self.pivots(a.reshape(left,n*right).T,left)
            combinations=[(i,)+s for i in range(n) for s in self.J[k+1]]
            self.J[k]=[combinations[i] for i in piv]

    def integral(self,weights):
        v=np.ones(1)
        for g,w in zip(self.cores,weights):
            a=np.einsum('ijk,j->ik',g,w)
            v=v @ a
            self.flops+=2*g.size+2*a.size
        return float(v[0])

    def predict(self,points):
        v=np.ones((len(points),1))
        for k,g in enumerate(self.cores):
            v=np.einsum('bi,ibj->bj',v,g[:,np.asarray(points)[:,k],:])
        return v[:,0]

class FaceGrid:
    def __init__(self,model,kind,order,cap=32768):
        self.model=model; self.kind=kind; self.order=order; self.T=model[2].shape[0]
        self.configs=len(C.geometry.configs(self.T,kind))
        n,w=np.polynomial.legendre.leggauss(order)
        self.nodes=(n+1)/2; self.weights=[np.ones(self.configs)/self.configs]+[w/2]*(self.T-1)
        self.shape=[self.configs]+[order]*(self.T-1)
        self.cache={}; self.cost=0.; self.cells=0; self.cap=cap

    def __call__(self,points):
        keys=[tuple(p) for p in points]
        new=list(dict.fromkeys(p for p in keys if p not in self.cache))
        if len(self.cache)+len(new)>self.cap: raise RuntimeError('face cap exceeded')
        if new:
            ix=np.asarray(new); U=np.empty((len(ix),self.T))
            U[:,0]=(ix[:,0]+.5)/self.configs
            U[:,1:]=self.nodes[ix[:,1:]]
            y,a,b,tau=C.face_points(self.T,self.kind,U)
            mass,cost,cells=WALK(self.model,y,a,b,tau,self.kind)
            self.cost+=float(cost.sum()); self.cells+=int(cells.sum())
            self.cache.update(zip(new,mass))
        return np.asarray([self.cache[p] for p in keys])

def estimate_details(model,order=3,rank=2,sweeps=2,audit=False):
    _,Z23,Z12=C.tables(model[2].shape[0]); error=0.; cost=0.; records=[]
    for ki,(kind,Z) in enumerate((('23',Z23),('12',Z12))):
        grid=FaceGrid(model,kind,order)
        fit=TensorCross(grid,grid.shape,rank,sweeps)
        integral=fit.integral(grid.weights)
        error+=Z*integral
        unit=C.fpe_flops(len(model[0]),model[2].shape[0])
        cost+=grid.cost+fit.flops/unit
        rec=dict(kind=kind,integral=integral,weighted_error=Z*integral,
                 faces=len(grid.cache),walk_fpe=grid.cost,algebra_flops=fit.flops,
                 cells=grid.cells,pivots=fit.conditions)
        if audit:
            # Uniform quantiles mapped to Gauss nodes by their product weights.
            U=C.sobol(model[2].shape[0],256,32001+ki)
            ix=np.empty_like(U,dtype=int); ix[:,0]=np.minimum((U[:,0]*grid.configs).astype(int),grid.configs-1)
            for k in range(1,model[2].shape[0]):
                ix[:,k]=np.minimum(np.searchsorted(np.cumsum(grid.weights[k]),U[:,k]),order-1)
            before=grid.cost; nf=len(grid.cache)
            actual=grid(ix); pred=fit.predict(ix); tail=actual>=np.quantile(actual,.95)
            rec['audit']=dict(new_faces=len(grid.cache)-nf,walk_fpe=grid.cost-before,
                true_mean=float(actual.mean()),predicted_mean=float(pred.mean()),
                relative_l2=float(np.linalg.norm(pred-actual)/max(np.linalg.norm(actual),1e-30)),
                top5_true_mass_share=float(actual[tail].sum()/max(actual.sum(),1e-30)),
                top5_abs_error_mass=float(np.abs(pred[tail]-actual[tail]).sum()/max(actual[tail].sum(),1e-30)),
                negative_fraction=float(np.mean(pred<0)))
        records.append(rec)
    return float(1-error),float(cost),records

def est(model,budget,rng=None):
    # Fixed predeclared default; budget does not select a test-fitted setting.
    result,cost,_=estimate_details(model,order=3,rank=2)
    return result,cost
