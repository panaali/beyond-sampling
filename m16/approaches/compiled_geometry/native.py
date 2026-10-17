"""ctypes interface for owned projective-circle kernel; failures never truncated."""
import ctypes
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent
_libraries={}

def walk(model,u,v,max_pieces=5000,engine='projective'):
    if engine not in ('projective','incremental','stable'):raise ValueError('unknown engine')
    wi,wh,wo=(np.ascontiguousarray(a,dtype=np.float64) for a in model)
    u,v=(np.ascontiguousarray(a,dtype=np.float64) for a in (u,v))
    if u.ndim!=2 or v.shape!=u.shape or wi.ndim!=1:raise ValueError('plane/model shape')
    n,t=u.shape;h=len(wi)
    if n<1 or t<2 or h<1 or wh.shape!=(h,h) or wo.shape!=(t,h):raise ValueError('model dimensions')
    if not all(np.isfinite(a).all() for a in (wi,wh,wo,u,v)):raise ValueError('finite arrays required')
    if not np.allclose(np.sum(u*u,axis=1),1,rtol=0,atol=1e-10) or not np.allclose(np.sum(v*v,axis=1),1,rtol=0,atol=1e-10) or not np.allclose(np.sum(u*v,axis=1),0,rtol=0,atol=1e-10):raise ValueError('orthonormal planes required')
    if not isinstance(max_pieces,int) or max_pieces<0:raise ValueError('nonnegative integer cap required')
    if engine not in _libraries:
        library=ctypes.CDLL(str(HERE/'data'/f'{engine}.so'))
        d=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');i=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        library.projective_walk.argtypes=[ctypes.c_int]*3+[d]*5+[ctypes.c_int,d,d,i]
        library.projective_walk.restype=ctypes.c_int;_libraries[engine]=library
    values=np.zeros(n);coverage=np.zeros(n);counts=np.zeros((n,7),dtype=np.int64)
    status=_libraries[engine].projective_walk(n,h,t,wi,wh,wo,u,v,max_pieces,values,coverage,counts)
    if status==1:raise RuntimeError('max_pieces exhausted; no partial circle returned')
    if status==2:raise ArithmeticError('affine gate walker failed to advance')
    if status==3:raise FloatingPointError('nonfinite affine arithmetic/root')
    if status:raise RuntimeError(f'unknown kernel status {status}')
    names=('gate_pieces','coefficient_steps','roots_examined','interior_pair_roots','classified_intervals','sort_comparisons','atan_calls')
    return values,dict(coverage=coverage.tolist(),**{key:counts[:,j].tolist() for j,key in enumerate(names)})
