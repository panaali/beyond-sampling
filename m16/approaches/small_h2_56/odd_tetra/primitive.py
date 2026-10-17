"""Guarded first S3 moment for the inverse-fifth affine tetrahedron integral."""
from functools import lru_cache
from pathlib import Path
import sys,hashlib
import numpy as np
from scipy.special import roots_jacobi
PARENT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PARENT))
from segment import inverse_fifth_segment
sys.path.pop(0)
SOURCE=PARENT.parent/'feynman_fundamentals'
if not (SOURCE/'chebyshev_positive.py').exists():SOURCE=Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals')
sys.path.insert(0,str(SOURCE))
from chebyshev_positive import positive_tail_bound
sys.path.pop(0)
DEPENDENCIES={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PARENT/'segment.py',SOURCE/'chebyshev_positive.py')}

@lru_cache(maxsize=12)
def fallback_rule(n):
 x,w=roots_jacobi(n,2,0);x=(x+1)/2;w=w/8
 z,v=roots_jacobi(n,1,0);z=(z+1)/2;v=v/4
 a,b=np.meshgrid(x,z,indexing='ij');weights=(w[:,None]*v[None,:]).ravel()
 y=np.stack([a,(1-a)*b],axis=-1).reshape(-1,2);rest=((1-a)*(1-b)).ravel()
 return y,rest,weights

def radial_interval(v):
 center=v.sum(axis=-2);center/=np.linalg.norm(center,axis=-1,keepdims=True)
 projection=np.sum(v*center[...,None,:],axis=-1).min(axis=-1)
 signs=np.sign(v.sum(axis=-2));qsign=np.sum(v*signs[...,None,:],axis=-1).min(axis=-1)**2/np.sum(signs*signs,axis=-1)
 return np.maximum(projection**2,qsign),np.sum(v*v,axis=-1).max(axis=-1)

def positive_fallback(v,tolerance):
 a,b=radial_interval(v);trials=0
 for n in (4,8,12,16,24,32,48,64,96,128):
  trials+=1;bound=float(positive_tail_bound(float(a),float(b),2.5,n-1))/3
  if bound<=tolerance:break
 else:raise ArithmeticError('tetra positive fallback cannot meet tolerance')
 y,rest,w=fallback_rule(n);prefix=y@v[:2];aa=prefix+rest[:,None]*v[2];bb=prefix+rest[:,None]*v[3]
 return float(w@inverse_fifth_segment(aa,bb)),bound,n*n,trials

def inverse_fifth_tetra(vertices,known_plane_normal=None,absolute_tolerance=1e-11,min_singular_ratio=1e-6,roundoff_screen=1e-10):
 original=np.asarray(vertices,float)
 if original.shape[-2:]!=(4,5):raise ValueError('need (...,4,5) T5 endpoint arrays')
 shape=original.shape[:-2];vv=original.reshape(-1,4,5);number=len(vv)
 if not np.isfinite(vv).all() or np.any((vv.min(axis=1)<0)&(vv.max(axis=1)>0)):raise ValueError('finite shared-orthant endpoints required')
 tolerances=np.broadcast_to(np.asarray(absolute_tolerance,float),shape).reshape(-1)
 if np.any(tolerances<=0):raise ValueError('positive tolerance required')
 normal=None;residual=0.
 if known_plane_normal is not None:
  normal=np.broadcast_to(np.asarray(known_plane_normal,float),shape+(5,)).reshape(-1,5)
  residual=float(np.max(np.abs(np.einsum('nvd,nd->nv',vv,normal)-1)))
  if residual>1e-10:raise ValueError('invalid known affine plane normal')
 scale=np.max(np.abs(vv),axis=(1,2))
 if np.any(scale==0):raise ValueError('zero tetrahedron')
 v=vv/scale[:,None,None];q,r=np.linalg.qr(np.swapaxes(v,1,2),mode='reduced');singular=np.linalg.svd(r,compute_uv=False);ratio=singular[:,-1]/singular[:,0]
 good=ratio>=min_singular_ratio;ids=np.flatnonzero(good);values=np.empty(number);quad_bounds=np.zeros(number);attempts=len(ids);rejections=0;maxscreen=0.
 if attempts:
  rr=r[ids];det=np.linalg.det(rr)
  c=np.einsum('ndi,nd->ni',q[ids],normal[ids]*scale[ids,None]) if normal is not None else np.linalg.solve(np.swapaxes(rr,1,2),np.ones((attempts,4,1))).squeeze(-1)
  cofactor=np.empty((attempts,4,4))
  for i in range(4):
   columns=[j for j in range(4) if j!=i]
   for k in range(4):
    rows=[j for j in range(4) if j!=k]
    cofactor[:,i,k]=(-1.)**(i+k)*np.linalg.det(rr[:,rows][:,:,columns])
  lengths=np.linalg.norm(v[ids],axis=2);unit=v[ids]/lengths[:,:,None];dot=unit@np.swapaxes(unit,1,2)
  normcofactor=np.linalg.norm(cofactor,axis=2);terms=np.empty((attempts,4))
  for i in range(4):
   j,k,l=[z for z in range(4) if z!=i]
   solid=2*np.arctan2(normcofactor[:,i]/(lengths[:,j]*lengths[:,k]*lengths[:,l]),1+dot[:,j,k]+dot[:,k,l]+dot[:,l,j])
   inward=np.sign(det)*np.sum(cofactor[:,i]*c,axis=1)/normcofactor[:,i]
   terms[:,i]=inward*solid
  numerator=terms.sum(axis=1);estimate=numerator/(3*abs(det))
  heuristic=32*np.finfo(float).eps*np.sum(abs(terms),axis=1)/np.maximum(abs(numerator),np.finfo(float).tiny)
  maxscreen=float(heuristic.max());a,b=radial_interval(v[ids]);lower=b**-2.5/6;upper=a**-2.5/6
  accept=np.isfinite(estimate)&(estimate>=lower*(1-1e-10))&(estimate<=upper*(1+1e-10))&(heuristic<=roundoff_screen)
  values[ids[accept]]=estimate[accept]/scale[ids[accept]]**5;good[ids[~accept]]=False;rejections=int((~accept).sum())
 nodes=trials=0
 for i in np.flatnonzero(~good):
  values[i],quad_bounds[i],count,attempt=positive_fallback(vv[i],float(tolerances[i]));nodes+=count;trials+=attempt
 stats=dict(tetrahedra=number,analytic_tetrahedra=int(good.sum()),positive_fallbacks=int((~good).sum()),fallback_segment_nodes=nodes,fallback_bound_trials=trials,qr_factorizations=number,svd_condition_estimates=number,determinants4=attempts,cofactor_determinants3=16*attempts,linear_solves=attempts if normal is None else 0,known_normal_projections=attempts if normal is not None else 0,facet_atan2_calls=4*attempts,numerical_screen_rejections=rejections,minimum_singular_ratio=float(ratio.min()),maximum_heuristic_relative_roundoff=maxscreen,known_plane_residual=residual,quadrature_error_bounds=quad_bounds.reshape(shape),quadrature_error_bound_sum=float(quad_bounds.sum()),maximum_quadrature_error_bound=float(quad_bounds.max()),roundoff_certified=False,dependencies=DEPENDENCIES)
 return values.reshape(shape),stats
