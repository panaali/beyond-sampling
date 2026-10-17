"""One numerical coordinate for T5 Gaussian simplex mass."""
from functools import lru_cache
from math import pi
import time
import numpy as np
from scipy.special import roots_jacobi
from primitive import inverse_fifth_tetra

@lru_cache(None)
def rule(n):
 x,w=roots_jacobi(n,3,0);return (x+1)/2,w/16

def grouped_mass(rays,orders,bounds,deadline=None,batch_cones=256,inner_tolerance=1e-11,inner_budget=None,primitive_mode='original'):
 if primitive_mode not in ('original','fixed_tangent'):raise ValueError('unknown primitive mode')
 total=inner=0.;counts={};maximum_screen=0.;start=time.process_time();const=3/(8*pi*pi)
 keys=('tetrahedra','analytic_tetrahedra','positive_fallbacks','fallback_segment_nodes','fallback_bound_trials','qr_factorizations','svd_condition_estimates','determinants4','cofactor_determinants3','linear_solves','known_normal_projections','facet_atan2_calls','numerical_screen_rejections','tangent_cross_products')
 for n in np.unique(orders):
  x,w=rule(int(n));ids=np.flatnonzero(orders==n)
  for begin in range(0,len(ids),batch_cones):
   if deadline is not None and time.process_time()>deadline:raise TimeoutError('tetra outer integration CPU cap')
   chosen=ids[begin:begin+batch_cones];v=rays[chosen]
   if len(chosen)*n>65536:raise ValueError('primitive batch exceeds node guard')
   tetra=x[None,:,None,None]*v[:,:,0][:,None,None,:]+(1-x[None,:,None,None])*np.swapaxes(v[:,:,1:],1,2)[:,None,:,:]
   normal=np.sign(v.sum(axis=2))[:,None,:];factor=const*abs(np.linalg.det(v))
   tolerance=inner_tolerance if inner_budget is None else (inner_budget/(len(rays)*factor*w.sum()))[:,None]
   try:
    if primitive_mode=='original':value,stats=inverse_fifth_tetra(tetra,known_plane_normal=normal,absolute_tolerance=tolerance)
    else:
     from fixed_tangent import fixed_tangent_tetra
     value,stats=fixed_tangent_tetra(np.swapaxes(v,1,2),x,absolute_tolerance=tolerance)
   except Exception:
    np.savez_compressed(__import__('pathlib').Path(__file__).parent/'failure.npz',rays=v,order=int(n),tetra=tetra)
    raise
   total+=float(factor@(value@w));inner+=float(factor@(stats['quadrature_error_bounds']@w))
   maximum_screen=max(maximum_screen,stats['maximum_heuristic_relative_roundoff'])
   for key in keys:counts[key]=counts.get(key,0)+stats.get(key,0)
 return dict(estimate=total,conditional_outer_quadrature_bound=float(np.sum(bounds)),positive_inner_quadrature_bound=inner,primitive_counts=counts,maximum_heuristic_relative_roundoff=maximum_screen,roundoff_certified=False,requested_total_inner_budget=inner_budget,primitive_mode=primitive_mode,cpu_seconds=time.process_time()-start)
