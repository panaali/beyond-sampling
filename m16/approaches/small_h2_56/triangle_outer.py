"""Positive Jacobi outer integration around T55's guarded triangle primitive."""
from pathlib import Path
from functools import lru_cache
from math import gamma,pi
import sys,hashlib,time
import numpy as np
from scipy.special import roots_jacobi
source=Path(__file__).resolve().parents[1]/'feynman_fundamentals'
if not (source/'triangle_batch.py').exists():source=Path('/workspace/work/feynman_fundamentals/m16/approaches/feynman_fundamentals')
sys.path.insert(0,str(source))
from triangle_batch import signed_l1_triangles
sys.path.pop(0)
SOURCE_SHA256=hashlib.sha256((source/'triangle_batch.py').read_bytes()).hexdigest()
@lru_cache(maxsize=8)
def rule(d,n):
 count=n**(d-3)
 if count>65536:raise ValueError('outer rule exceeds chunk guard')
 idx=np.array(np.unravel_index(np.arange(count),(n,)*(d-3))).T
 y=np.empty((count,d-3));rest=np.ones(count);weight=np.ones(count)
 for j in range(d-3):
  x,w=roots_jacobi(n,d-2-j,0);u=(x[idx[:,j]]+1)/2;weight*=w[idx[:,j]]/2**(d-1-j);y[:,j]=rest*u;rest*=1-u
 return y,rest,weight

def endpoints(v,y,rest):
 d=len(v);prefix=y@v[:,:d-3].T
 return prefix[:,None,:]+rest[:,None,None]*v[:,d-3:].T[None,:,:]

def grouped_mass(rays,orders,bounds,deadline=None):
 d=rays.shape[1];const=gamma(d/2)/(2*pi**(d/2));total=0.;inner=0.;series=0.;counts={};maxscreen=0.;start=time.process_time()
 keys=('triangles','analytic_triangles','positive_quadrature_fallbacks','quadrature_nodes','tangent_basis_constructions','great_circle_edges','great_circle_arctan2_calls','solid_angle_arctan2_calls','elementary_tan_calls','elementary_sin_calls','elementary_cos_calls','numerical_screen_rejections','bound_trials','bound_lgamma_calls','bound_exp_calls','bound_log_calls','legendre_rule_generations','legendre_rule_generation_nodes')
 for v,n in zip(rays,orders):
  if deadline is not None and time.process_time()>deadline:raise TimeoutError('triangle pilot CPU guard')
  y,rest,w=rule(d,int(n));vv=endpoints(v,y,rest);value,stats=signed_l1_triangles(vv,absolute_tolerance=1e-11)
  factor=const*abs(np.linalg.det(v));total+=factor*float(w@value)
  inner+=factor*min(w.sum()*stats['maximum_quadrature_error_bound'],w.max()*stats['quadrature_error_bound'])
  series+=factor*min(w.sum()*stats['maximum_analytic_series_truncation_bound'],w.max()*stats['analytic_series_truncation_bound_sum'])
  maxscreen=max(maxscreen,stats['maximum_heuristic_relative_roundoff'])
  for k in keys:counts[k]=counts.get(k,0)+stats[k]
 return dict(estimate=total,conditional_outer_quadrature_bound=float(bounds.sum()),inner_positive_quadrature_bound=inner,inner_series_truncation_bound=series,primitive_counts=counts,maximum_heuristic_relative_roundoff=maxscreen,roundoff_certified=False,cpu_seconds=time.process_time()-start,source_sha256=SOURCE_SHA256)
