"""T6 analytic last-coordinate elimination with matched Jacobi quadrature."""
from functools import lru_cache
from pathlib import Path
import importlib.util,time,hashlib
import numpy as np
from scipy.special import roots_jacobi
from volume import bernstein_bound

source=Path(__file__).resolve().parents[1]/'feynman_fundamentals'/'segment.py'
if not source.exists():source=Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals/segment.py')
spec=importlib.util.spec_from_file_location('_t55_segment_primitive',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
inverse_six_segment=module.inverse_six_segment
SOURCE_SHA256=hashlib.sha256(source.read_bytes()).hexdigest()

@lru_cache(None)
def axis(n,alpha):
 x,w=roots_jacobi(n,alpha,0);return (x+1)/2,w/2**(alpha+1)
@lru_cache(maxsize=8)
def block(n,start,end):
 idx=np.array(np.unravel_index(np.arange(start,end),(n,)*4)).T;y=np.empty((end-start,4));weight=np.ones(end-start);rest=np.ones(end-start)
 for j in range(4):
  x,w=axis(n,4-j);u=x[idx[:,j]];weight*=w[idx[:,j]];y[:,j]=rest*u;rest*=1-u
 return y,rest,weight

def plan(rays,tolerance=1e-4,orders=(4,6,8,10,12,16,20,24,28,32)):
 ns=[];bs=[]
 for v in rays:
  for n in orders:
   b=bernstein_bound(v,n,degree=n-1)
   if b<=tolerance/len(rays):break
  ns.append(n);bs.append(b)
 return np.array(ns),np.array(bs)

def grouped_mass(rays,orders,bounds,deadline=None):
 values=np.zeros(len(rays));series_bound=0.;nodes=0;counts={};const=1/(np.pi**3)
 for n in np.unique(orders):
  idx=np.where(orders==n)[0];dets=np.abs(np.linalg.det(rays[idx]));count=int(n)**4
  for start in range(0,count,65536):
   y,rest,w=block(int(n),start,min(start+65536,count))
   for j,det in zip(idx,dets):
    if deadline is not None and time.process_time()>deadline:raise TimeoutError('T6 integration CPU cap')
    prefix=y@rays[j,:,:4].T;a=prefix+rest[:,None]*rays[j,:,4];b=prefix+rest[:,None]*rays[j,:,5]
    value,stats=inverse_six_segment(a,b,diagnostics=True)
    # Closed form has no series truncation; NaN marks its unbounded floating
    # rounding, which is also excluded by the inherited geometry/real-arithmetic bound.
    trunc=stats.pop('series_truncation_bound');series_bound+=const*det*float(w@np.where(np.isnan(trunc),0.,trunc))
    values[j]+=const*det*float(w@value);nodes+=len(y)
    for key in ('endpoint_pairs','norms','dot_products','series_pairs','closed_form_pairs','series_terms_total','series_next_terms_for_bound'):
     counts[key]=counts.get(key,0)+stats[key]
 return dict(estimate=float(values.sum()),conditional_quadrature_error_bound=float(bounds.sum()),series_only_truncation_bound=series_bound,nodes=int(nodes),orders={str(n):int(np.sum(orders==n)) for n in np.unique(orders)},primitive_counts=counts,primitive_source_sha256=SOURCE_SHA256,roundoff_bound_available=False)
