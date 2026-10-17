"""Analytically remove the last coordinate of T5 simplex integration.

The inherited Bernstein bound is unchanged; only three Duffy coordinates are
numerically integrated. Geometry and setup remain separately charged.
"""
from functools import lru_cache
from math import pi
import time
import numpy as np
from volume import gl


def inverse_fifth_segment(a,b):
 """Integral_0^1 ||(1-u)a+u*b||^-5 du for non-antipodal endpoints.

 16 scalar arithmetic operations + 2 sqrt after 3 dot products. Same-orthant
 normalized rays ensure the denominators are bounded away from zero.
 """
 q0=np.sum(a*a,axis=-1);q1=np.sum(b*b,axis=-1);dot=np.sum(a*b,axis=-1)
 r=np.sqrt(q0);s=np.sqrt(q1);p=r*s;total=p+dot;n=r+s;n2=n*n
 return n*((q0+q1-p)*total+n2*p)/(3*(p*p*p)*(total*total))

@lru_cache(maxsize=12)
def outer_rule(n):
 x,w=gl(n);idx=np.array(np.unravel_index(np.arange(n**3),(n,n,n))).T
 if len(idx)>65536:raise ValueError('outer rule exceeds chunk guard')
 u=x[idx];weight=np.prod(w[idx],axis=1);y=np.empty((len(idx),3));rest=np.ones(len(idx))
 for j in range(3):
  y[:,j]=rest*u[:,j];c=1-u[:,j]
  for _ in range(3-j):weight*=c
  rest*=c
 return y,rest,weight

def grouped_mass(rays,orders,bounds,deadline=None):
 if rays.shape[1:]!=(5,5):raise ValueError('T5 only')
 values=np.zeros(len(rays));nodes=0;const=3/(8*pi*pi)
 for n in np.unique(orders):
  y,rest,w=outer_rule(int(n));idx=np.where(orders==n)[0];dets=np.abs(np.linalg.det(rays[idx]))
  for j,det in zip(idx,dets):
   if deadline is not None and time.process_time()>deadline:raise TimeoutError('integration CPU deadline')
   prefix=y@rays[j,:,:3].T
   a=prefix+rest[:,None]*rays[j,:,3]
   b=prefix+rest[:,None]*rays[j,:,4]
   integral=inverse_fifth_segment(a,b)
   values[j]=const*det*float(w@integral);nodes+=len(y)
 return dict(estimate=float(values.sum()),conditional_quadrature_error_bound=float(bounds.sum()),nodes=int(nodes),order_histogram={str(n):int(np.sum(orders==n)) for n in np.unique(orders)},arithmetic_ops_excluding_sqrt_and_setup=int(nodes*90+2*len(rays)+max(0,len(rays)-1)),sqrt_calls=int(nodes*2),determinants=len(rays))
