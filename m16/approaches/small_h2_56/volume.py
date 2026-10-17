"""Positive simplex integration with a truncation bound conditional on geometry.

No random points or network evaluations. Stream cubes in chunks <=65536.
"""
from functools import lru_cache
from math import gamma,pi,factorial,lgamma,log,exp
import numpy as np
from numpy.polynomial.legendre import leggauss

@lru_cache(None)
def gl(n):
 x,w=leggauss(n);return(x+1)/2,w/2

def _block(d,n,start,end):
 x,w=gl(n);idx=np.array(np.unravel_index(np.arange(start,end),(n,)*(d-1))).T
 u=x[idx];weight=np.prod(w[idx],axis=1);y=np.empty((end-start,d));rest=np.ones(end-start)
 for j in range(d-1):
  y[:,j]=rest*u[:,j];weight*=np.power(1-u[:,j],d-2-j);rest*=1-u[:,j]
 y[:,-1]=rest
 return y,weight
@lru_cache(maxsize=12)
def _small_rule(d,n):return _block(d,n,0,n**(d-1))
def blocks(d,n):
 count=n**(d-1)
 if count<=65536:yield _small_rule(d,n);return
 for start in range(0,count,65536):yield _block(d,n,start,min(count,start+65536))

def details(rays,n):
 d=len(rays);p=d/2;det=abs(np.linalg.det(rays));const=gamma(p)*det/(2*pi**p)
 g=rays.T@rays;a=max(1/d,float(g.min()));b=float(np.diag(g).max());m=(a+b)/2;rho=max(0.,(b-a)/(b+a));k=(2*n-d+1)//2
 if k<0:bound=float('inf')
 elif rho==0:bound=0.
 else:
  ratio=rho*(p+k+1)/(k+2)
  if ratio>=1:bound=float('inf')
  else:bound=const*2/factorial(d-1)*exp(-p*log(m)+lgamma(p+k+1)-lgamma(p)-lgamma(k+2)+(k+1)*log(rho))/(1-ratio)
 return const,bound

def mass(rays,n):
 const,bound=details(rays,n);d=len(rays);total=0.;nodes=0
 for y,w in blocks(d,n):
  x=y@rays.T;q=np.sum(x*x,axis=1);total+=float(w@np.power(q,-d/2));nodes+=len(y)
 return const*total,bound,nodes

def sum_mass(rays,n):
 value=0.;bound=0.;nodes=0
 for v in rays:
  x,b,k=mass(v,n);value+=x;bound+=b;nodes+=k
 return dict(estimate=value,conditional_quadrature_error_bound=bound,nodes=nodes,order=n)
