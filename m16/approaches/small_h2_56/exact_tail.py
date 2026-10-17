"""Closed dimension-six Chebyshev tail; floating geometry remains conditional."""
from math import pi
import numpy as np

def tail(a,b,degree):
 a,b,degree=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float),np.asarray(degree,int))
 if np.any(a<=0) or np.any(b<a) or np.any(degree<0):raise ValueError('need0<a<=b and degree>=0')
 s=np.sqrt(a*b);alpha=(a+b)/(2*s);sa=np.sqrt(a);sb=np.sqrt(b);r=(sb-sa)/(sb+sa);n=degree+1
 return r**n/(s**3*(1-r))*(n*n+3*alpha*n+3*alpha*alpha-1+(2*n+3*alpha)*r/(1-r)+r*(1+r)/(1-r)**2)

def bounds(rays,orders):
 if rays.shape[1:]!=(6,6):raise ValueError('dimension6 only')
 g=np.swapaxes(rays,1,2)@rays;a=np.maximum((1-1e-12)/6,g.min(axis=(1,2))-1e-12)
 center=rays.sum(axis=2);center/=np.sqrt(np.sum(center*center,axis=1))[:,None]
 lower=np.einsum('bij,bi->bj',rays,center).min(axis=1);a=np.maximum(a,np.maximum(lower,0)**2-1e-12)
 b=np.maximum(a,np.diagonal(g,axis1=1,axis2=2).max(axis=1)+1e-12)
 return abs(np.linalg.det(rays))*tail(a,b,orders-1)/(60*pi**3)
