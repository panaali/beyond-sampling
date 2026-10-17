"""Counted positive T5 volume with exact inverse-fifth segment integration."""
import math
import numpy as np
from ..accounting import gauss_rule
from .geometry import determinant

def outer_rule(n,audit):
 if n<2 or n**3>65536:raise ValueError('unsupported rule size')
 x,w=gauss_rule(n,audit);u,v,z=np.meshgrid(x,x,x,indexing='ij');wu,wv,wz=np.meshgrid(w,w,w,indexing='ij')
 a=1-u;b=1-v;c=1-z
 y=np.stack((u,a*v,a*b*z),axis=-1).reshape(-1,3)
 rest=(a*b*c).ravel();weight=(wu*wv*wz*a*a*a*b*b*c).ravel()
 audit.add('rule_duffy',40*n**3)
 return y,rest,weight

def segment(a,b,audit):
 q0=np.sum(a*a,axis=-1);q1=np.sum(b*b,axis=-1);dot=np.sum(a*b,axis=-1)
 r=np.sqrt(q0);s=np.sqrt(q1);p=r*s;total=p+dot;n=r+s;n2=n*n
 result=n*((q0+q1-p)*total+n2*p)/(3*(p*p*p)*(total*total))
 # Three width5 dot products <=30 scalar ops;16 endpoint arithmetic;
 # two square roots20 each. Charge50 scalar ops conservatively.
 audit.add('segment_scalar',50*result.size);audit.add('segment_sqrt',40*result.size)
 audit.events['segment_endpoint_pairs']+=result.size
 return result

def bernstein(ray,det,n,audit):
 # Same bounds as T48, without opaque determinant or vector specialfunctions.
 gram=ray.T@ray;a=max((1-1e-12)/5,float(gram.min())-1e-12)
 b=max(a,float(np.diag(gram).max())+1e-12);m=(a+b)/2;delta=(b-a)/2
 audit.add('bound_geometry',2*5**3+25+5+30)
 if delta==0:return 0.
 sa=math.sqrt(a);sb=math.sqrt(b);limit=(sb+sa)/(sb-sa);audit.add('bound_setup',2*20+10)
 best=float('inf');K=n-2
 for j in range(32):
  rho=1+(limit-1)*(.1+.85*j/31);lower=m-delta*(rho+1/rho)/2
  audit.add('bound_trials',32)
  if lower<=0:continue
  value=math.exp(-2.5*math.log(lower)-K*math.log(rho)-math.log(rho-1))
  audit.add('bound_specialfunctions',4*20);audit.add('bound_trials',12)
  best=min(best,value)
 audit.add('bound_prefactor',10)
 return det/(16*math.pi*math.pi)*best

def volume(rays,n,audit):
 y,rest,w=outer_rule(n,audit);total=0.;bound=0.
 for ray in rays:
  det=abs(determinant(ray,audit));audit.add('mass_scalar',1)
  prefix=y@ray[:,:3].T
  a=prefix+rest[:,None]*ray[:,3];b=prefix+rest[:,None]*ray[:,4]
  audit.add('segment_endpoints',50*len(w))
  integral=segment(a,b,audit);total+=3*det/(8*math.pi*math.pi)*float(w@integral)
  audit.add('mass_weighted_sum',2*len(w));audit.add('mass_scalar',20)
  bound+=bernstein(ray,det,n,audit);audit.add('bound_sum',1)
 return float(total),float(bound)
