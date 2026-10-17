"""Bounded scalar Gauss-Jacobi setup for weight(1-u)^alpha on[0,1]."""
import math
import numpy as np

def rule(n,alpha,audit):
 if not 2<=n<=40 or alpha not in (2,3,4):raise ValueError('unsupported Jacobi parameters')
 diagonal=[];off=[];audit.add('jacobi_setup_control',100)
 for j in range(n):
  v=2*j+alpha;diagonal.append((1-alpha*alpha/(v*(v+2)))/2)
  audit.add('jacobi_coefficients',20)
  if j:
   value=4*j*j*(j+alpha)*(j+alpha)/(v*v*(v*v-1))
   off.append(math.sqrt(value)/2);audit.add('jacobi_coefficients',60)
 def count_below(x):
  q=diagonal[0]-x
  if abs(q)<1e-300:q=-1e-300
  count=int(q<0)
  for j in range(1,n):
   q=diagonal[j]-x-off[j-1]*off[j-1]/q
   if abs(q)<1e-300:q=-1e-300
   count+=int(q<0)
  audit.add('jacobi_sturm',20*n)
  return count
 nodes=[];weights=[]
 for i in range(n):
  lo=0.;hi=1.
  for _ in range(60):
   mid=(lo+hi)/2
   if count_below(mid)<=i:lo=mid
   else:hi=mid
   audit.add('jacobi_bisection_control',10)
  x=(lo+hi)/2;previous=0.;current=1.;total=1.
  for j in range(n-1):
   next_value=((x-diagonal[j])*current-(off[j-1]*previous if j else 0.))/off[j]
   previous,current=current,next_value;total+=current*current
   audit.add('jacobi_christoffel',20)
  nodes.append(x);weights.append(1/((alpha+1)*total));audit.add('jacobi_christoffel',10)
 return np.array(nodes),np.array(weights)

def outer_rule(n,audit):
 if n**3>65536:raise ValueError('outer node cap')
 rules=[rule(n,alpha,audit) for alpha in (4,3,2)]
 u,v,z=np.meshgrid(*(r[0] for r in rules),indexing='ij');wu,wv,wz=np.meshgrid(*(r[1] for r in rules),indexing='ij')
 a=1-u;b=1-v;c=1-z
 y=np.stack((u,a*v,a*b*z),axis=-1).reshape(-1,3);rest=(a*b*c).ravel();weights=(wu*wv*wz).ravel()
 audit.add('jacobi_duffy',30*n**3)
 return y,rest,weights
