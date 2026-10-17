"""Audited T55 triangle source with counted, local positive fallback only."""
import hashlib,math,types
from pathlib import Path
import numpy as np
from ..accounting import gauss_rule
SOURCE=Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals/triangle_batch.py')
EXPECTED_SHA='ed72ccfb52ee6f43afc19384dcc7f164f5022dc7c2610ba94da52f58b8e81461'

class Engine:
 def __init__(self,audit):
  self.audit=audit;self.rules={};source=SOURCE.read_text();self.sha=hashlib.sha256(source.encode()).hexdigest()
  if self.sha!=EXPECTED_SHA:raise ValueError('T55 source revision requires a fresh counter review')
  marker='from triangle import _positive_original'
  if source.count(marker)!=1:raise ValueError('unexpected T55 import layout')
  namespace=dict(_positive_original=self.positive)
  exec(compile(source.replace(marker,''),str(SOURCE),'exec'),namespace)
  self._batch=namespace['signed_l1_triangles'];audit.add('triangle_engine_setup',100)

 def rule(self,n):
  if n in self.rules:return self.rules[n],0
  x,w=gauss_rule(n,self.audit);s,t=np.meshgrid(x,x,indexing='ij')
  bary=np.array([s,(1-s)*t,(1-s)*(1-t)]).reshape(3,-1)
  weights=(w[:,None]*w[None,:]*(1-s)).reshape(-1)
  self.audit.add('positive_duffy_setup',30*n*n)
  self.rules[n]=(bary,weights)
  return self.rules[n],1

 def positive(self,vertices,exponent,tolerance,max_order):
  if exponent!=6 or vertices.shape!=(3,6):raise ValueError('counted fallback supports dimension6 only')
  self.audit.events['positive_fallback_calls']+=1
  center=vertices.sum(axis=0);center/=math.sqrt(float(center@center))
  lo=float(np.min(vertices@center))**2
  signs=np.sign(vertices.sum(axis=0));projection=float(np.min(vertices@signs))
  if projection>0:lo=max(lo,projection*projection/float(signs@signs))
  hi=float(np.max(np.sum(vertices*vertices,axis=1)));lo=min(lo,hi)
  mid=(lo+hi)/2;ratio=(hi-lo)/(hi+lo)
  self.audit.add('positive_radial_setup',512);choice=None;trials=0
  for n in (4,8,12,16,24,32,48,64,96,128):
   if n>max_order:continue
   next_ratio=ratio*(n+3)/(n+1);trials+=1;self.audit.add('positive_bound_trials',120)
   if next_ratio>=1:continue
   error=0. if ratio==0 else ((n+1)*(n+2)/2)*(ratio**n)/(mid*mid*mid*(1-next_ratio))
   if error<=tolerance:choice=n,error;break
  if choice is None:raise ArithmeticError('positive triangle error bound unavailable')
  n,error=choice;(bary,w),generated=self.rule(n)
  points=vertices.T@bary;q=np.sum(points*points,axis=0)
  value=float(w@(1/(q*q*q)))
  self.audit.add('positive_nodes',100*n*n);self.audit.add('positive_metadata',100)
  self.audit.events['positive_quadrature_nodes']+=n*n
  return value,dict(quadrature_nodes=n*n,quadrature_error_bound=error,quadrature_order=n,legendre_rule_generations=generated,legendre_rule_generation_nodes=n*generated,qmin=lo,qmax=hi,binomial_ratio=ratio,bound_trials=trials,bound_lgamma_calls=0,bound_exp_calls=0,bound_log_calls=0)

 def batch(self,vertices,tolerance=1e-11):
  if vertices.ndim!=3 or vertices.shape[1:]!=(3,6):raise ValueError('expected batch of3x6 triangles')
  count=len(vertices);self.audit.add('triangle_analytic_upper_allowance',10000*count+10000)
  self.audit.events['triangle_batches']+=1;self.audit.events['triangles']+=count
  return self._batch(vertices,exponent=6,absolute_tolerance=tolerance)
