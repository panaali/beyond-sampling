"""Counted fixed GL24 H2T5 estimator; advisory budget, no random inputs."""
import time
from ..accounting import Audit
from .geometry import enumerate_correct
from .volume import volume

def calculate(model,order=24):
 start=time.process_time();audit=Audit()
 try:
  rays,geometry,_=enumerate_correct(model,audit);geometry_cpu=time.process_time()-start
  p,bound=volume(rays,order,audit)
 except Exception as exc:
  exc.partial_stats=dict(cost=audit.report(),cpu_seconds=time.process_time()-start,order=order)
  raise
 return p,dict(order=order,geometry=geometry,correct_simplices=len(rays),geometry_cpu=geometry_cpu,
  cpu_seconds=time.process_time()-start,integration_only_bound=bound,cost=audit.report(),
  affine_only_forward_scalar_ops=65,affine_only_forward_equivalents=audit.known/65)

def est(model,budget,rng):
 try:p,s=calculate(model)
 except Exception as exc:
  if hasattr(exc,'partial_stats'):exc.partial_stats['advisory_budget']=budget
  raise
 return p,s['affine_only_forward_equivalents']
