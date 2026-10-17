"""Indivisible fixed GL16 deductive estimator with explicit scalar cost.

Budget is advisory: this estimator always reports its full charged cost, even
when greater than budget. No production SVD or random model inputs are used.
The error bound covers integration only, conditional on floating geometry.
"""
from pathlib import Path
import time
from accounting import Audit, enumerate_correct, volume
DEFAULT_DD=Path('/workspace/clean-neurons/m16/approaches/exact_ladder/cones.py')

def calculate(model,dd_path=DEFAULT_DD,order=16):
 start=time.process_time();audit=Audit()
 rays,geometry=enumerate_correct(model,dd_path,audit,rank_mode='bounded_qr')
 geometry_cpu=time.process_time()-start
 p,bound=volume(rays,order,audit)
 H=len(model[0]);fwd=2*(4*H*(H+1)+4*H)
 affine=4*(2*H*H+3*H-1)
 return p,dict(order=order,geometry=geometry,correct_simplices=len(rays),
  integration_only_bound=bound,cost=audit.report(),geometry_cpu=geometry_cpu,
  cpu_seconds=time.process_time()-start,conventional_forward_scalar_ops=fwd,
  conventional_forward_equivalents=audit.known/fwd,
  affine_only_forward_scalar_ops=affine,affine_only_forward_equivalents=audit.known/affine)

def est(model,budget,rng):
 p,stats=calculate(model)
 return p,stats['affine_only_forward_equivalents']
