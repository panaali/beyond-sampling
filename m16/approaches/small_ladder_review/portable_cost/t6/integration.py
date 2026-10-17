"""Counted cone integration around the guarded T55 triangle primitive."""
import math
import numpy as np
from .geometry import determinant
from .jacobi import outer_rule
from .triangle import Engine

class MassEngine:
 def __init__(self,audit):
  self.audit=audit;self.triangle=Engine(audit);self.rules={}
 def one(self,ray,order,inner_budget):
  audit=self.audit;det=abs(determinant(ray,audit));factor=det/(math.pi*math.pi*math.pi);audit.add('mass_prefactor',10)
  if det==0:
   audit.events['collapsed_integral_cones']+=1
   return 0.,0.,0.,dict(triangles=0,positive_quadrature_fallbacks=0,quadrature_nodes=0)
  if order not in self.rules:self.rules[order]=outer_rule(order,audit)
  y,rest,w=self.rules[order];prefix=y@ray[:,:3].T
  vertices=prefix[:,None,:]+rest[:,None,None]*ray[:,3:].T[None,:,:]
  total_weight=float(w.sum());maximum_weight=float(w.max());tolerance=inner_budget/(factor*total_weight)
  values,stats=self.triangle.batch(vertices,tolerance)
  value=factor*float(w@values)
  error=factor*min(total_weight*stats['maximum_quadrature_error_bound'],maximum_weight*stats['quadrature_error_bound'])
  series=factor*min(total_weight*stats['maximum_analytic_series_truncation_bound'],maximum_weight*stats['analytic_series_truncation_bound_sum'])
  audit.add('outer_endpoint_and_weight_arithmetic',80*len(w)+200)
  return value,error,series,stats
