"""T5 deductive accuracy estimate with explicit conditional numerical bounds.

The structured API is usable now. The harness adapter reports infinite total
cost for enumerated models until setup/Qhull costs have been audited; known
arithmetic is reported separately, never silently substituted for total cost.
"""
import time
import numpy as np
try:
 from .geometry import Geometry
 from .volume import plan
 from .segment import grouped_mass
except ImportError:
 from geometry import Geometry
 from volume import plan
 from segment import grouped_mass

COST_AUDITED=False
CLASSIFICATION='deductive cone geometry with analytically bounded numerical integration'

def estimate(model,tolerance=1e-4,seconds=115):
 start=time.process_time();wi,wh,wo=(np.asarray(w,dtype=float) for w in model);wi=wi.ravel();model=(wi,wh,wo)
 if len(wo)!=5:raise ValueError('This implementation is validated for T5 only')
 if not all(np.all(np.isfinite(w)) for w in model):raise ValueError('nonfinite model weights')
 if np.all(wo[0]>=wo):
  return dict(estimate=1/len(wo),conditional_quadrature_error_bound=0.,geometry_omission_bound=0.,method='constant first-index classifier',arithmetic_ops=1,comparisons=2*wo.size,cpu_seconds=time.process_time()-start,total_cost_audited=True)
 g=Geometry(min(25,seconds));cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct)
 setup=time.process_time()-start;planning=time.process_time();orders,bounds=plan(rays,tolerance);planning=time.process_time()-planning
 result=grouped_mass(rays,orders,bounds,deadline=start+seconds)
 result.update(geometry_omission_bound=g.counts['omitted_mass_upper_bound'],geometry_counts=g.counts,geometry_cpu_seconds=setup,planning_cpu_seconds=planning,cpu_seconds=time.process_time()-start,total_cost_audited=False,unmeasured_operations=['Qhull internal work','geometry clipping and sorting','node generation','determinants','bound planning'],model_inputs_evaluated=0)
 return result

def est(model,budget,rng):
 """Provisional ARC signature. rng unused. Budget enforcement is not yet audited.

 Full enumeration runs at fixed tolerance1e-4. Infinite cost is intentional for
 unaudited models: the adapter cannot produce a false equal-compute victory.
 Use estimate() for the actual estimate, error bounds and known cost inventory.
 """
 result=estimate(model)
 if result['total_cost_audited']:
  h=len(model[0]);t=len(model[2]);unit=t*(2*h*h+2*h)+2*h*t
  return result['estimate'],(result['arithmetic_ops']+result['comparisons'])/unit
 return result['estimate'],float('inf')
