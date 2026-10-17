"""Deterministic T4 Gaussian cone integration with an integration-only error bound.

The estimator reuses T48 geometry. Floating clipping/tessellation is not certified.
"""
from pathlib import Path
import hashlib,time,types
import numpy as np
from stable_volume import mass,adaptive_mass


def load_geometry(path=None):
    path=Path(path) if path else Path(__file__).resolve().parent.parent/'small_h2_56'/'geometry.py'
    source=path.read_bytes()
    module=types.ModuleType('t51_geometry_snapshot')
    exec(compile(source,str(path),'exec'),module.__dict__)
    return module,hashlib.sha256(source).hexdigest()


def cone_estimate(model,order=20,geometry_path=None,max_cpu=110.,check_partition=False,integration_tolerance=None):
    wi,wh,wo=(np.asarray(w,dtype=float) for w in model)
    if wi.ndim!=1 or wh.shape!=(len(wi),len(wi)) or wo.shape!=(4,len(wi)):
        raise ValueError('Requires Wi(H),Wh(H,H),Wo(4,H)')
    start=time.process_time();wall=time.perf_counter()
    geometry,source_sha=load_geometry(geometry_path)
    g=geometry.Geometry(seconds=max_cpu)
    cells=g.gates((wi,wh,wo));correct=g.correct((wi,wh,wo),cells);rays=g.simplices(correct)
    geometry_cpu=time.process_time()-start
    estimate=0.;bound=0.;nodes=0;orders={}
    for i,r in enumerate(rays):
        if integration_tolerance is None:
            value,error=mass(r,order);chosen=order
        else:
            value,error,meta=adaptive_mass(r,integration_tolerance);chosen=meta['order']
        nodes+=chosen**3;orders[chosen]=orders.get(chosen,0)+1
        estimate+=value;bound+=error
        if i%128==0 and time.process_time()-start>max_cpu:
            raise TimeoutError('cone estimator CPU cap')
    production_cpu=time.process_time()-start
    stats=dict(geometry_sha256=source_sha,order=order,accuracy_simplices=len(rays),quadrature_nodes=nodes,order_counts=orders,requested_integration_tolerance=integration_tolerance,
               integration_only_bound=bound,geometry_cpu=geometry_cpu,production_cpu=production_cpu,
               determinant_sum=float(np.abs(np.linalg.det(rays)).sum()),geometry_counts=g.counts.copy())
    if check_partition:
        gate_rays=g.simplices([v for v,A in cells]);mass_sum=0.;mass_bound=0.
        for r in gate_rays:
            v,b=mass(r,16);mass_sum+=v;mass_bound+=b
        stats['partition']=dict(mass=mass_sum,integration_only_bound=mass_bound,determinant_sum=float(np.abs(np.linalg.det(gate_rays)).sum()),simplices=len(gate_rays))
    stats.update(total_cpu=time.process_time()-start,wall_seconds=time.perf_counter()-wall)
    return float(estimate),stats


def est(model,budget=None,rng=None):
    """Fixed GL20; ignores budget, rng. Returns CPU-calibrated forward equivalents.

    Timing-only deterministic-zero forwards do not affect the accuracy estimate.
    Calibration is charged. This is hardware-dependent, not a scalar-FLOP ledger.
    """
    value,stats=cone_estimate(model)
    wi,wh,wo=model;x=np.zeros((32768,4));h=np.zeros((len(x),len(wi)))
    start=time.process_time()
    for k in range(4):h=np.maximum(h@wh.T+x[:,k,None]*wi,0)
    h@wo.T
    elapsed=time.process_time()-start
    return value,(stats['production_cpu']+elapsed)/(elapsed/len(x))
