"""T4 generic deductive cone estimator using T48 geometry and T51 positive mass.
The returned integration bound is conditional on supplied floating geometry.
"""
from pathlib import Path
import importlib.util,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent

def dependency(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def calculate(model,geometry_path=None,mass_path=None,order=16,retain=False):
    if len(model[2])!=4:raise ValueError('T4 only')
    geometry_path=Path(geometry_path or HERE.parent/'small_h2_56'/'geometry.py')
    mass_path=Path(mass_path or HERE.parent/'small_h4_4'/'stable_volume.py')
    start=time.process_time();geo=dependency('cone_geometry_t47',geometry_path);vol=dependency('cone_mass_t47',mass_path)
    g=geo.Geometry(seconds=30);cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct)
    geometry_cpu=time.process_time()-start
    values=np.array([vol.mass(ray,order) for ray in rays])
    p=float(values[:,0].sum()) if len(values) else 0.
    bound=float(values[:,1].sum()) if len(values) else 0.
    stats=dict(order=order,geometry_cpu=geometry_cpu,cpu_seconds=time.process_time()-start,
        integration_only_bound=bound,geometry_counts=g.counts.copy(),simplices=len(rays),
        quadrature_nodes=len(rays)*order**3,
        ray_l1_error=float(np.max(np.abs(np.sum(np.abs(rays),axis=1)-1))) if len(rays) else 0.,
        same_orthant_rays=bool(np.all(np.min(rays,axis=2)*np.max(rays,axis=2)>=-1e-14)) if len(rays) else True,
        geometry_sha256=hashlib.sha256(geometry_path.read_bytes()).hexdigest(),mass_sha256=hashlib.sha256(mass_path.read_bytes()).hexdigest())
    if retain:return p,stats,(g,cells,rays,vol)
    return p,stats

def forward(model,x):
    wi,wh,wo=model;h=np.zeros((len(x),len(wi)))
    for t in range(x.shape[1]):h=np.maximum(h@wh.T+x[:,t,None]*wi,0)
    return h@wo.T

def est(model,budget,rng):
    """Indivisible fixedGL16; actual cost is hardware-dependent CPU-equivalent forwards.
    Deterministic timing outputs are discarded. Calibration cost is charged too.
    """
    p,s=calculate(model)
    x=np.zeros((65536,4));start=time.process_time();forward(model,x);cal=time.process_time()-start
    return p,(s['cpu_seconds']+cal)/(cal/len(x))
