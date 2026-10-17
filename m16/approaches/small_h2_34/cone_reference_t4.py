"""Independent geometric reference combining T48 geometry and T51 positive mass."""
import argparse,importlib.util,json,time,hashlib
from pathlib import Path
import numpy as np
from run_t3 import load
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--geometry',default=str(HERE.parent/'small_h2_56'/'geometry.py'));p.add_argument('--mass',default=str(HERE.parent/'small_h4_4'/'stable_volume.py'));a=p.parse_args()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
geo=module('t48_geometry',a.geometry);vol=module('t51_volume',a.mass);model=load(4);start=time.process_time();g=geo.Geometry(seconds=25)
cells=g.gates(model);correct=g.correct(model,cells);rays=g.simplices(correct);gate_rays=g.simplices([v for v,B in cells])
wi,wh,wo=model;max_affine_error=0.
for vertices,B in cells:
 x=vertices.mean(axis=0);h=np.zeros(len(wi))
 for value in x:h=np.maximum(wh@h+wi*value,0)
 max_affine_error=max(max_affine_error,float(np.max(np.abs(h-B@x))))
report=dict(geometry=a.geometry,geometry_sha256=hashlib.sha256(Path(a.geometry).read_bytes()).hexdigest(),mass=a.mass,mass_sha256=hashlib.sha256(Path(a.mass).read_bytes()).hexdigest(),geometry_cpu=time.process_time()-start,counts=g.counts.copy(),correct_simplices=len(rays),gate_simplices=len(gate_rays),gate_determinant_sum=float(np.abs(np.linalg.det(gate_rays)).sum()),correct_determinant_sum=float(np.abs(np.linalg.det(rays)).sum()),max_affine_error=max_affine_error,rows=[])
for n in (12,16,20):
 begin=time.process_time();values=np.array([vol.mass(r,n) for r in rays]);report['rows'].append(dict(order=n,accuracy=float(values[:,0].sum()),integration_only_bound=float(values[:,1].sum()),cpu_seconds=time.process_time()-begin,quadrature_nodes=len(rays)*n**3))
 (HERE/'cone_reference_t4.json').write_text(json.dumps(report,indent=2)+'\n');print(report['rows'][-1],flush=True)
begin=time.process_time();values=np.array([vol.mass(r,16) for r in gate_rays]);report['gate_mass_check']=dict(mass=float(values[:,0].sum()),integration_only_bound=float(values[:,1].sum()),cpu_seconds=time.process_time()-begin,quadrature_nodes=len(gate_rays)*16**3)
report['total_cpu']=time.process_time()-start
report['classification']='Integration error bound conditional on exact supplied geometry/arithmetic; floating clipping/tessellation not certified'
(HERE/'cone_reference_t4.json').write_text(json.dumps(report,indent=2)+'\n');print('checks',report['gate_mass_check'],report['max_affine_error'],report['gate_determinant_sum'],flush=True)
