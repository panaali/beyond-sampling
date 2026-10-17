"""Native H2T5 reference geometry with unchanged counted T71 GL24 mass."""
from pathlib import Path
import sys,time
import numpy as np
from geometry_native import geometry
T71=Path(__file__).resolve().parents[2]/'small_h2_56/odd_tetra'
sys.path.insert(0,str(T71))
from counted import Audit,volume
import estimator as prior
CLASSIFICATION='deductive native DD and analytic tetrahedron moments'
COST_REVIEWED=False

def calculate(model):
    start=time.process_time();wall=time.perf_counter();native=geometry(model)
    geometry_cpu=time.process_time()-start;geometry_wall=time.perf_counter()-wall
    rays=np.swapaxes(native['good'],1,2).copy();audit=Audit()
    begin=time.process_time();masswall=time.perf_counter();value,info=volume(rays,audit,order=24,inner_budget=1e-9)
    return value,dict(geometry_cpu_seconds=geometry_cpu,geometry_wall_seconds=geometry_wall,mass_cpu_seconds=time.process_time()-begin,mass_wall_seconds=time.perf_counter()-masswall,cpu_seconds=time.process_time()-start,wall_seconds=time.perf_counter()-wall,geometry_counts=native['counts'].tolist(),correct_simplices=len(rays),integration=info,mass_only_cost=audit.report(),native_geometry_portable_cost='unaudited opaque SVD/lifecycle',total_forward_equivalents=float('inf'),geometry_and_roundoff_certified=False)

def est(model,budget,rng):
    value,stats=calculate(model)
    return value,float('inf')
