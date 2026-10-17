import argparse,json,time
from pathlib import Path
import numpy as np
from run_t3 import load
from cone_estimator import calculate
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--geometry');p.add_argument('--mass');a=p.parse_args()
base=load(4);rng=np.random.default_rng(47021);cases=[]
for sigma in (.02,.10):
 for i in range(4):cases.append((f'rms{sigma}_copy{i}',tuple(w+sigma*np.sqrt(np.mean(w*w))*rng.normal(size=w.shape) for w in base),None))
for scale in (1e-16,1e16):cases.append((f'scale{scale}',(base[0]*scale,base[1],base[2]*scale),.30773074302392744))
for label,wo in [('zero',np.zeros_like(base[2])),('duplicate',np.ones_like(base[2]))]:cases.append((label,(base[0],base[1],wo),.25))
rows=[];start=time.process_time()
for name,model,expected in cases:
 try:
  value,stats,(g,cells,rays,vol)=calculate(model,a.geometry,a.mass,retain=True)
  vals=np.array([vol.mass(r,20) for r in rays]);reference=float(vals[:,0].sum());bound=float(vals[:,1].sum())
  gate_rays=g.simplices([v for v,B in cells]);det=float(np.abs(np.linalg.det(gate_rays)).sum())
  rows.append(dict(name=name,accuracy=value,reference=reference,difference=value-reference,expected=expected,
                   expected_error=value-expected if expected is not None else None,reference_bound=bound,
                   gate_determinant_sum=det,stats=stats))
  print(name,value,stats['cpu_seconds'],flush=True)
 except Exception as exc:
  rows.append(dict(name=name,error_type=type(exc).__name__,error=str(exc)));print(name,type(exc).__name__,str(exc)[:120],flush=True)
 output=dict(noise_seed=47021,rows=rows,total_cpu=time.process_time()-start,random_model_inputs=0)
 (HERE/'stress_t4.json').write_text(json.dumps(output,indent=2)+'\n')
 if time.process_time()-start>110:break
