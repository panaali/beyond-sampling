import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
import json
from pathlib import Path
import time
import numpy as np
from check_moments import mp_moments
from moments import unit,tail_mass
from positive import polygon_tail_mass

start=time.process_time()
cases=[("octant",np.eye(3))]
cases.append(("oblique",np.array([unit(v) for v in ((1,.2,.1),(1,-.3,.8),(.1,.8,.6))])))
for eps in (1e-3,1e-7):
    cases.append((f"equatorial_{eps}",np.array([unit(v) for v in (
        (eps,1,0),(eps,np.cos(.4),np.sin(.4)),(2*eps,np.cos(.2),np.sin(.2)))])))
for eps in (1e-3,1e-8):
    cases.append((f"thin_polar_{eps}",np.array([unit(v) for v in ((1,0,0),(1,eps,0),(1,0,eps))])))
failure=json.loads(Path(__file__).with_name("trained_moment_failure.json").read_text())
cases.append(("trained_H4T4_failure",np.array(failure["vertices"])))
rows=[]
for name,vertices in cases:
    reference=mp_moments(vertices,9)
    for k in (1,2,4,8,10):
        target=float(k/(2*np.pi)*reference[k-1])
        for tolerance in (1e-8,1e-12):
            value,meta=polygon_tail_mass(vertices,k,tolerance)
            difference=abs(value-target)
            rows.append(dict(case=name,k=k,tolerance=tolerance,value=value,
                             high_precision_reference=target,absolute_difference=difference,
                             exceeds_quadrature_bound=difference>meta["quadrature_error_bound"],
                             metadata=meta))
output=dict(rows=rows,cpu_seconds=time.process_time()-start,
            total_nodes=sum(r["metadata"]["quadrature_nodes"] for r in rows),
            max_order=max(r["metadata"]["max_order"] for r in rows),
            max_absolute_difference=max(r["absolute_difference"] for r in rows),
            roundoff_exceedances=[dict(case=r["case"],k=r["k"],tolerance=r["tolerance"],
                                      difference=r["absolute_difference"],
                                      quadrature_bound=r["metadata"]["quadrature_error_bound"])
                                  for r in rows if r["exceeds_quadrature_bound"]])
Path(__file__).with_name("positive_results.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({key:value for key,value in output.items() if key!="rows"}))
