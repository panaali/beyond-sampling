"""Independent SciPy comparison and exact weighted moments; no model inputs."""
import json,math,time
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi
from ..accounting import Audit
from .jacobi import rule,outer_rule
HERE=Path(__file__).resolve().parent;start=time.process_time();rows=[]
for n in (4,8,16,24,32):
 for alpha in (2,3,4):
  a=Audit();x,w=rule(n,alpha,a);xx,ww=roots_jacobi(n,alpha,0);xx=(xx+1)/2;ww/=2**(alpha+1)
  errors=[abs(float(w@(x**k))-math.factorial(k)*math.factorial(alpha)/math.factorial(k+alpha+1)) for k in range(2*n)]
  assert max(errors)<2e-14 and np.max(abs(x-xx))<2e-14 and np.max(abs(w-ww))<2e-14
  rows.append(dict(order=n,alpha=alpha,max_moment_error=max(errors),max_node_difference=float(np.max(abs(x-xx))),max_weight_difference=float(np.max(abs(w-ww))),cost=a.report()))
a=Audit();y,rest,w=outer_rule(4,a)
assert abs(float(w.sum())-1/60)<1e-15
out=dict(rules=rows,outer_weight_sum=float(w.sum()),expected_outer_weight_sum=1/60,reference_scipy_rule_calls=len(rows),cpu_seconds=time.process_time()-start)
(HERE/'jacobi_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),max_moment_error=max(r['max_moment_error'] for r in rows),max_node_difference=max(r['max_node_difference'] for r in rows),max_weight_difference=max(r['max_weight_difference'] for r in rows),cpu=out['cpu_seconds'])))
