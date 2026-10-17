"""Existing T55 primitive controls and cold high-order rule checks, no model mass."""
import json,time
from pathlib import Path
import numpy as np
from ..accounting import Audit,gauss_rule
from .triangle import Engine
HERE=Path(__file__).resolve().parent;start=time.process_time()
saved=json.loads(Path('/workspace/clean-neurons/m16/approaches/feynman_fundamentals/triangle_batch_results.json').read_text())
ref=next(x for x in saved['width_sweeps'] if x['dimension']==6);basis=np.eye(3,6);cases=[]
for kind,width in ref['case_labels']:
 if kind=='symmetric':v=np.ones((3,6))+width*basis;v/=v.sum(axis=1)[:,None]
 else:
  anchor=.2*np.ones(6)/6+.8*np.eye(6)[0]
  v=np.stack([anchor,anchor+width*(basis[1]-basis[0]),anchor+width*(basis[2]-basis[0])])
 cases.append(v)
a=Audit();engine=Engine(a);values,stats=engine.batch(np.array(cases));relative=np.abs(values/np.array(ref['positive_reference'])-1)
assert relative.max()<2e-10
constant=np.ones((1,3,6))/6;value,constant_stats=engine.batch(constant)
assert abs(value[0]-108)<1e-10
rule_checks=[]
for n in (48,64,96,128):
 b=Audit();x,w=gauss_rule(n,b);error=max(abs(float(w@x**k)-1/(k+1)) for k in range(2*n))
 assert error<2e-14
 rule_checks.append(dict(order=n,max_moment_error=error,cost=b.report()))
out=dict(cases=len(cases),source_sha256=engine.sha,max_relative_difference_from_saved_positive_reference=float(relative.max()),constant_triangle=float(value[0]),expected_constant=108,stats=stats,constant_stats=constant_stats,production_primitive_cost=a.report(),rule_checks=rule_checks,cpu_seconds=time.process_time()-start,model_mass_evaluations=0)
(HERE/'triangle_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(cases=out['cases'],max_relative_error=out['max_relative_difference_from_saved_positive_reference'],cost=a.known,cpu=out['cpu_seconds'],fallbacks=stats['positive_quadrature_fallbacks'])))
