"""Law, first-index tie and state-independent terminal controls."""
import json
from pathlib import Path
import numpy as np
from diagnostic import evaluate

HERE=Path(__file__).resolve().parent
wi=np.array([-.8,.5]);wh=np.array([[.2,.3],[-.7,.1]])
rows=[]
for name,model in (
    ('constant',(wi,wh,np.zeros((4,2)))),
    ('duplicate',(wi,wh,np.tile([.3,-.7],(4,1)))),
    ('Wh0',(wi,np.zeros((2,2)),np.array([[1.,0.],[0.,1.],[-1.,0.],[-1.,-1.]])))):
    result=evaluate(model,4,4,(5,),('baseline','curvature','gate'),seconds=15.)
    error=max(r['absolute'] for r in result['rows'])
    assert error<1e-12,(name,error)
    if name!='Wh0':assert abs(result['source_quadrature_estimate']-.25)<1e-13
    assert abs(result['raw_target_rule_mass']-1)<1e-13
    rows.append(dict(name=name,result=result))
(HERE/'controls.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(maximum_interpolation_residual=max(
    r['absolute'] for entry in rows for r in entry['result']['rows']),
    total_cpu=sum(entry['result']['cpu_seconds'] for entry in rows))))
