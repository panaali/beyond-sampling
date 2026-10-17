"""Independent closed-form check of spherical area and signed facet flux."""
import json
from pathlib import Path
import numpy as np
from probe import octants,clip,area,plane_arc

lam=.3
mass=0.;derivative=0.;arcs=0
for p in octants():
    sign=np.sign(p.sum(axis=0)[0]);n=sign*np.array([1.,lam,0.]);dn=sign*np.array([0.,1.,0.])
    mass+=area(clip(p,n))/(4*np.pi)
    edge=plane_arc(p,n)
    if edge is not None:
        u,v,angle=edge
        derivative+=(dn@(u+v))*np.tan(angle/2)/np.linalg.norm(n)/(4*np.pi)
        arcs+=1
expected=1-np.arctan(lam)/np.pi
expected_derivative=-1/(np.pi*(1+lam*lam))
result=dict(lambda_value=lam,accuracy=mass,analytic_accuracy=expected,
            flux=derivative,analytic_derivative=expected_derivative,arcs=arcs,
            area_error=abs(mass-expected),flux_error=abs(derivative-expected_derivative),
            model_evaluations=0,description='Post-protocol independent geometry calibration proposed by parent; not a candidate model result')
assert result['area_error']<1e-12 and result['flux_error']<1e-12,result
Path(__file__).with_name('geometry_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
