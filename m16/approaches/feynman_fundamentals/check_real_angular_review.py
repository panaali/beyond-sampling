"""Real angular operator checked against independent positive quadrature."""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import mpmath as mp
start=time.process_time();mp.mp.dps=50
path=Path('/workspace/work/recover-small_h4_4/m16/approaches/anisotropic_harmonic/operator_cycle/operator_space.py')
spec=importlib.util.spec_from_file_location('review_real_space',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Literal normalized angular interval, distinct from arc geometry construction.
a,b=mp.mpf('0.17'),mp.mpf('0.63');M=2
moments=[b-a]+[(mp.exp(2j*mp.pi*k*b)-mp.exp(2j*mp.pi*k*a))/(2j*mp.pi*k) for k in range(1,2*M+1)]
def basis(t):
    return [mp.mpf(1)]+[f(2*mp.pi*k*t)*mp.sqrt(2) for k in range(1,M+1) for f in (mp.cos,mp.sin)]
reference=np.array([[float(mp.quad(lambda t:basis(t)[i]*basis(t)[j],[a,b]))
                     for j in range(2*M+1)] for i in range(2*M+1)])
rows=[]
for vectorized in (False,True):
    space=module.Space(normal_degree=2,tangent_degree=M,quadrature_order=8,vectorized=vectorized)
    matrix=space.angular(np.array([[complex(v) for v in moments]]))[0]
    error=float(np.max(abs(matrix-reference)))
    assert error<1e-14
    coefficients=np.arange(15,dtype=float)/13-.4
    complex_coefficients=module.to_complex(coefficients,M)
    assert np.max(abs(module.to_real(complex_coefficients,M)-coefficients))<1e-14
    # Independently evaluate the angular Fourier fields on three points for each radial row.
    field_error=0.
    for t in (.11,.37,.81):
        realvalues=coefficients.reshape(3,5)@np.array([float(x) for x in basis(mp.mpf(t))])
        complexvalues=complex_coefficients.reshape(3,5)@np.exp(2j*np.pi*np.arange(-M,M+1)*t)
        field_error=max(field_error,float(np.max(abs(realvalues-complexvalues))))
    assert field_error<1e-14
    rows.append(dict(vectorized=vectorized,maximum_angular_error=error,
                     field_error=field_error,minimum_eigenvalue=float(np.linalg.eigvalsh(matrix).min())))
out=dict(rows=rows,cpu_seconds=time.process_time()-start,
         scope='Independent literal-interval angular quadrature and real/complex field evaluation; no full estimator or cost certification.')
Path(__file__).with_name('real_angular_review_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
