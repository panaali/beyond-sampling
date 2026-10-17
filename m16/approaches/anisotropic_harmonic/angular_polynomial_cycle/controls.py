"""Independent polynomial primitives and matched-size interval leakage."""
import json
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import legint,legval
from scipy.integrate import quad
from poly_space import Space,PHI0,WIDTH

HERE=Path(__file__).resolve().parent
start=time.process_time();s=Space();out=dict(intervals=[],entries=[],roots=[])
full=s.interval_gram(np.array([PHI0]),np.array([PHI0+WIDTH]))[0]
left=s.interval_gram(np.array([PHI0]),np.array([PHI0+WIDTH/2]))[0]
right=s.interval_gram(np.array([PHI0+WIDTH/2]),np.array([PHI0+WIDTH]))[0]
out['identity_error']=float(np.max(abs(full-np.eye(9))))
out['half_complement_error']=float(np.max(abs(left+right-np.eye(9))))
parity=(-1.)**np.arange(9)
out['half_parity_error']=float(np.max(abs(right-left*parity[:,None]*parity[None,:])))
assert max(out[k] for k in ('identity_error','half_complement_error','half_parity_error'))<1e-12
for fraction in (.1,.5,.9):
    phi=PHI0+WIDTH*fraction
    normal=np.array([np.sin(phi),-np.cos(phi),0.])
    projected=s.cut(s.one(),normal)
    coefficients=[]
    for n in range(9):
        p=np.zeros(n+1);p[-1]=np.sqrt(2*n+1)
        antiderivative=legint(p)
        coefficients.append((legval(2*fraction-1,antiderivative)-legval(-1.,antiderivative))/2)
    expected=np.zeros_like(projected);expected[:9]=coefficients
    error=float(np.max(abs(projected-expected)))
    repeated=s.cut(projected,normal)
    norm=float(projected@projected)
    k=np.arange(1,5)
    fourier_mass=float(fraction*fraction+2*np.sum((np.sin(np.pi*k*fraction)/(np.pi*k))**2))
    row=dict(fraction=fraction,coefficient_error=error,repeated_mass=float(12*s.mass(repeated)),
        coefficient_norm_squared=norm,polynomial_deficit=fraction-norm,
        fourier_deficit=fraction-fourier_mass)
    out['intervals'].append(row)
    assert error<1e-12 and abs(row['repeated_mass']-norm)<1e-12
    gram=s.interval_gram(np.array([PHI0]),np.array([phi]))[0]
    eigen=np.linalg.eigvalsh(gram)
    assert eigen.min()>-1e-12 and eigen.max()<1+1e-12
lo,hi=PHI0+.17*WIDTH,PHI0+.63*WIDTH
gram=s.interval_gram(np.array([lo]),np.array([hi]))[0]
for i,j in ((2,7),(8,8),(0,1)):
    ci=np.zeros(i+1);ci[-1]=np.sqrt(2*i+1)
    cj=np.zeros(j+1);cj[-1]=np.sqrt(2*j+1)
    result,bound,info=quad(lambda phi:legval(2*(phi-PHI0)/WIDTH-1,ci)*
        legval(2*(phi-PHI0)/WIDTH-1,cj)/WIDTH,lo,hi,epsabs=1e-13,epsrel=1e-13,full_output=1)
    out['entries'].append(dict(i=i,j=j,error=abs(gram[i,j]-result),quad_bound=bound,
                               independent_integrand_calls=info['neval']))
    assert abs(gram[i,j]-result)<1e-12
c=np.sin(np.arange(81)*.31+.4)/(1+np.arange(81))
for normal in ([1.,0.,0.],[-np.sqrt(3),1.,0.]):
    error=float(np.max(abs(s.cut(c,normal)-c)))
    out['roots'].append(dict(normal=normal,identity_error=error));assert error<1e-12
out.update(cpu_seconds=time.process_time()-start,work=dict(s.work))
(HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(intervals=out['intervals'],maximum_entry_error=max(r['error'] for r in out['entries']),
                     cpu_seconds=out['cpu_seconds'])),flush=True)
