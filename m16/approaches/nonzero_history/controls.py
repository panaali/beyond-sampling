"""Independent scalar-mean controls and affine/rank invariants."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from affine import calculate,ROOT2PI

HERE=Path(__file__).resolve().parent
start=time.process_time();result=dict(mean_controls=[],invariants=[])
for c,w in ((-130.,1e-5),(-2.,1.),(0.,0.),(0.,1.),(2.,1.),(130.,1e-5)):
    r=np.hypot(c,w)
    exact=(w*w/(r-c) if c<0 else r+c)/(2*ROOT2PI)
    # Direct integration over positive first input, with conditional second-input mean.
    def integrand(x):
        a=c*x
        cond=max(a,0.) if w==0 else abs(w)*np.exp(-.5*(a/abs(w))**2)/ROOT2PI+a*ndtr(a/abs(w))
        return cond*np.exp(-x*x/2)/ROOT2PI
    # Resolve the narrow origin layer explicitly in the negative-c case.
    scale=abs(w/c) if c!=0 and w!=0 else 1.
    split=min(1.,10*scale)
    numeric=quad(integrand,0,split,epsabs=1e-20,epsrel=1e-11)[0]+quad(integrand,split,np.inf,epsabs=1e-20,epsrel=1e-11)[0]
    result['mean_controls'].append(dict(c=c,w=w,formula=exact,independent=numeric,error=abs(exact-numeric)))
    assert abs(exact-numeric)<1e-12*max(1.,exact)

wi=np.array([-.8,.5]);wh=np.array([[.2,.3],[-.7,.1]])
wo=np.array([[.2,-.3],[.6,.1],[-.4,.8],[.1,.2]])
for name,model in [('zero_readout',(wi,wh,np.zeros_like(wo))),
                   ('identical_readout',(wi,wh,np.ones_like(wo))),
                   ('zero_input',(np.zeros(2),wh,wo))]:
    for mode in ('zero','plugin','exact2'):
        value,stats=calculate(model,mode,32)
        result['invariants'].append(dict(name=name,mode=mode,value=value,error=abs(value-.25),stats=stats))
        assert abs(value-.25)<stats['omitted_y_mass_bound']+1e-10
        assert abs(stats['partition_defect'])<1e-10

for mode in ('zero','plugin','exact2'):
    model=(wi,np.zeros_like(wh),wo)
    value,stats=calculate(model,mode,32)
    result['invariants'].append(dict(name='Wh0',mode=mode,value=value,stats=stats))
wh0=[r['value'] for r in result['invariants'] if r['name']=='Wh0']
assert max(wh0)-min(wh0)<1e-13
model=(wi,wh,wo)
base,stats=calculate(model,'exact2',32)
for scale in (np.array([.001,1000.]),np.array([1e-16,1e-16])):
    transformed=(wi*scale,wh*scale[:,None]/scale[None,:],wo/scale[None,:])
    value,s=calculate(transformed,'exact2',32)
    result['invariants'].append(dict(name='coordinate_scale',scale=scale.tolist(),value=value,error=abs(value-base)))
    assert abs(value-base)<1e-11
result['cpu_seconds']=time.process_time()-start
(HERE/'controls.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(cpu_seconds=result['cpu_seconds'],maximum_mean_error=max(r['error'] for r in result['mean_controls']))))
