import json,time
from pathlib import Path
import numpy as np
from native import walk
from fraction_oracle import exact_circle
HERE=Path(__file__).parent

def fixtures():
    wi=np.array([1.,-1.]);wh=np.array([[-1.,1.],[1.,-.5]])
    wo=np.array([[1.,-1.],[-1.,1.],[.5,.5]])
    yield 'ordinary',(wi,wh,wo)
    yield 'zero_readout',(wi,wh,np.zeros_like(wo))
    yield 'duplicate_readout',(wi,wh,np.array([[1.,-1.],[1.,-1.],[-1.,1.]]))
    yield 'tiny_positive_gap',(np.array([1.]),np.array([[.5]]),np.array([[0.],[1e-20],[0.]]))
    yield 'close_gate',(np.array([1.,1.]),np.array([[-1.,0.],[-1.,1e-12]]),wo)
    yield 'identical_gates',(np.array([1.,1.]),np.array([[-.5,.25],[-.5,.25]]),wo)
    yield 'T4',(wi,wh,np.array([[1.,-1.],[-1.,1.],[.5,.5],[0.,0.]]))

def main():
    started=time.process_time();out={'cases':[]}
    for name,model in fixtures():
        t=len(model[2]);u=np.zeros(t);v=np.zeros(t);u[:3]=(1/3,2/3,2/3);v[:3]=(2/3,1/3,-2/3)
        for scale in (1.,1e-12,1e12):
            scaled=(model[0]*scale,model[1],model[2]*scale);exact,s=exact_circle(scaled,u,v);p,a=walk(scaled,u[None],v[None])
            row=dict(name=name,scale=scale,exact=exact,native=float(p[0]),absolute_difference=abs(float(p[0])-exact),oracle=s,native_counts=a)
            out['cases'].append(row);assert row['absolute_difference']<1e-12 and abs(a['coverage'][0]-1)<1e-14
    caps=[];model=next(fixtures())[1]
    for cap in (0,1):
        try:walk(model,u[:3][None],v[:3][None],cap)
        except RuntimeError as exc:caps.append(dict(cap=cap,raised=str(exc)))
        else:raise AssertionError('cap returned partial integral')
    out.update(cap_checks=caps,cpu_seconds=time.process_time()-started)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print('controls',len(out['cases']),'max error',max(r['absolute_difference'] for r in out['cases']),'cpu',out['cpu_seconds'])

if __name__=='__main__':main()
