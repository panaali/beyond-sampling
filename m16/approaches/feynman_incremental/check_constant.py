"""Independent closed constant-label face masses; diagnostic only."""
import json,math,time
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import quad
from scipy.special import ndtr
from .estimator import Engine
from .native import walk_mass
HERE=Path(__file__).parent

def main():
    start=time.process_time();mp.mp.dps=80;engine=Engine();rows=[]
    # Hidden weights do not affect zero output; use H1 control to minimize gate work.
    model=(np.ones(1),np.ones((1,1)),np.zeros((10,1)))
    def independent(model,y,a,b,tau,tex,top=False):
        engine.faces.update(np.asarray(y,dtype="<f8").tobytes())
        for array in (a,b,tau,tex):engine.faces.update(np.asarray(array,dtype="<f8").tobytes())
        truth=[]
        for ai,bi,lo,hi in zip(a,b,tex,tau):
            mass=mp.mpf(0)
            if hi>lo:
                mass=(mp.erfc(mp.mpf(float(lo))/mp.sqrt(2))-mp.erfc(mp.mpf(float(hi))/mp.sqrt(2)))/2
            truth.append(float(mass)*(int(ai!=0)+int(bi!=0)))
        mass,cost,ledger=walk_mass(model,y,a,b,tau,tex,top=top,details=True)
        error=float(np.max(np.abs(mass-np.array(truth))))
        assert error<1e-13
        rows.append(dict(top=top,rows=len(y),absolute_difference=error,native_flops=ledger['known_flops']))
        return np.array(truth),cost
    engine.namespace['C'].walk_mass=independent
    value,_,ledger=engine.estimate(model,512)
    prior=json.loads((HERE/'estimator_results.json').read_text())['cases'][2]
    assert ledger['node_sha256']==prior['native']['ledger']['node_sha256']
    assert ledger['face_sha256']==prior['native']['ledger']['face_sha256']
    assert abs(value-prior['native']['estimate'])<1e-13
    tables={}
    for name,power,survival in (('Z23',7,True),('Z12',8,False)):
        def density(x):return math.exp(-x*x)/(2*math.pi)*float(ndtr(x))**power*(float(ndtr(-x)) if survival else 1.)
        exact,error=quad(density,-12,12,epsabs=1e-14,epsrel=1e-13)
        original=getattr(engine.core,name)
        tables[name]=dict(table=original,adaptive_integral=exact,adaptive_error=error,difference=original-exact)
        assert abs(original-exact)<1e-12
    out=dict(complete=True,independent_estimate=value,analytic_truth=.1,bias=value-.1,
             maximum_face_difference=max(r['absolute_difference'] for r in rows),face_calls=len(rows),
             table_checks=tables,rows=rows,node_sha256=ledger['node_sha256'],face_sha256=ledger['face_sha256'],
             diagnostic_native_flops=sum(r['native_flops'] for r in rows),cpu_seconds=time.process_time()-start,
             model='H1 T10 zero output; same output rule and model-free nodes as frozen H16 control',
             production_cost_claim=False)
    (HERE/'constant_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('independent_estimate','bias','maximum_face_difference','cpu_seconds')}))
if __name__=='__main__':main()
