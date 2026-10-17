"""Conditional constant laws, Wh0 and zero-history reproduction."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.special import betainc
from engine import calculate

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from affine import calculate as old_calculate
sys.path.pop(0)
start=time.process_time();out=dict(rows=[])
for T in (4,5):
    wi=np.array([-.8,.5]);wh=np.array([[.2,.3],[-.7,.1]])
    wo=np.zeros((T,2));model=wi,wh,wo
    value,stats=calculate(model,order=32)
    beta=betainc(T-1,2,.5)
    expected={1:2*(1-beta)/T,-1:2*beta/T}
    errors=[abs(row['estimate']-expected[row['sign']]) for row in stats['signs']]
    out['rows'].append(dict(kind='zero_readout',T=T,value=value,conditional_errors=errors,stats=stats))
    assert abs(value-1/T)<stats['omitted_y_mass_bound']+1e-10
    assert max(errors)<stats['omitted_y_mass_bound']+1e-10
    wo=np.stack((np.linspace(.7,-.2,T),np.linspace(-.3,.5,T)),axis=1)
    for name,matrix,history in [('Wh0',np.zeros_like(wh),'conditional'),('zero_history',wh,'zero')]:
        model=wi,matrix,wo
        value,stats=calculate(model,order=32,history=history)
        old,oldstats=old_calculate(model,'zero',32,normal_landmarks=True)
        error=abs(value-old)
        out['rows'].append(dict(kind=name,T=T,value=value,prior=old,error=error,stats=stats))
        assert error<1e-12
out['cpu_seconds']=time.process_time()-start
(HERE/'model_controls.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(cpu_seconds=out['cpu_seconds'],maximum_reproduction_error=max(r.get('error',0.) for r in out['rows']))))
