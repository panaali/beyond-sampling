"""Frozen duplicate-input-magnitude adversarial lifecycle check."""
import json,time
import numpy as np
from geometry_check import reference,compare,measure,HERE
from geometry_native import geometry

def run():
    start=time.process_time();out={'cases':[]}
    for name,mag in [('equal',np.ones(16)),('two_magnitudes',1+np.arange(16)%2)]:
        wi=mag*np.where(np.arange(16)%2==0,1.,-1.);wh=np.diag(.25+.02*np.arange(16));wo=np.eye(16)[:3];model=(wi,wh,wo)
        order=np.argsort(-abs(wi));stable=np.argsort(-abs(wi),kind='stable');row=dict(name=name,numpy_order=order.tolist(),stable_order=stable.tolist())
        old=reference(model);new=geometry(model)
        try:row.update(differences=compare(old,new,model),native_value=measure(new),reference_value=measure(old),passed=True)
        except AssertionError as exc:row.update(passed=False,error=str(exc),native_counts=new['counts'].tolist(),reference_counts=old['counts'].tolist())
        out['cases'].append(row);out['cpu_seconds']=time.process_time()-start
        (HERE/'sort_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(name,row.get('passed'),row.get('error'),order.tolist(),stable.tolist(),flush=True)
if __name__=='__main__':run()
