"""Frozen seven constant-label models; no sampled inputs or prior mass tables."""
import time
START=time.process_time()
from pathlib import Path
import hashlib,json,resource
import numpy as np
import adapter
P=Path(__file__).resolve().parent

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22))
    E,guard,M,old,kernel,rescue,radial=adapter.load()
    cases=[('alloff',None),('active_identical_nonzero',[1,1,1,1]),('active_identical_zero',[0,0,0,0]),('tie_winner0',[1,1,0,-1]),('tie_winner1',[-1,2,2,0]),('tie_winner2',[-1,0,3,3]),('strict_winner3',[-1,0,1,2])]
    out=dict(complete=False,models=[],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((P/'adapter.py').read_bytes()).hexdigest(),guard_sha256=hashlib.sha256((P/'guard.py').read_bytes()).hexdigest(),truth='constant first-index argmax; iid continuous input symmetry gives1/4',random_network_inputs=0)
    def save():
        out.update(cpu_including_imports=time.process_time()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (P/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    for name,slopes in cases:
        if time.process_time()-START>18:out['stop_reason']='18CPU soft cap';save();return
        model=(np.array([1.,-1.]) if slopes is not None else np.zeros(2),np.zeros((2,2)),np.repeat(np.asarray(slopes if slopes is not None else [0,0,0,0],float)[:,None],2,axis=1))
        rays,capture=adapter.collect(model,E,guard);result=adapter.evaluate(rays,M,old,kernel,rescue,radial)
        row=dict(name=name,slopes=slopes,constant_prediction=int(np.argmax(slopes)) if slopes is not None else 0,exact_accuracy=.25,capture=capture,evaluation=result)
        if result['complete']:
            row.update(error=result['accuracy']-.25,analytic_pass=abs(result['accuracy']-.25)<=result['conditional_bound']+5e-13)
        out['models'].append(row);save()
        if row.get('analytic_pass') is False:out['stop_reason']='analytic accuracy failure';save();return
    out['complete']=True;save();print(json.dumps(dict(complete=out['complete'],cpu=out['cpu_including_imports'],models=[{k:v for k,v in r.items() if k not in ('capture','evaluation')}|{'N':r['capture']['N'],'returned':r['evaluation']['complete'],'known_mass_total':r['evaluation']['known_mass_total'],'bound':r['evaluation'].get('conditional_bound')} for r in out['models']]),indent=2))
if __name__=='__main__':main()
