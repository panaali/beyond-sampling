"""Separately frozen missing retry coverage, original864 outputs immutable."""
from pathlib import Path
import itertools,json,time,hashlib
import numpy as np
from parent_controls import parent,native
HERE=Path(__file__).resolve().parent

def main():
    assert not (HERE/'retry_controls.json').exists() and not (HERE/'retry_progress.jsonl').exists()
    start=time.process_time();rows=[]
    with (HERE/'retry_progress.jsonl').open('x') as progress:
      for index,(length,pp,dm,e) in enumerate(itertools.product([3,4],[0,1],[-.5,-2.],[.01,1.])):
        z=np.tile([1.,1.,e,e],length);values=[dm,0.,0.,1.,.2,.3,.1,.2,.3,.25,0.]
        pars=dict(index=index,length=length,pp=pp,dmin=dm,e=e)
        progress.write(json.dumps(pars)+'\n');progress.flush()
        try:
            a=parent(z,1,length,pp,*values[:4],0,0,0,True,0,*values[4:]);b=native(z,1,length,pp,values)
            ia=['n0','pp','nfail','iteration','ndiv','ttype'];x=np.array(a['z']+a['values']);y=np.array(b['z']+b['values'])
            parity=all(a[k]==b[k] for k in ia) and np.array_equal(x.view(np.uint64),y.view(np.uint64));within=sum(a['counts'].values())<=a['upper']
            rows.append(dict(**pars,reference=a,native=b,parity=bool(parity),within=within,passed=bool(parity and within)))
        except Exception as exc:rows.append(dict(**pars,passed=False,error=repr(exc)))
    out=dict(cases=len(rows),rows=rows,failures=sum(not r['passed'] for r in rows),max_q5=max(r.get('reference',{}).get('children',{}).get('q5',0) for r in rows),cpu=time.process_time()-start,pins={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['q3.py','parent_controls.py','retry_controls.py','retry_controls_protocol.json']})
    (HERE/'retry_controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
