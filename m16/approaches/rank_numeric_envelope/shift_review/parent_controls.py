"""Frozen finite Q3 parent controls; native children shared by translation."""
from pathlib import Path
import ctypes as C
import hashlib,itertools,json,time
import numpy as np
from q3 import parent,lib,IP,DP
HERE=Path(__file__).resolve().parent
fn=lib.scipy_dlasq3_64_;fn.argtypes=[IP,IP,DP,IP]+[DP]*4+[IP]*5+[DP]*7;fn.restype=None

def native(z,i0,n0,pp,values):
    arr=np.array(z);ii,nn,par,nfail,it,ndiv,ieee,typ=map(C.c_int64,(i0,n0,pp,0,0,0,1,0));vs=list(map(C.c_double,values))
    fn(C.byref(ii),C.byref(nn),arr.ctypes.data_as(DP),C.byref(par),*[C.byref(v) for v in vs[:4]],C.byref(nfail),C.byref(it),C.byref(ndiv),C.byref(ieee),C.byref(typ),*[C.byref(v) for v in vs[4:]])
    return dict(z=arr.tolist(),n0=nn.value,pp=par.value,nfail=nfail.value,iteration=it.value,ndiv=ndiv.value,ttype=typ.value,values=[v.value for v in vs])

def main():
    assert not (HERE/'parent_controls.json').exists() and not (HERE/'parent_controls_progress.jsonl').exists()
    start=time.process_time();rows=[]
    panel=itertools.product([1,2,3,4],[1,2],[0,1],[-.1,0.,.1],[0.,.5],[0.,.01,1.],['flat','ramp','doubling'])
    with (HERE/'parent_controls_progress.jsonl').open('x') as progress:
      for index,(length,i0,pp,dm,sigma,e,pattern) in enumerate(panel):
        n0=i0+length-1
        q=np.ones(n0) if pattern=='flat' else (1+np.arange(n0)/8 if pattern=='ramp' else 2.**np.arange(n0))
        z=np.array([[x,x,e,e] for x in q]).reshape(-1)
        values=[dm,sigma,0.,float(max(q)),.2,.3,.1,.2,.3,.25,0.]
        params=dict(index=index,length=length,i0=i0,pp=pp,dmin=dm,sigma=sigma,e=e,pattern=pattern)
        progress.write(json.dumps(params)+'\n');progress.flush()
        try:
            a=parent(z,i0,n0,pp,*values[:4],0,0,0,True,0,*values[4:])
            b=native(z,i0,n0,pp,values)
            ints=['n0','pp','nfail','iteration','ndiv','ttype'];floata=np.array(a['z']+a['values']);floatb=np.array(b['z']+b['values'])
            exact=all(a[k]==b[k] for k in ints) and np.array_equal(floata.view(np.uint64),floatb.view(np.uint64))
            bounded=sum(a['counts'].values())<=a['upper']
            rows.append(dict(**params,reference=a,native=b,parity=bool(exact),within=bounded,passed=bool(exact and bounded)))
        except Exception as exc:
            rows.append(dict(**params,passed=False,error=repr(exc)))
    out=dict(cases=len(rows),failures=sum(not r['passed'] for r in rows),rows=rows,cpu=time.process_time()-start,pins={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['q3.py','parent_controls.py','parent_controls_protocol.json']},scope='Finite parent translation with unchanged native children versus whole native Q3. Does not independently verify child arithmetic or prove all-input binary equivalence.')
    (HERE/'parent_controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
