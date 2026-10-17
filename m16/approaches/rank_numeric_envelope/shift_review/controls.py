"""Frozen synthetic source/native shift panel; no rank or model evaluation."""
from pathlib import Path
import ctypes as C
import hashlib,json,time
import numpy as np
from q4 import q4
HERE=Path(__file__).resolve().parent
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
IP=C.POINTER(C.c_int64);DP=C.POINTER(C.c_double)
fn=C.CDLL(str(LIB)).scipy_dlasq4_64_;fn.argtypes=[IP,IP,DP,IP,IP,DP,DP,DP,DP,DP,DP,DP,IP,DP];fn.restype=None
CONFIGS=[[-.1,.2,.4,.1,.2,.4],[0,.2,.4,.1,.2,.4],[.1,.2,.4,.1,.2,.4],[.1,.3,.4,.1,.2,.4],[.2,.3,.4,.1,.2,.4],[.4,.3,.4,.1,.2,.4],[.05,.3,.5,.1,.2,.4],[.05,.2,.4,.1,.2,.4]]

def main():
    assert not (HERE/'controls.json').exists()
    start=time.process_time();rows=[]
    for length in [3,4]:
     for pp in [0,1]:
      for deflated in [0,1,2,3]:
       for config_id,vals in enumerate(CONFIGS):
        for pattern in ['flat','qd','ramp']:
         for oldtype in [-6,-18,0]:
          if pattern=='flat':z=np.ones(4*length)
          elif pattern=='qd':z=np.tile([2.,2.,.01,.01],length)
          else:z=1+np.arange(4*length)/16
          original=z.copy();ref=q4(z,1,length,pp,length+deflated,*vals,-99.,oldtype,.25)
          i0,n0,par,nin,typ=map(C.c_int64,(1,length,pp,length+deflated,oldtype))
          ds=list(map(C.c_double,vals+[-99.,.25]))
          fn(C.byref(i0),C.byref(n0),z.ctypes.data_as(DP),C.byref(par),C.byref(nin),*[C.byref(x) for x in ds[:7]],C.byref(typ),C.byref(ds[7]))
          actual=dict(tau=ds[6].value,ttype=typ.value,g=ds[7].value)
          parity=all(actual[k]==ref[k] and (k=='ttype' or float(actual[k]).hex()==float(ref[k]).hex()) for k in actual)
          unchanged=np.array_equal(z,original);units=sum(ref['counts'].values());upper={3:35,4:44}[length]
          rows.append(dict(length=length,pp=pp,deflated=deflated,config_id=config_id,pattern=pattern,oldtype=oldtype,actual=actual,reference=ref,parity=parity,z_unchanged=unchanged,units=units,upper=upper,within=units<=upper))
    out=dict(rows=rows,cases=len(rows),parity_failures=sum(not r['parity'] for r in rows),bound_failures=sum(not r['within'] for r in rows),z_failures=sum(not r['z_unchanged'] for r in rows),types=sorted(set(r['actual']['ttype'] for r in rows)),max_units_by_length={str(n):max(r['units'] for r in rows if r['length']==n) for n in [3,4]},pins={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['q4.py','controls.py','controls_protocol.json']},cpu=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
