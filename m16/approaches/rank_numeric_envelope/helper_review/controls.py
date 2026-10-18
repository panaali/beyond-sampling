"""Frozen helper ABI/branch controls; no model or rank execution."""
from pathlib import Path
import ctypes as C
import hashlib
import json
import math
import time
import numpy as np
HERE=Path(__file__).resolve().parent
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
lib=C.CDLL(str(LIB));dp=C.POINTER(C.c_double);ip=C.POINTER(C.c_int64)
hyp=lib.scipy_dlapy2_64_;hyp.argtypes=[dp,dp];hyp.restype=C.c_double
scal=lib.scipy_dscal_64_;scal.argtypes=[ip,dp,dp,ip];scal.restype=None
lam=lib.scipy_dlamch_64_;lam.argtypes=[C.c_char_p,C.c_int64];lam.restype=C.c_double

def same(a,b):return (math.isnan(a) and math.isnan(b)) or np.float64(a).tobytes()==np.float64(b).tobytes()
def hypot_ref(x,y):
    if math.isnan(y):return y
    if math.isnan(x):return x
    a,b=abs(x),abs(y);w,z=max(a,b),min(a,b)
    if z==0 or w>np.finfo(float).max:return w
    r=np.float64(z)/w
    return float(np.float64(w)*np.sqrt(np.float64(1)+r*r))

def main():
    assert not (HERE/'controls.json').exists()
    start=time.process_time();proto=json.loads((HERE/'controls_protocol.json').read_text());rows=[]
    with np.errstate(all='ignore'):
      for xx in proto['dlapy2_values']:
       for yy in proto['dlapy2_values']:
        x,y=float(xx),float(yy);cx,cy=C.c_double(x),C.c_double(y)
        a=hyp(C.byref(cx),C.byref(cy));b=hypot_ref(x,y)
        rows.append(dict(kind='dlapy2',x=x.hex(),y=y.hex(),actual=a.hex(),expected=b.hex(),same=same(a,b),upper_named_units=13))
      f=np.finfo(float)
      constants=dict(E=f.eps/2,S=f.tiny,B=2.,P=f.eps,N=53.,R=1.,M=-1021.,U=f.tiny,L=1024.,O=f.max)
      for key,value in constants.items():
       for char in [key,key.lower()]:
        a=lam(char.encode(),1);b=float(value)
        rows.append(dict(kind='dlamch',char=char,actual=a.hex(),expected=b.hex(),same=same(a,b),named_float_units=0,unpriced_character_control=True))
      for n in proto['dscal_lengths']:
       for stride in proto['dscal_strides']:
        for aa in proto['dscal_alpha']:
         for stress in [False,True]:
          alpha=float(aa);x=(-1.)**np.arange(n)*(1+np.arange(n)/8)
          if stress and n:x[0]=np.inf;x[n//2]=np.nan
          buf=np.full(max(1,n*stride),-17.);buf[:n*stride:stride]=x
          expected=buf.copy()
          if alpha!=1:expected[:n*stride:stride]=np.float64(alpha)*x
          nn,ss,ca=C.c_int64(n),C.c_int64(stride),C.c_double(alpha)
          scal(C.byref(nn),C.byref(ca),buf.ctypes.data_as(dp),C.byref(ss))
          ok=all(same(a,b) for a,b in zip(buf,expected))
          rows.append(dict(kind='dscal',n=n,stride=stride,alpha=alpha.hex(),stress=stress,actual=[float(x).hex() for x in buf],expected=[float(x).hex() for x in expected],same=ok,upper_named_units=n+3))
    out=dict(rows=rows,failures=sum(not r['same'] for r in rows),numpy=np.__version__,protocol_sha256=hashlib.sha256((HERE/'controls_protocol.json').read_bytes()).hexdigest(),cpu=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
