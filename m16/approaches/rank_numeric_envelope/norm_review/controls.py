"""Frozen finite native norm controls; same-order longdouble reference."""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
fn=C.CDLL(str(LIB)).scipy_dnrm2_64_
fn.argtypes=[C.POINTER(C.c_int64),C.POINTER(C.c_double),C.POINTER(C.c_int64)]
fn.restype=C.c_double

def ref(x):
    if len(x)==0: return 0.0
    if len(x)==1: return float(abs(x[0]))
    s=np.zeros(4,dtype=np.longdouble)
    bulk=len(x)//8*8
    for j in range(bulk):
        y=np.longdouble(x[j]);s[j%4]+=y*y
    for j in range(bulk,len(x)):
        y=np.longdouble(x[j]);s[0]+=y*y
    return float(np.sqrt(((s[0]+s[2])+s[1])+s[3]))

def main():
    assert not (HERE/'controls.json').exists(), "Refuse to overwrite an attempted panel"
    start=time.process_time()
    proto=json.loads((HERE/'controls_protocol.json').read_text())
    rows=[]
    for n in proto['lengths']:
      for stride in proto['strides']:
       for pattern in proto['patterns']:
        x=1+np.arange(n,dtype=float)/8
        if pattern=='alternating': x*=(-1.)**np.arange(n)
        if pattern=='tiny': x*=1e-310
        if pattern=='huge': x*=1e300
        if pattern=='mixed': x=np.where(np.arange(n)%2,1e-310,1e300)
        if pattern=='zero': x=np.full(n,-0.)
        if pattern in ('inf','nan') and n: x[n//2]=float(pattern)
        buf=np.full(max(1,n*stride),-17.)
        if n: buf[:n*stride:stride]=x
        nn,ss=C.c_int64(n),C.c_int64(stride)
        actual=fn(C.byref(nn),buf.ctypes.data_as(C.POINTER(C.c_double)),C.byref(ss))
        expected=ref(x)
        same=(math.isnan(actual) and math.isnan(expected)) or np.float64(actual).tobytes()==np.float64(expected).tobytes()
        count={'mul':n if n>1 else 0,'add':n+3 if n>1 else 0,'sqrt':int(n>1),'abs':int(n==1)}
        rows.append(dict(n=n,stride=stride,pattern=pattern,actual_hex=actual.hex(),expected_hex=expected.hex(),same=same,count=count))
    out=dict(protocol_sha256=hashlib.sha256((HERE/'controls_protocol.json').read_bytes()).hexdigest(),numpy=np.__version__,longdouble_mantissa=np.finfo(np.longdouble).nmant,rows=rows,failures=sum(not r['same'] for r in rows),cpu=time.process_time()-start)
    (HERE/'controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__': main()
