import ctypes,hashlib,itertools,json,resource,time
from pathlib import Path
import numpy as np
from q5 import q5,envelope
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'protocol.json').read_text())

def native(z,i0,n0,pp,tau,sigma,ieee,eps):
    libpath=next(p for p in P['pins'] if 'numpy.libs/libscipy_openblas' in p);assert hashlib.sha256(Path(libpath).read_bytes()).hexdigest()==P['pins'][libpath]
    lib=ctypes.CDLL(libpath);f=lib.scipy_dlasq5_64_;I=ctypes.c_int64;D=ctypes.c_double
    ip,dp=ctypes.POINTER(I),ctypes.POINTER(D);f.argtypes=[ip,ip,dp,ip,dp,dp,dp,dp,dp,dp,dp,dp,ip,dp];f.restype=None
    z=np.array(z,dtype=np.float64);ints=[I(i0),I(n0),I(pp),I(int(ieee))];vals=[D(tau),D(sigma)]+[D(-99) for _ in range(6)]+[D(eps)]
    f(ctypes.byref(ints[0]),ctypes.byref(ints[1]),z.ctypes.data_as(dp),ctypes.byref(ints[2]),*[ctypes.byref(v) for v in vals[:-1]],ctypes.byref(ints[3]),ctypes.byref(vals[-1]))
    return dict(z=z.tolist(),tau=vals[0].value,**dict(zip(('dmin','dmin1','dmin2','dn','dnm1','dnm2'),[v.value for v in vals[2:8]])))

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();c=P['controls'];rows=[]
    for length,i0,pp,tau,ieee in itertools.product(c['active_lengths'],c['i0'],c['pp'],c['tau'],c['ieee']):
        n0=i0+length-1;z=[1+j/8 for j in range(4*n0)];row=dict(length=length,i0=i0,n0=n0,pp=pp,tau=tau,ieee=ieee)
        try:
            a=q5(z,i0,n0,pp,tau,c['sigma'],ieee,c['eps']);b=native(z,i0,n0,pp,tau,c['sigma'],ieee,c['eps']);x=np.array(a['z']+[a[k] for k in ('tau','dmin','dmin1','dmin2','dn','dnm1','dnm2')]);y=np.array(b['z']+[b[k] for k in ('tau','dmin','dmin1','dmin2','dn','dnm1','dnm2')]);bound=envelope(length,ieee);bounded=all(a['counts'].get(k,0)<=v for k,v in bound.items());parity=bool(np.allclose(x,y,atol=P['absolute_tolerance'],rtol=P['relative_tolerance'],equal_nan=True))
            row.update(source=a,native=b,bound=bound,bounded=bounded,bitwise_equal=bool(np.array_equal(x.view(np.uint64),y.view(np.uint64))),max_absolute_difference=float(np.max(abs(x-y))),passed=bool(parity and bounded))
        except Exception as exc:row.update(passed=False,error=repr(exc))
        rows.append(row)
    out=dict(complete=all(r['passed'] for r in rows),rows=rows,cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen finite source/native controls, not full binary equivalence or complete SVD pricing.')
    (HERE/'controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));assert out['complete']
if __name__=='__main__':main()
