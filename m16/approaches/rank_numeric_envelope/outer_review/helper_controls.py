"""Frozen small LAPACK helper source/native controls; no model work."""
import ctypes as C
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
lib=C.CDLL(str(LIB));IP=C.POINTER(C.c_int64);DP=C.POINTER(C.c_double)
f2=lib.scipy_dlas2_64_;f2.argtypes=[DP]*5;f2.restype=None
fc=lib.scipy_dlascl_64_;fc.argtypes=[C.c_char_p,IP,IP,DP,DP,IP,IP,DP,IP,IP];fc.restype=None
fs=lib.scipy_dlasrt_64_;fs.argtypes=[C.c_char_p,IP,DP,IP];fs.restype=None
fi=lib.scipy_ieeeck_64_;fi.argtypes=[IP,C.POINTER(C.c_float),C.POINTER(C.c_float)];fi.restype=C.c_int64

class N:
    count=0
    def __init__(self,x):self.x=np.float64(x)
    @staticmethod
    def val(x):return x.x if isinstance(x,N) else np.float64(x)
    def op(self,o,f):N.count+=1;return N(f(self.x,self.val(o)))
    def __add__(self,o):return self.op(o,lambda a,b:a+b)
    def __sub__(self,o):return self.op(o,lambda a,b:a-b)
    def __mul__(self,o):return self.op(o,lambda a,b:a*b)
    def __truediv__(self,o):return self.op(o,lambda a,b:a/b)
    def __abs__(self):N.count+=1;return N(abs(self.x))
    def cmp(self,o,f):N.count+=1;return f(self.x,self.val(o))
    def __eq__(self,o):return self.cmp(o,lambda a,b:a==b)
    def __ne__(self,o):return self.cmp(o,lambda a,b:a!=b)
    def __lt__(self,o):return self.cmp(o,lambda a,b:a<b)
    def __gt__(self,o):return self.cmp(o,lambda a,b:a>b)
def sqrt(x):N.count+=1;return N(np.sqrt(x.x))
def mn(a,b):N.count+=1;return N(min(a.x,b.x))
def mx(a,b):N.count+=1;return N(max(a.x,b.x))

def las2(f,g,h):
    N.count=0;z,o,t=map(N,(0,1,2));fa,ga,ha=map(abs,map(N,(f,g,h)))
    lo,hi=mn(fa,ha),mx(fa,ha)
    if lo==z:
        small=z
        if hi==z:big=ga
        else:
            r=mn(hi,ga)/mx(hi,ga);big=mx(hi,ga)*sqrt(o+r*r)
    elif ga<hi:
        a=o+lo/hi;b=(hi-lo)/hi;u=ga/hi;u=u*u
        c=t/(sqrt(a*a+u)+sqrt(b*b+u));small=lo*c;big=hi/c
    else:
        u=hi/ga
        if u==z:small=(lo*hi)/ga;big=ga
        else:
            a=o+lo/hi;b=(hi-lo)/hi;au=a*u;bu=b*u
            c=o/(sqrt(o+au*au)+sqrt(o+bu*bu));small=(lo*c)*u
            small=small+small;big=ga/(c+c)
    return [small.x,big.x],N.count

def lascl(a,cf,ct):
    N.count=3 # successful positive-finite validation: zero and two NaN checks
    zero,one,small=map(N,(0,1,np.finfo(float).tiny));big=one/small
    cf,ct=N(cf),N(ct);a=list(map(N,a));iterations=0
    while True:
        iterations+=1
        assert iterations<=4,'Harness safety guard, not a bound proof'
        cf1=cf*small
        if cf1==cf:mul=ct/cf;done=True
        else:
            ct1=ct/big
            if ct1==ct:mul=ct;done=True;cf=one
            elif abs(cf1)>abs(ct) and ct!=zero:mul=small;done=False;cf=cf1
            elif abs(ct1)>abs(cf):mul=big;done=False;ct=ct1
            else:
                mul=ct/cf;done=True
                if mul==one:break
        a=[v*mul for v in a]
        if done:break
    return [v.x for v in a],N.count,iterations

def sort_source(a):
    # Descending insertion sort, matching Fortran backward scan and strict swap.
    a=list(a);count=0
    for i in range(1,len(a)):
        for j in range(i,0,-1):
            count+=1
            if a[j]>a[j-1]:a[j],a[j-1]=a[j-1],a[j]
            else:break
    return a,count

def hx(a):return [float(v).hex() for v in a]
def equal(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return bool(np.all((a.view(np.uint64)==b.view(np.uint64))|(np.isnan(a)&np.isnan(b))))

def main():
    out=HERE/'helper_controls.json';progress=HERE/'helper_progress.jsonl'
    assert not out.exists() and not progress.exists(),'Preserve existing attempt'
    rows=[];start=time.process_time()
    def save(kind,args,expected,actual,count,upper,extra=None):
        row=dict(kind=kind,args=args,expected=hx(expected),actual=hx(actual),count=count,upper=upper,match=equal(expected,actual),extra=extra)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
        rows.append(row)
    with np.errstate(all='ignore'):
        for args in itertools.product([0.,-0.,1.,-1.,1e-310,1e300],repeat=3):
            expected,count=las2(*args);native=[C.c_double(x) for x in args]+[C.c_double(),C.c_double()]
            f2(*map(C.byref,native));save('dlas2',hx(args),expected,[v.value for v in native[3:]],count,28)
        ends=[np.nextafter(0.,1.),np.finfo(float).tiny,1.,np.finfo(float).max]
        for k,cf,ct in itertools.product([1,4,7],ends,ends):
            a=np.array([(-1.)**j*(1. if j%2==0 else 1e-300) for j in range(k)])
            expected,count,it=lascl(a,cf,ct);actual=a.copy();iz=C.c_int64(0);m=C.c_int64(k);n=C.c_int64(1);info=C.c_int64();x=C.c_double(cf);y=C.c_double(ct)
            fc(b'G',C.byref(iz),C.byref(iz),C.byref(x),C.byref(y),C.byref(m),C.byref(n),actual.ctypes.data_as(DP),C.byref(m),C.byref(info))
            save('dlascl',dict(k=k,cfrom=float(cf).hex(),cto=float(ct).hex(),input=hx(a)),expected,actual,count,43+3*k,dict(iterations=it,info=info.value))
        cases=[a for n in range(1,5) for a in itertools.product([-1.,-0.,0.,1.],repeat=n)]+list(itertools.permutations([-2.,-1.,1.,2.]))
        for a in cases:
            expected,count=sort_source(a);actual=np.array(a);n=C.c_int64(len(a));info=C.c_int64()
            fs(b'D',C.byref(n),actual.ctypes.data_as(DP),C.byref(info))
            save('dlasrt',hx(a),expected,actual,count,len(a)*(len(a)-1)//2,dict(info=info.value))
        for spec in [0,1]:
            s=C.c_int64(spec);z=C.c_float(0);o=C.c_float(1);result=fi(C.byref(s),C.byref(z),C.byref(o))
            # Named source operations are independently enumerated in DERIVATION.
            save('ieeeck',dict(spec=spec,abi='float32'),[1],[result],18+12*spec,30)
    result=dict(rows=rows,cases=len(rows),cpu_seconds=time.process_time()-start,failures=sum(not r['match'] or r['count']>r['upper'] for r in rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    assert result['failures']==0
    assert all(r['extra']['iterations']<=3 for r in rows if r['kind']=='dlascl')
    assert all(r['extra']['info']==0 for r in rows if r['kind'] in ('dlascl','dlasrt'))

if __name__=='__main__':main()
