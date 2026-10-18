"""DLARFG exclusive source/count controls; native children are deliberately shared."""
import ctypes,hashlib,itertools,json,math,resource,time
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;P=json.loads((HERE/'protocol.json').read_text());I=ctypes.c_int64;D=ctypes.c_double;ip=ctypes.POINTER(I);dp=ctypes.POINTER(D)
LIBPATH=next(p for p in P['pins'] if 'numpy.libs/libscipy_openblas' in p)
class Native:
    def __init__(self):
        assert hashlib.sha256(Path(LIBPATH).read_bytes()).hexdigest()==P['pins'][LIBPATH];self.lib=ctypes.CDLL(LIBPATH)
        self.norm=self.lib.scipy_dnrm2_64_;self.norm.argtypes=[ip,dp,ip];self.norm.restype=D
        self.hypot=self.lib.scipy_dlapy2_64_;self.hypot.argtypes=[dp,dp];self.hypot.restype=D
        self.mach=self.lib.scipy_dlamch_64_;self.mach.argtypes=[ctypes.c_char_p,ctypes.c_size_t];self.mach.restype=D
        self.scale=self.lib.scipy_dscal_64_;self.scale.argtypes=[ip,dp,dp,ip];self.scale.restype=None
        self.reflect=self.lib.scipy_dlarfg_64_;self.reflect.argtypes=[ip,dp,dp,ip,dp];self.reflect.restype=None
    def dnrm2(self,n,x,inc):return self.norm(ctypes.byref(I(n)),x.ctypes.data_as(dp),ctypes.byref(I(inc)))
    def dlapy2(self,a,b):return self.hypot(ctypes.byref(D(a)),ctypes.byref(D(b)))
    def dlamch(self,c):return self.mach(c.encode(),1)
    def dscal(self,n,a,x,inc):self.scale(ctypes.byref(I(n)),ctypes.byref(D(a)),x.ctypes.data_as(dp),ctypes.byref(I(inc)))
    def dlarfg(self,n,alpha,x,inc):
        x=x.copy();a=D(alpha);tau=D(-99);self.reflect(ctypes.byref(I(n)),ctypes.byref(a),x.ctypes.data_as(dp),ctypes.byref(I(inc)),ctypes.byref(tau));return dict(alpha=a.value,tau=tau.value,x=x.tolist())

def counted(native,n,alpha,x,inc):
    x=x.copy();cnt=Counter();calls=Counter();knt=0
    def op(a,b,f):cnt['arithmetic']+=1;return f(a,b)
    def add(a,b):return op(a,b,lambda u,v:u+v)
    def sub(a,b):return op(a,b,lambda u,v:u-v)
    def mul(a,b):return op(a,b,lambda u,v:u*v)
    def div(a,b):return op(a,b,lambda u,v:u/v)
    def neg(a):cnt['negation']+=1;return -a
    def sign(a,b):cnt['sign']+=1;return math.copysign(abs(a),b)
    def ab(a):cnt['abs']+=1;return abs(a)
    def lt(a,b):cnt['comparison']+=1;return a<b
    def eq(a,b):cnt['comparison']+=1;return a==b
    def child(name,*args):calls[name]+=1;return getattr(native,name)(*args)
    def result(tau):return dict(alpha=alpha,tau=tau,x=x.tolist(),counts=dict(cnt),children=dict(calls),rescale_iterations=knt)
    if n<=1:return result(0.)
    xnorm=child('dnrm2',n-1,x,inc)
    if eq(xnorm,0):return result(0.)
    beta=neg(sign(child('dlapy2',alpha,xnorm),alpha));safmin=div(child('dlamch','S'),child('dlamch','E'))
    if lt(ab(beta),safmin):
        rsafmn=div(1.,safmin)
        while True:
            knt+=1;child('dscal',n-1,rsafmn,x,inc);beta=mul(beta,rsafmn);alpha=mul(alpha,rsafmn)
            if not (lt(ab(beta),safmin) and knt<20):break
        xnorm=child('dnrm2',n-1,x,inc);beta=neg(sign(child('dlapy2',alpha,xnorm),alpha))
    tau=div(sub(beta,alpha),beta);child('dscal',n-1,div(1.,sub(alpha,beta)),x,inc)
    for j in range(knt):beta=mul(beta,safmin)
    alpha=beta;return result(tau)

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();native=Native();c=P['reflector_controls'];rows=[];bound=dict(arithmetic=66,negation=2,sign=2,abs=21,comparison=22);childbound=dict(dnrm2=2,dlapy2=2,dlamch=2,dscal=21)
    for n,inc,alpha,pattern in itertools.product(c['n'],c['incx'],c['alpha'],c['x_patterns']):
        x=np.full(max(1,(n-2)*inc+1),-17.,dtype=float)
        for j in range(n-1):x[j*inc]=0 if pattern=='zero' else (1+j/8)*((-1)**j if pattern=='alternating' else 1)*(1e-310 if pattern=='tiny_positive' else 1)
        row=dict(n=n,incx=inc,alpha=alpha,pattern=pattern)
        try:
            a=counted(native,n,alpha,x,inc);b=native.dlarfg(n,alpha,x,inc);xx=np.array([a['alpha'],a['tau']]+a['x']);yy=np.array([b['alpha'],b['tau']]+b['x']);localbound={k:(v if n>1 else 0) for k,v in bound.items()};bounded=all(a['counts'].get(k,0)<=v for k,v in localbound.items()) and all(a['children'].get(k,0)<=(v if n>1 else 0) for k,v in childbound.items())
            parity=bool(np.allclose(xx,yy,rtol=P['relative_tolerance'],atol=P['absolute_tolerance'],equal_nan=True));row.update(source=a,native=b,bounded=bounded,bitwise_equal=bool(np.array_equal(xx.view(np.uint64),yy.view(np.uint64))),max_absolute_difference=float(np.max(abs(xx-yy))),passed=bool(parity and bounded))
        except Exception as exc:row.update(passed=False,error=repr(exc))
        rows.append(row)
    out=dict(complete=all(r['passed'] for r in rows),rows=rows,numeric_exclusive_bound=bound,child_call_bound=childbound,cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,qualification='Source wrapper with same native children versus native DLARFG; child correctness/cost is not independently validated.')
    (HERE/'reflector_controls.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}));assert out['complete']
if __name__=='__main__':main()
