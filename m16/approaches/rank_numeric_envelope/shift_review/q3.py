"""Counted Q3 parent with unchanged native Q4/Q5/Q6 children."""
import ctypes as C
import hashlib,math
from collections import Counter
from pathlib import Path
import numpy as np
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
lib=C.CDLL(str(LIB));IP=C.POINTER(C.c_int64);DP=C.POINTER(C.c_double)
f4=lib.scipy_dlasq4_64_;f4.argtypes=[IP,IP,DP,IP,IP,DP,DP,DP,DP,DP,DP,DP,IP,DP];f4.restype=None
f5=lib.scipy_dlasq5_64_;f5.argtypes=[IP,IP,DP,IP,DP,DP,DP,DP,DP,DP,DP,DP,IP,DP];f5.restype=None
f6=lib.scipy_dlasq6_64_;f6.argtypes=[IP,IP,DP,IP,DP,DP,DP,DP,DP,DP];f6.restype=None

class N:
    counts=Counter()
    def __init__(self,x):self.x=float(x)
    @staticmethod
    def val(x):return x.x if isinstance(x,N) else float(x)
    def op(self,o,f):N.counts['arithmetic']+=1;return N(f(self.x,N.val(o)))
    def __add__(self,o):return self.op(o,lambda a,b:a+b)
    def __sub__(self,o):return self.op(o,lambda a,b:a-b)
    def __mul__(self,o):return self.op(o,lambda a,b:a*b)
    def __truediv__(self,o):return self.op(o,lambda a,b:a/b)
    def __neg__(self):N.counts['negation']+=1;return N(-self.x)
    def __abs__(self):N.counts['abs']+=1;return N(abs(self.x))
    def cmp(self,o,f):N.counts['comparison']+=1;return f(self.x,N.val(o))
    def __eq__(self,o):return self.cmp(o,lambda a,b:a==b)
    def __ne__(self,o):return self.cmp(o,lambda a,b:a!=b)
    def __lt__(self,o):return self.cmp(o,lambda a,b:a<b)
    def __gt__(self,o):return self.cmp(o,lambda a,b:a>b)
    def __le__(self,o):return self.cmp(o,lambda a,b:a<=b)
    def __ge__(self,o):return self.cmp(o,lambda a,b:a>=b)
def sqrt(x):N.counts['sqrt']+=1;return N(math.sqrt(x.x))
def isnan(x):N.counts['comparison']+=1;return math.isnan(x.x)

def parent(z,i0,n0,pp,dmin,sigma,desig,qmax,nfail,iteration,ndiv,ieee,ttype,dmin1,dmin2,dn,dn1,dn2,g,tau):
    N.counts=Counter();z=[N(0)]+list(map(N,z));children=Counter();entry=n0-i0+1
    dmin,sigma,desig,qmax,dmin1,dmin2,dn,dn1,dn2,g,tau=map(N,(dmin,sigma,desig,qmax,dmin1,dmin2,dn,dn1,dn2,g,tau))
    zero,quarter,half,one,two,hundred,bias,eps=map(N,(0,.25,.5,1,2,100,1.5,np.finfo(float).eps))
    def result():return dict(z=[v.x for v in z[1:]],n0=n0,pp=pp,nfail=nfail,iteration=iteration,ndiv=ndiv,ttype=ttype,values=[v.x for v in (dmin,sigma,desig,qmax,dmin1,dmin2,dn,dn1,dn2,g,tau)],counts=dict(N.counts),children=dict(children),upper={1:3,2:25,3:51,4:73}[entry]+16*children['q5'])
    n0in=n0;tol=eps*hundred;tol2=tol*tol
    while True:
        if n0<i0:return result()
        if n0==i0:
            z[4*n0-3]=z[4*n0+pp-3]+sigma;n0-=1;continue
        nn=4*n0+pp
        if n0!=i0+1:
            if not (z[nn-5]>tol2*(sigma+z[nn-3]) and z[nn-2*pp-4]>tol2*z[nn-7]):
                z[4*n0-3]=z[4*n0+pp-3]+sigma;n0-=1;continue
            if z[nn-9]>tol2*sigma and z[nn-2*pp-8]>tol2*z[nn-11]:break
        if z[nn-3]>z[nn-7]:z[nn-3],z[nn-7]=z[nn-7],z[nn-3]
        t=half*((z[nn-7]-z[nn-3])+z[nn-5])
        if z[nn-5]>z[nn-3]*tol2 and t!=zero:
            s=z[nn-3]*(z[nn-5]/t)
            if s<=t:s=z[nn-3]*(z[nn-5]/(t*(one+sqrt(one+s/t))))
            else:s=z[nn-3]*(z[nn-5]/(t+sqrt(t)*sqrt(t+s)))
            t=z[nn-7]+(s+z[nn-5]);z[nn-3]=z[nn-3]*(z[nn-7]/t);z[nn-7]=t
        z[4*n0-7]=z[nn-7]+sigma;z[4*n0-3]=z[nn-3]+sigma;n0-=2
    if pp==2:pp=0
    if dmin<=zero or n0<n0in:
        if bias*z[4*i0+pp-3]<z[4*n0+pp-3]:
            ipn4=4*(i0+n0)
            for j4 in range(4*i0,2*(i0+n0-1)+1,4):
                for a,b in [(j4-3,ipn4-j4-3),(j4-2,ipn4-j4-2),(j4-1,ipn4-j4-5),(j4,ipn4-j4-4)]:z[a],z[b]=z[b],z[a]
            if n0-i0<=4:z[4*n0+pp-1]=z[4*i0+pp-1];z[4*n0-pp]=z[4*i0-pp]
            dmin2=min(dmin2,z[4*n0+pp-1]);z[4*n0+pp-1]=min(z[4*n0+pp-1],z[4*i0+pp-1],z[4*i0+pp+3]);z[4*n0-pp]=min(z[4*n0-pp],z[4*i0-pp],z[4*i0-pp+4]);qmax=max(qmax,z[4*i0+pp-3],z[4*i0+pp+1]);dmin=-zero
    arr=np.array([v.x for v in z[1:]]);ii,nnn,par,nin,typ=map(C.c_int64,(i0,n0,pp,n0in,ttype));vs=[C.c_double(v.x) for v in (dmin,dmin1,dmin2,dn,dn1,dn2,tau,g)]
    f4(C.byref(ii),C.byref(nnn),arr.ctypes.data_as(DP),C.byref(par),C.byref(nin),*[C.byref(v) for v in vs[:7]],C.byref(typ),C.byref(vs[7]));children['q4']+=1;tau=N(vs[6].value);g=N(vs[7].value);ttype=typ.value
    while True:
        assert children['q5']<8,'Frozen control-only retry guard, not source theorem'
        arr=np.array([v.x for v in z[1:]]);ii,nnn,par,flag=map(C.c_int64,(i0,n0,pp,int(ieee)));vs=[C.c_double(v.x) for v in (tau,sigma,dmin,dmin1,dmin2,dn,dn1,dn2,eps)]
        f5(C.byref(ii),C.byref(nnn),arr.ctypes.data_as(DP),C.byref(par),*[C.byref(v) for v in vs[:8]],C.byref(flag),C.byref(vs[8]));children['q5']+=1
        z=[N(0)]+list(map(N,arr));tau,sigma,dmin,dmin1,dmin2,dn,dn1,dn2=map(N,[v.value for v in vs[:8]]);ndiv+=n0-i0+2;iteration+=1
        if dmin>=zero and dmin1>=zero:break
        elif dmin<zero and dmin1>zero and z[4*(n0-1)-pp]<tol*(sigma+dn1) and abs(dn)<tol*sigma:
            z[4*(n0-1)-pp+2]=zero;dmin=zero;break
        elif dmin<zero:
            nfail+=1
            if ttype< -22:tau=zero
            elif dmin1>zero:tau=(tau+dmin)*(one-two*eps);ttype-=11
            else:tau=quarter*tau;ttype-=12
            continue
        elif isnan(dmin):
            if tau!=zero:tau=zero;continue
        arr=np.array([v.x for v in z[1:]]);vs=[C.c_double(v.x) for v in (dmin,dmin1,dmin2,dn,dn1,dn2)]
        f6(C.byref(ii),C.byref(nnn),arr.ctypes.data_as(DP),C.byref(par),*[C.byref(v) for v in vs]);children['q6']+=1
        z=[N(0)]+list(map(N,arr));dmin,dmin1,dmin2,dn,dn1,dn2=map(N,[v.value for v in vs]);ndiv+=n0-i0+2;iteration+=1;tau=zero;break
    if tau<sigma:
        desig=desig+tau;t=sigma+desig;desig=desig-(t-sigma)
    else:t=sigma+tau;desig=sigma-(t-tau)+desig
    sigma=t
    return result()
