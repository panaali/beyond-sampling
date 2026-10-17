"""DLARF1F parent translation, using original native BLAS children."""
import ctypes as C
import hashlib
from pathlib import Path
import numpy as np
LIB=Path('/workspace/clean-neurons/.venv/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_-f48b354e.so')
assert hashlib.sha256(LIB.read_bytes()).hexdigest()=='432e1c1739731468ba305771af96773939e43016643e2d630f58cb45b6670e96'
lib=C.CDLL(str(LIB));I=C.c_int64;D=C.c_double;IP=C.POINTER(I);DP=C.POINTER(D)
native=lib.dlarf1f_;native.argtypes=[C.c_char_p,IP,IP,DP,IP,DP,DP,IP,DP];native.restype=None
gemv=lib.scipy_dgemv_64_;gemv.argtypes=[C.c_char_p,IP,IP,DP,DP,IP,DP,IP,DP,DP,IP];gemv.restype=None
axpy=lib.scipy_daxpy_64_;axpy.argtypes=[IP,DP,DP,IP,DP,IP];axpy.restype=None
ger=lib.scipy_dger_64_;ger.argtypes=[IP,IP,DP,DP,IP,DP,IP,DP,IP];ger.restype=None
scal=lib.scipy_dscal_64_;scal.argtypes=[IP,DP,DP,IP];scal.restype=None
def ptr(a,offset=0):return C.cast(a.ctypes.data+8*offset,DP)
def ref(x):return C.byref(x)

def translate(side,m,n,v,inc,tau,a,lda,work):
    count=1;children=[];left=side=='L';lastv=1;lastc=0
    if tau!=0:
        lastv=m if left else n
        while lastv>1:
            count+=1
            if v[(lastv-1)*inc]!=0:break
            lastv-=1
        if left:
            # ILADLC: at most two corners, then column-major backward scan.
            count+=1;corner=a[0,n-1]!=0
            if not corner:count+=1;corner=a[lastv-1,n-1]!=0
            if corner:lastc=n
            else:
                for j in range(n-1,-1,-1):
                    for i in range(lastv):
                        count+=1
                        if a[i,j]!=0:lastc=j+1;break
                    if lastc:break
        else:
            # ILADLR: count the possible repeated row1 comparison at I=0.
            count+=1;corner=a[m-1,0]!=0
            if not corner:count+=1;corner=a[m-1,lastv-1]!=0
            if corner:lastc=m
            else:
                for j in range(lastv):
                    i=m
                    while True:
                        count+=1
                        if a[max(i,1)-1,j]!=0 or i<1:break
                        i-=1
                    lastc=max(lastc,i)
    if lastc==0:return dict(count=count,lastv=lastv,lastc=lastc,children=children)
    one,zero=D(1),D(0);i1=I(1);ld=I(lda);iv=I(inc);lc=I(lastc)
    if lastv==1:
        count+=1;alpha=D(1-tau);stride=I(lda if left else 1)
        scal(ref(lc),ref(alpha),ptr(a),ref(stride));children.append(['scal',lastc,stride.value])
    else:
        lv=I(lastv-1);alpha=D(-tau);count+=2 # source evaluates -TAU twice
        if left:
            gemv(b'T',ref(lv),ref(lc),ref(one),ptr(a,1),ref(ld),ptr(v,inc),ref(iv),ref(zero),ptr(work),ref(i1))
            axpy(ref(lc),ref(one),ptr(a),ref(ld),ptr(work),ref(i1))
            axpy(ref(lc),ref(alpha),ptr(work),ref(i1),ptr(a),ref(ld))
            ger(ref(lv),ref(lc),ref(alpha),ptr(v,inc),ref(iv),ptr(work),ref(i1),ptr(a,1),ref(ld))
            children.extend([['gemvT',lastv-1,lastc],['axpy',lastc],['axpy',lastc],['ger',lastv-1,lastc]])
        else:
            gemv(b'N',ref(lc),ref(lv),ref(one),ptr(a,lda),ref(ld),ptr(v,inc),ref(iv),ref(zero),ptr(work),ref(i1))
            axpy(ref(lc),ref(one),ptr(a),ref(i1),ptr(work),ref(i1))
            axpy(ref(lc),ref(alpha),ptr(work),ref(i1),ptr(a),ref(i1))
            ger(ref(lc),ref(lv),ref(alpha),ptr(work),ref(i1),ptr(v,inc),ref(iv),ptr(a,lda),ref(ld))
            children.extend([['gemvN',lastc,lastv-1],['axpy',lastc],['axpy',lastc],['ger',lastc,lastv-1]])
    return dict(count=count,lastv=lastv,lastc=lastc,children=children)

def call(side,m,n,v,inc,tau,a,lda,work):
    mm,nn,ii,ll=map(I,(m,n,inc,lda));tt=D(tau)
    native(side.encode(),ref(mm),ref(nn),ptr(v),ref(ii),ref(tt),ptr(a),ref(ll),ptr(work))
