"""Frozen small BLAS ABI/layout controls, exact dyadic/Fraction references."""
from fractions import Fraction as F
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
from application import I,D,ref,ptr,axpy,ger,gemv
HERE=Path(__file__).resolve().parent
def bits(a):return np.asarray(a).ravel(order='F').view(np.uint64).tolist()
def fma(a,b,c):return float(F(float(a))*F(float(b))+F(float(c)))
def main():
    out=HERE/'blas_controls.json';progress=HERE/'blas_progress.jsonl'
    assert not out.exists() and not progress.exists()
    rows=[];start=time.process_time()
    def before(args):
        with progress.open('a') as f:f.write(json.dumps(dict(start=args))+'\n')
    def save(args,expected,actual,untouched=True):
        row=dict(args=args,expected=bits(expected),actual=bits(actual),untouched=bool(untouched))
        row['match']=row['expected']==row['actual'] and row['untouched'];rows.append(row)
        with progress.open('a') as f:f.write(json.dumps(row)+'\n')
    lengths=[1,2,3,4,5,7,8,9,15,16,17,23]
    for n,ix,iy,alpha,pattern in itertools.product(lengths,[1,3],[1,3],[0.,.25,1.,-1.,float(np.pi),1e-100],[0,1]):
        args=dict(kind='axpy',n=n,ix=ix,iy=iy,alpha=alpha,pattern=pattern);before(args)
        x=np.full(n*ix,29.);x[::ix]=[(-1.)**j*(j+1)/8 for j in range(n)]
        y=np.full(n*iy,31.);y[::iy]=[(-1.)**(j+1)*(j+2)/16 for j in range(n)]
        if pattern:x[::ix][1::2]=0
        xx=x.copy();expected=y.copy()
        if alpha!=0:
            for j in range(n):expected[j*iy]=fma(alpha,x[j*ix],y[j*iy])
        nn,xi,yi=map(I,(n,ix,iy));aa=D(alpha)
        axpy(ref(nn),ref(aa),ptr(x),ref(xi),ptr(y),ref(yi))
        save(args,expected,y,np.array_equal(x,xx))
    for m,n,ix,iy,alpha,pattern in itertools.product([1,2,3,5,7,23],[1,2,3],[1,3],[1,3],[0.,-.25,1.],[0,1]):
        args=dict(kind='ger',m=m,n=n,ix=ix,iy=iy,alpha=alpha,pattern=pattern);before(args)
        x=np.full(m*ix,29.);x[::ix]=[(-1.)**j*(j+1)/8 for j in range(m)]
        y=np.full(n*iy,31.);y[::iy]=[(j+1)/16 for j in range(n)]
        if pattern:y[::iy][::2]=0
        a=np.full((m+1,n),37.,order='F');a[:m,:]=np.arange(1,m*n+1).reshape(m,n,order='F')/32
        expected=a.copy(order='F');xx=x.copy();yy=y.copy()
        if alpha!=0:
            for j in range(n):
                t=float(alpha*y[j*iy])
                for i in range(m):expected[i,j]=fma(t,x[i*ix],a[i,j])
        mm,nn,xi,yi,ld=map(I,(m,n,ix,iy,m+1));aa=D(alpha)
        ger(ref(mm),ref(nn),ref(aa),ptr(x),ref(xi),ptr(y),ref(yi),ptr(a),ref(ld))
        save(args,expected,a,np.array_equal(x,xx) and np.array_equal(y,yy))
    for trans,mmax in [('T',23),('N',5)]:
        for m,n,inc,pattern in itertools.product(range(1,mmax+1),[1,2,3],[1,3],[0,1]):
            args=dict(kind='gemv',trans=trans,m=m,n=n,inc=inc,pattern=pattern);before(args)
            a=np.full((m+1,n),37.,order='F');a[:m,:]=np.arange(1,m*n+1).reshape(m,n,order='F')/8
            if pattern:a[:m,:][::2,:]=0
            size=m if trans=='T' else n;outputs=n if trans=='T' else m
            x=np.full(size*inc,29.);x[::inc]=[(-1.)**j*(j+1)/16 for j in range(size)]
            y=np.full(outputs+2,np.nan);y[-2:]=43.;expected=y.copy()
            for k in range(outputs):
                values=a[:m,k] if trans=='T' else a[k,:]
                expected[k]=float(sum((F(float(z))*F(float(x[j*inc])) for j,z in enumerate(values)),F(0)))
            xx=x.copy();aa=a.copy(order='F');mm,nn,ld,ii,io=map(I,(m,n,m+1,inc,1));one,zero=D(1),D(0)
            gemv(trans.encode(),ref(mm),ref(nn),ref(one),ptr(a),ref(ld),ptr(x),ref(ii),ref(zero),ptr(y),ref(io))
            save(args,expected,y,np.array_equal(x,xx) and np.array_equal(a,aa))
    result=dict(rows=rows,cases=len(rows),failures=sum(not r['match'] for r in rows),cpu_seconds=time.process_time()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Exact Fraction scalar-FMA reference for AXPY/GER; exact dyadic GEMV reference with NaN output beta0 overwrite. Tests finite layout/branch semantics, not full compiled instruction counts or global numerical error.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}));assert result['cases']==1344 and result['failures']==0
if __name__=='__main__':main()
