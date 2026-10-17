"""Bounded driver helper controls, no rank/model evaluation."""
import ctypes as C
import hashlib,itertools,json,sys,time,importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('outer_counted_helpers',HERE.parent/'outer_review/helper_controls.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
N,sqrt,mn,mx=old.N,old.sqrt,old.mn,old.mx
lib=old.lib;IP=old.IP;DP=old.DP;I=C.c_int64;D=C.c_double
rot=lib.scipy_dlartg_64_;rot.argtypes=[DP]*5;rot.restype=None
norm=lib.scipy_dlange_64_;norm.argtypes=[C.c_char_p,IP,IP,DP,IP,DP,C.c_size_t];norm.restype=C.c_double
roundup=lib.droundup_lwork_;roundup.argtypes=[IP];roundup.restype=C.c_double
env=lib.scipy_ilaenv_64_;env.argtypes=[IP,C.c_char_p,C.c_char_p,IP,IP,IP,IP,C.c_size_t,C.c_size_t];env.restype=I
def sign(a,b):N.count+=1;return N(np.copysign(a.x,b.x))
def rotation(f,g):
    N.count=3 # two square roots and safmax/2 constant expression
    tiny=np.finfo(float).tiny;big=np.float64(1)/tiny
    rtmin,rtmax=N(np.sqrt(tiny)),N(np.sqrt(big/2))
    zero,one=N(0),N(1);f,g=N(f),N(g);f1,g1=abs(f),abs(g)
    if g==zero:c,s,r=one,zero,f
    elif f==zero:c,s,r=zero,sign(one,g),g1
    elif f1>rtmin and f1<rtmax and g1>rtmin and g1<rtmax:
        d=sqrt(f*f+g*g);c=f1/d;r=sign(d,f);s=g/r
    else:
        u=mn(N(big),mx(N(tiny),mx(f1,g1)));fs=f/u;gs=g/u
        d=sqrt(fs*fs+gs*gs);c=abs(fs)/d;r=sign(d,f);s=gs/r;r=r*u
    return [c.x,s.x,r.x],N.count
def hx(a):return [float(v).hex() for v in a]
def equal(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return bool(np.all((a.view(np.uint64)==b.view(np.uint64))|(np.isnan(a)&np.isnan(b))))

def main():
    out=HERE/'helper_controls.json';progress=HERE/'helper_progress.jsonl'
    assert not out.exists() and not progress.exists()
    proto=json.loads((HERE/'helper_protocol.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==proto['source_sha256']
    rows=[];start=time.process_time()
    def before(args):
        with progress.open('a') as fp:fp.write(json.dumps(dict(start=args))+'\n')
    def save(args,expected,actual,count,upper,extra=None):
        row=dict(args=args,expected=hx(expected),actual=hx(actual),count=count,upper=upper,match=equal(expected,actual),extra=extra);rows.append(row)
        with progress.open('a') as fp:fp.write(json.dumps(row)+'\n')
    grid=[0.,-0.,1.,-1.,np.finfo(float).tiny,np.nextafter(0.,1.),1e-300,1e300,np.finfo(float).max,-np.finfo(float).max]
    with np.errstate(all='ignore'):
        for f,g in itertools.product(grid,repeat=2):
            args=dict(kind='rotation',f=float(f).hex(),g=float(g).hex());before(args)
            expected,count=rotation(f,g);vals=[D(f),D(g),D(),D(),D()];rot(*map(C.byref,vals))
            save(args,expected,[v.value for v in vals[2:]],count,25)
        for m,n,pattern in itertools.product(range(1,25),range(1,5),range(3)):
            args=dict(kind='maxnorm',m=m,n=n,pattern=pattern);before(args)
            a=np.full((m+1,n),1e300,order='F');a[:m,:]=np.arange(m*n).reshape(m,n,order='F')/8-3
            if pattern==1:a[:m,:]=0
            if pattern==2:a[m//2,n//2]=np.nan
            expected=0.
            for j in range(n):
                for i in range(m):
                    t=abs(a[i,j])
                    if expected<t or np.isnan(t):expected=t
            mm,nn,ld=map(I,(m,n,m+1));w=np.zeros(max(m,n));actual=norm(b'M',C.byref(mm),C.byref(nn),a.ctypes.data_as(DP),C.byref(ld),w.ctypes.data_as(DP),1)
            save(args,[expected],[actual],3*m*n,3*m*n)
        for lwork in [0,1,5,268,3000,2**53-1,2**53,2**53+1,2**53+3,2**62-1]:
            args=dict(kind='roundup',lwork=lwork);before(args);v=float(lwork);count=0
            if int(v)<lwork:v=v*(1+np.finfo(float).eps);count=2
            integer=I(lwork);actual=roundup(C.byref(integer));save(args,[v],[actual],count,2)
        for name,m,n in proto['environment_shapes']:
            args=dict(kind='blocksize',name=name,m=m,n=n);before(args)
            ispec,mm,nn,neg=map(I,(1,m,n,-1));nb=env(C.byref(ispec),name.upper().encode(),b' ',C.byref(mm),C.byref(nn),C.byref(neg),C.byref(neg),len(name),1)
            save(args,[32],[nb],0,0,dict(blocks_absent=nb>=min(m,n),integer_character_control_unpriced=True))
    result=dict(rows=rows,cases=len(rows),failures=sum(not r['match'] or r['count']>r['upper'] for r in rows),cpu_seconds=time.process_time()-start,scope='Finite named source/helper ABI controls, not whole SVD replay; integer/character/memory/cast operations unpriced. Both-NaN comparison accepted explicitly for maxnorm stress controls.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    assert result['cases']==proto['cases'] and result['failures']==0
    assert all(r['extra']['blocks_absent'] for r in rows if r['args']['kind']=='blocksize')
if __name__=='__main__':main()
