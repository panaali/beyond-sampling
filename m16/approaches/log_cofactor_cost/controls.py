"""Predeclared actual3x3 determinant controls; no mass or model evaluation."""
import argparse,ctypes,hashlib,itertools,json,math,resource,subprocess,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
NAMES=('getrf_calls','m3','n3','lda3','success','singular','invalid','nontrivial_pivots','negative_diagonal_success','nonfinite_diagonal_success','log_calls','nonfinite_log_inputs','exp_calls','nonfinite_exp_inputs')
KERNELS=('iamax','dot','scal','swap','gemv','gemm','copy','upper','lower','upack','lpack')
def fixtures():
    D=np.array([[4.,1,2],[1,5,0],[2,0,6]])
    return [('identity',np.eye(3)),('positive_diagonal',np.diag([1.,2,4])),('negative_diagonal',np.diag([-1.,-2,-4])),('one_swap',np.eye(3)[[1,0,2]]),('dense',D),('three_cycle',D[[1,2,0]]),('near_singular',np.array([[1.,1,0],[1,1+2.**-40,0],[0,0,1]])),('zero',np.zeros((3,3))),('duplicate_row',D[[0,1,1]]),('subnormal_diagonal',np.diag([2.**-1023,1,1])),('small_dense',D*2.**-250),('large_dense',D*2.**250),('overflow_diagonal',np.eye(3)*2.**400),('underflow_diagonal',np.eye(3)*2.**-400),('balanced_diagonal',np.diag([2.**500,2.**-500,1])),('nan_diagonal',np.diag([float('nan'),1,1])),('infinite_diagonal',np.diag([float('inf'),1,1])),('batch',np.stack([np.eye(3),D]))]
def exact_det(A):
    total=F(0)
    for p in itertools.permutations(range(3)):
        term=F((-1)**sum(p[i]>p[j] for i in range(3) for j in range(i+1,3)))
        for i in range(3):term*=F(float(A[i,p[i]]))
        total+=term
    return total
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('baseline','instrumented','kernels'));mode=ap.parse_args().mode
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
    env=json.loads((HERE/'frozen/det_environment.json').read_text())
    for k,v in env['binary_hashes'].items():assert hashlib.sha256(Path(env['binary_paths'][k]).read_bytes()).hexdigest()==v
    lib=ctypes.CDLL(env['binary_paths']['openblas']);lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    spy=ctypes.CDLL(str(HERE/'spy.so')) if mode=='instrumented' else None
    kernel=ctypes.CDLL(str(HERE/'kernel_spy.so')) if mode=='kernels' else None
    if spy:spy.audit_stats.argtypes=[ctypes.POINTER(ctypes.c_int64)]
    if kernel:
        nm=subprocess.check_output(['nm','-n',env['binary_paths']['openblas']],text=True)
        syms={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])}
        base=ctypes.cast(lib.scipy_dgetrf_64_,ctypes.c_void_p).value-syms['scipy_dgetrf_64_'];table=ctypes.c_void_p.from_address(base+syms['gotoblas']).value
        kernel.kernel_install.argtypes=[ctypes.c_size_t];kernel.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];kernel.kernel_events.restype=ctypes.c_int64;kernel.kernel_install(table)
    rows=[]
    try:
        for name,A in fixtures():
            r=dict(name=name,shape=list(A.shape),input_hex=[float(x).hex() for x in A.ravel()])
            with np.errstate(all='ignore'):
                if spy:spy.audit_start()
                if kernel:kernel.kernel_reset()
                try:value=np.linalg.det(A)
                finally:
                    if spy:spy.audit_stop()
            r['value_hex']=[float(x).hex() for x in np.asarray(value).ravel()]
            if spy:
                data=(ctypes.c_int64*14)();spy.audit_stats(data);r['spy']=dict(zip(NAMES,map(int,data)));n=2 if name=='batch' else 1
                assert all(r['spy'][k]==n for k in ('getrf_calls','m3','n3','lda3','exp_calls'))
                assert r['spy']['log_calls']==3*r['spy']['success']
            if kernel:
                data=(ctypes.c_int64*(256*5))();count=kernel.kernel_events(data);assert count>=0
                r['events']=[dict(kernel=KERNELS[data[5*i]],args=list(data[5*i+1:5*i+5])) for i in range(count)]
                assert all(e['kernel'] in ('iamax','dot','scal','swap','gemv') for e in r['events'])
                assert all(e['args'][0]<=3 for e in r['events'])
            if np.isfinite(A).all():
                mats=A[None] if A.ndim==2 else A;qs=[exact_det(M) for M in mats];r['exact_determinants']=[str(q) for q in qs];checks=[]
                for v,q in zip(np.asarray(value).ravel(),qs):
                    try:ref=float(q)
                    except OverflowError:ref=math.copysign(float('inf'),1 if q>0 else -1)
                    if math.isinf(ref):ok=math.isinf(float(v)) and np.signbit(v)==np.signbit(ref);kind='overflow'
                    elif ref==0:ok=float(v)==0.;kind='exact_zero_or_underflow'
                    else:ok=abs(float(v)-ref)<=1e-12*abs(ref)+2.**-1074;kind='relative_or_subnormal'
                    checks.append(dict(kind=kind,passed=bool(ok),reference_hex=ref.hex()))
                r['reference_checks']=checks;assert all(c['passed'] for c in checks),(name,checks)
            rows.append(r)
    finally:
        if kernel:kernel.kernel_restore()
    if mode!='baseline':
        old=json.loads((HERE/'baseline.json').read_text())['rows'];assert [{k:v for k,v in r.items() if k not in ('spy','events')} for r in rows]==old
    out=dict(mode=mode,rows=rows,complete=True,cpu_seconds=time.process_time()-start,binary_hashes=env['binary_hashes'],scope='Frozen18 analytic fixtures19 matrices; native numeric categories only; no mass/model replay.')
    (HERE/(mode+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(mode=mode,complete=True,cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
