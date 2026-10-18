"""Fixed transparent actual det controls; no trained matrices or mass replay."""
import argparse,ctypes,hashlib,itertools,json,math,resource,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
NAMES=('getrf_calls','m4','n4','lda4','success','singular','invalid','nontrivial_pivots','negative_diagonal_success','nonfinite_diagonal_success','log_calls','nonfinite_log_inputs','exp_calls','nonfinite_exp_inputs')
def exact_det(A):
    total=F(0)
    for p in itertools.permutations(range(4)):
        term=F((-1)**sum(p[i]>p[j] for i in range(4) for j in range(i+1,4)))
        for i in range(4):term*=F(float(A[i,p[i]]))
        total+=term
    return total

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('baseline','instrumented'));args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time()
    env=json.loads((HERE/'environment.json').read_text());assert all(hashlib.sha256(Path(env['binary_paths'][k]).read_bytes()).hexdigest()==v for k,v in env['binary_hashes'].items())
    spy=ctypes.CDLL(str(HERE/'spy.so')) if args.mode=='instrumented' else None
    if spy:spy.audit_stats.argtypes=[ctypes.POINTER(ctypes.c_int64)]
    D=np.array([[4.,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]])
    fixtures=[('identity',np.eye(4)),('positive_diagonal',np.diag([1.,2,4,8])),('negative_diagonal',np.diag([-1.,-2,-4,-8])),('one_swap',np.eye(4)[[1,0,2,3]]),('dense',D),('three_cycle',D[[1,2,0,3]]),('near_singular',np.array([[1.,1,0,0],[1,1+2.**-40,0,0],[0,0,1,0],[0,0,0,1]])),('zero',np.zeros((4,4))),('duplicate_row',D[[0,1,1,3]]),('subnormal_diagonal',np.diag([2.**-1023,1,1,1])),('small_dense',D*2.**-250),('large_dense',D*2.**250),('overflow_diagonal',np.eye(4)*2.**300),('underflow_diagonal',np.eye(4)*2.**-300),('balanced_diagonal',np.diag([2.**500,2.**-500,2.**400,2.**-400])),('nan_diagonal',np.diag([float('nan'),1,1,1])),('infinite_diagonal',np.diag([float('inf'),1,1,1])),('batch',np.stack([np.eye(4),D]))]
    rows=[]
    for name,A in fixtures:
        row=dict(name=name,shape=list(A.shape),input_hex=[float(x).hex() for x in A.ravel()])
        with np.errstate(all='ignore'):
            if spy:spy.audit_start()
            try:value=np.linalg.det(A)
            finally:
                if spy:spy.audit_stop()
        row['value_hex']=[float(x).hex() for x in np.asarray(value).ravel()]
        if spy:
            data=(ctypes.c_int64*14)();spy.audit_stats(data);row['spy']=dict(zip(NAMES,map(int,data)));n=2 if name=='batch' else 1
            assert all(row['spy'][k]==n for k in ('getrf_calls','m4','n4','lda4','exp_calls'))
            assert row['spy']['log_calls']==4*row['spy']['success']
        if np.isfinite(A).all():
            mats=A[None] if A.ndim==2 else A
            exacts=[exact_det(M) for M in mats];row['exact_determinants']=[str(x) for x in exacts];checks=[]
            for v,q in zip(np.asarray(value).ravel(),exacts):
                try:ref=float(q)
                except OverflowError:ref=math.copysign(float('inf'),1 if q>0 else -1)
                if math.isinf(ref):passed=math.isinf(float(v)) and np.signbit(v)==np.signbit(ref);kind='overflow'
                elif ref==0:passed=float(v)==0.;kind='exact_zero_or_underflow'
                elif math.isnan(float(v)):passed=False;kind='unexpected_nan'
                else:passed=abs(float(v)-ref)<=1e-12*abs(ref)+2.**-1074;kind='relative_or_subnormal'
                checks.append(dict(kind=kind,passed=bool(passed),reference_hex=ref.hex()))
            row['reference_checks']=checks
            assert all(c['passed'] for c in checks),(name,checks,row['value_hex'])
        rows.append(row)
    if spy:
        old=json.loads((HERE/'baseline.json').read_text())['rows']
        assert [{k:v for k,v in r.items() if k!='spy'} for r in rows]==old
    out=dict(mode=args.mode,complete=True,rows=rows,cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,qualification='Fixed controls verify transparent dispatch only; exact determinants do not certify general log-exp roundoff.',binary_hashes=env['binary_hashes'])
    (HERE/(args.mode+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(mode=args.mode,complete=True,fixtures=len(rows),cpu=out['cpu_seconds'])))
if __name__=='__main__':main()
