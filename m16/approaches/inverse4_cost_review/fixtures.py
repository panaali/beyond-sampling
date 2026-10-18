"""Fixed finite/singular controls for actual inv→GESV dispatch, no mass calls."""
import argparse,ctypes,hashlib,json,resource,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent

def exact_inverse(A):
    M=[[F(float(x)) for x in row]+[F(int(i==j)) for j in range(4)] for i,row in enumerate(A)]
    for j in range(4):
        pivot=next(i for i in range(j,4) if M[i][j]);M[j],M[pivot]=M[pivot],M[j];d=M[j][j];M[j]=[x/d for x in M[j]]
        for i in range(4):
            if i!=j:d=M[i][j];M[i]=[x-d*y for x,y in zip(M[i],M[j])]
    return np.array([[float(x) for x in row[4:]] for row in M])
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('baseline','instrumented'));a=p.parse_args();resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();env=json.loads((HERE/'environment.json').read_text());assert all(hashlib.sha256(Path(env['binary_paths'][k]).read_bytes()).hexdigest()==v for k,v in env['binary_hashes'].items());spy=None
    if a.mode=='instrumented':spy=ctypes.CDLL(str(HERE/'spy.so'));spy.audit_stats.argtypes=[ctypes.POINTER(ctypes.c_int64)]
    D=np.array([[4.,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]])
    fixtures=[('identity',np.eye(4),True),('diagonal',np.diag([1.,2,4,8]),True),('reverse_permutation',np.eye(4)[::-1],True),('dense',D,True),('row_pivot',D[[3,1,0,2]],True),('scaled_dense',D*2.**-20,True),('near_singular',np.array([[1.,1,0,0],[1,1+2.**-40,0,0],[0,0,1,0],[0,0,0,1]]),True),('zero',np.zeros((4,4)),False),('duplicate_row',D[[0,1,1,3]],False),('subnormal_diagonal',np.diag([2.**-1023,1,1,1]),False),('batch',np.stack([np.eye(4),D]),False)]
    rows=[]
    for name,A,check in fixtures:
        if spy:spy.audit_reset()
        row=dict(name=name,shape=list(A.shape),finite_input=bool(np.isfinite(A).all()))
        try:
            with np.errstate(all='ignore'):value=np.linalg.inv(A)
            row.update(value_hex=[float(x).hex() for x in value.ravel()],finite_output=bool(np.isfinite(value).all()))
            if check:
                exact=exact_inverse(A);row['max_scaled_reference_error']=float(np.max(np.abs(value-exact))/max(1.,np.max(np.abs(exact))));row['reference_pass']=row['max_scaled_reference_error']<3e-14;assert row['reference_pass']
        except np.linalg.LinAlgError as exc:row.update(failure=type(exc).__name__,message=str(exc))
        if spy:
            data=(ctypes.c_int64*12)();spy.audit_stats(data);row['spy']=dict(zip(('calls','n4','nrhs4','lda4','ldb4','identity_rhs','finite_input','success','singular','invalid_argument','nontrivial_pivots','last_info'),map(int,data)));expected=2 if name=='batch' else 1;assert all(row['spy'][k]==expected for k in ('calls','n4','nrhs4','lda4','ldb4','identity_rhs','finite_input'))
        rows.append(row)
    if spy:
        baseline=json.loads((HERE/'baseline.json').read_text())['rows'];assert len(rows)==len(baseline)
        for row,old in zip(rows,baseline):assert {k:v for k,v in row.items() if k!='spy'}==old
    out=dict(mode=a.mode,rows=rows,cpu=time.process_time()-start,complete=True,no_trained_inputs=True,binary_hashes=env['binary_hashes']);(HERE/(a.mode+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(mode=a.mode,fixtures=len(rows),exact_reference_controls=sum('reference_pass' in r for r in rows),cpu=out['cpu'],complete=True)))
if __name__=='__main__':main()
