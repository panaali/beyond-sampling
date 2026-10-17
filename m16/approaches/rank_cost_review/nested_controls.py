"""Fixed observed shapes with known rank and exact baseline/interposition parity."""
import ast,ctypes,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main(mode):
    start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(20,20))
    env=json.loads((HERE/'environment.json').read_text())
    for name,sha in env['binary_hashes'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
    shapes=sorted({ast.literal_eval(s)[0] for c in env['census'].values() for s in c['shapes']})
    spy=ctypes.CDLL(str(HERE/'nested_spy.so')) if mode=='instrumented' else None
    if spy:spy.audit_stats.argtypes=[ctypes.POINTER(ctypes.c_int64)]
    rows=[]
    for m in shapes:
        k=min(m,4);D=np.zeros((m,4));D[np.arange(k),np.arange(k)]=np.array([1.,.5,.25,.125])[:k]
        near=D.copy();near[np.arange(k),np.arange(k)]=np.array([1.,2.**-40,2.**-60,0.])[:k]
        fixtures=[('zero',np.zeros((m,4)),0,0),('rank_one',np.ones((m,4)),1,1),('diagonal',D,k,k),('near_threshold',near,min(k,2),1),('large_diagonal',D*2.**500,k,k),('small_diagonal',D*2.**-500,k,0)]
        for name,A,default_expected,tol_expected in fixtures:
            if spy:spy.audit_reset();spy.nested_reset()
            values=np.linalg.svd(A,compute_uv=False);default=int(np.linalg.matrix_rank(A));fixed=int(np.linalg.matrix_rank(A,tol=1e-12))
            assert default==default_expected and fixed==tol_expected,(m,name,default,fixed)
            row=dict(m=m,n=4,name=name,singular_values_hex=[float(x).hex() for x in values],rank_default=default,rank_tol1e12=fixed)
            if spy:
                data=(ctypes.c_int64*12)();spy.audit_stats(data);keys=('calls','jobN','queries','executions','m','n','lda','last_lwork','success','nonconvergence','invalid_argument','last_info');row['spy']=dict(zip(keys,map(int,data)))
                assert row['spy']['calls']==row['spy']['jobN']==row['spy']['success']==6,row
                assert row['spy']['queries']==row['spy']['executions']==3,row
                assert row['spy']['m']==m and row['spy']['n']==4 and row['spy']['lda']>=m,row
            if spy:
                data=(ctypes.c_int64*10)();spy.nested_stats(data);row['nested']=dict(zip(('q1_calls','q1_order','q1_success','q1_failure','q2_calls','q2_order','q2_success','q2_failure','q2_iterations_sum','q2_iterations_max'),map(int,data)))
                assert k>2 or row['nested']['q2_calls']==0,row
            rows.append(row)
    if spy:
        old=json.loads((HERE/'baseline.json').read_text())['rows'];assert [{k:v for k,v in r.items() if k not in ('spy','nested')} for r in rows]==old
    result=dict(mode=mode,rows=rows,cpu_seconds=time.process_time()-start,binary_hashes=env['binary_hashes'],shapes=shapes,no_trained_matrix_inputs=True)
    (HERE/('nested_'+mode+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(mode=mode,cases=len(rows),cpu=result['cpu_seconds'])))
if __name__=='__main__':
    mode=sys.argv[1];assert mode in ('baseline','instrumented')
    try:main(mode)
    except Exception as exc:(HERE/('nested_'+mode+'_failure.json')).write_text(json.dumps(dict(failure=repr(exc)),indent=2)+'\n');raise
