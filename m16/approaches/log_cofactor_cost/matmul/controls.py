"""Frozen12 array-only native dispatch controls; no source mass/model calls."""
import argparse,ctypes,hashlib,json,resource,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
FIELDS=('kind','order','trans_or_uplo','trans2','m','n','k','lda','ldb','ldc','alpha_bits','beta_bits','same_pointer')
def fixtures():
    D=np.array([[4.,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]])
    I=np.eye(4);F=np.asfortranarray(D);S=(np.arange(64).reshape(8,8)%7-3).astype(float)[::2,::2]
    U=(np.arange(4000).reshape(1000,4)%7-3).astype(float)
    US=(np.arange(8000).reshape(2000,4)%7-3).astype(float)[::2]
    return [('identity_self',I,I.T),('dense_C_self',D,D.T),('dense_F_self',F,F.T),('strided_self',S,S.T),('independent_C',D,D.copy()),('mixed_F_C',F,D.copy()),('tall_zero',np.zeros((1000,4)),D),('tall_ones',np.ones((1000,4)),D),('tall_dense',U,D),('tall_transposed_right',U,D.T),('tall_F_left',np.asfortranarray(U),D),('tall_strided_left',US,D)]
def exact_product(A,B):
    return [[sum(int(A[i,k])*int(B[k,j]) for k in range(A.shape[1])) for j in range(B.shape[1])] for i in range(A.shape[0])]
def layout(A):
    return dict(shape=list(A.shape),strides=list(A.strides),C=bool(A.flags.c_contiguous),F=bool(A.flags.f_contiguous),sha256_C_values=hashlib.sha256(A.tobytes(order='C')).hexdigest())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('baseline','instrumented'));mode=ap.parse_args().mode
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text())
    for k,path in manifest['binary_paths'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==manifest['binary_sha256'][k]
    lib=ctypes.CDLL(manifest['binary_paths']['openblas']);lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    spy=ctypes.CDLL(str(HERE/'spy.so')) if mode=='instrumented' else None
    if spy:spy.audit_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.audit_events.restype=ctypes.c_int64
    rows=[]
    for name,A,B in fixtures():
        row=dict(name=name,A=layout(A),B=layout(B),input_same_pointer=bool(A.ctypes.data==B.ctypes.data))
        if spy:spy.audit_start()
        try:C=A@B
        finally:
            if spy:spy.audit_stop()
        row['output_hex']=[float(x).hex() for x in C.ravel()]
        expected=exact_product(A,B);assert np.array_equal(C,np.array(expected,dtype=float));row['exact_integer_reference']=True
        row['output_sha256']=hashlib.sha256(C.tobytes()).hexdigest()
        if spy:
            buf=(ctypes.c_int64*(64*13))();n=spy.audit_events(buf);assert 0<n<=64
            row['events']=[dict(zip(FIELDS,map(int,buf[i*13:(i+1)*13]))) for i in range(n)]
            assert all(e['kind'] in (0,1) and e['m'] in (4,1000) and e['k']==4 for e in row['events'])
        rows.append(row)
    if spy:
        old=json.loads((HERE/'baseline.json').read_text())['rows'];assert [{k:v for k,v in r.items() if k!='events'} for r in rows]==old
    out=dict(mode=mode,rows=rows,complete=True,cpu_seconds=time.process_time()-start,binary_sha256=manifest['binary_sha256'],new_mass_model_MC_calls=0,scope='Only12 frozen exact integer shape/layout fixtures; named dispatch/parity, no kernel price or general array guarantee.')
    (HERE/(mode+'.json')).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(mode=mode,complete=True,cpu_seconds=out['cpu_seconds'])))
if __name__=='__main__':main()
