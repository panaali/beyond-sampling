"""Transparent own-process kernel-call capture on the already frozen fixtures."""
import ctypes,hashlib,json,resource,time,subprocess
from pathlib import Path
import numpy as np
from .inventory import HERE,LIB

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();env=json.loads((HERE/'environment.json').read_text());assert hashlib.sha256(LIB.read_bytes()).hexdigest()==env['binary_hashes']['openblas'];lib=ctypes.CDLL(str(LIB));nm=subprocess.check_output(['nm','-n',str(LIB)],text=True);symbols={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])};base=ctypes.cast(lib.scipy_dgesv_64_,ctypes.c_void_p).value-symbols['scipy_dgesv_64_'];table=ctypes.c_void_p.from_address(base+symbols['gotoblas']).value;spy=ctypes.CDLL(str(HERE/'kernel_spy.so'));spy.kernel_install.argtypes=[ctypes.c_size_t];spy.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.kernel_events.restype=ctypes.c_int64
    D=np.array([[4.,1,2,0],[1,5,0,1],[2,0,6,1],[0,1,1,3]])
    fixtures=[('identity',np.eye(4)),('diagonal',np.diag([1.,2,4,8])),('reverse_permutation',np.eye(4)[::-1]),('dense',D),('row_pivot',D[[3,1,0,2]]),('scaled_dense',D*2.**-20),('near_singular',np.array([[1.,1,0,0],[1,1+2.**-40,0,0],[0,0,1,0],[0,0,0,1]])),('zero',np.zeros((4,4))),('duplicate_row',D[[0,1,1,3]]),('subnormal_diagonal',np.diag([2.**-1023,1,1,1])),('batch',np.stack([np.eye(4),D]))]
    baseline={r['name']:r for r in json.loads((HERE/'baseline.json').read_text())['rows']};names=['iamax','dot','scal','swap','gemv','gemm','copy','upper','lower','upack','lpack'];rows=[];spy.kernel_install(table)
    try:
        for name,A in fixtures:
            spy.kernel_reset();row=dict(name=name)
            try:
                value=np.linalg.inv(A);row['value_hex']=[float(x).hex() for x in value.ravel()];assert row['value_hex']==baseline[name]['value_hex']
            except np.linalg.LinAlgError as exc:row['failure']=type(exc).__name__;assert row['failure']==baseline[name]['failure']
            data=(ctypes.c_int64*(256*5))();count=spy.kernel_events(data);assert count>=0;row['events']=[dict(kernel=names[data[5*i]],args=list(data[5*i+1:5*i+5])) for i in range(count)];rows.append(row)
    finally:spy.kernel_restore()
    out=dict(rows=rows,complete=True,cpu=time.process_time()-start,all_baseline_outputs_bitwise_unchanged=True,temporary_own_process_table_restored=True,binary_sha256=env['binary_hashes']['openblas']);(HERE/'kernel_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
