"""New named-child capture of the frozen12 layouts, no mass/model calls."""
import ctypes,hashlib,json,resource,subprocess,time
from pathlib import Path
import numpy as np
from controls import fixtures,HERE,layout
NAMES=('scal','gemm','beta','incopy','itcopy','oncopy','otcopy','small_permit','small_nn','small_nt','small_tn','small_tt','small_b0_nn','small_b0_nt','small_b0_tn','small_b0_tt')
FIELDS=('kind','m','n','k','lda','ldb','ldc','alpha_bits','beta_bits','result')
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,20));start=time.process_time();manifest=json.loads((HERE/'manifest.json').read_text());path=manifest['binary_paths']['openblas']
    for k,p in manifest['binary_paths'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==manifest['binary_sha256'][k]
    lib=ctypes.CDLL(path);lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    nm=subprocess.check_output(['nm','-n',path],text=True);syms={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])}
    base=ctypes.cast(lib.scipy_dgemm_64_,ctypes.c_void_p).value-syms['scipy_dgemm_64_'];table=ctypes.c_void_p.from_address(base+syms['gotoblas']).value
    expected=json.loads((HERE/'runtime_table.json').read_text());assert table-base==int(expected['table_relative_offset'],16)
    for e in expected['entries']:assert ctypes.c_void_p.from_address(table+int(e['offset'],16)).value-base==int(e['target'],16)
    baseline={r['name']:r for r in json.loads((HERE/'baseline.json').read_text())['rows']};spy=ctypes.CDLL(str(HERE/'kernel_spy.so'));spy.kernel_install.argtypes=[ctypes.c_size_t];spy.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.kernel_events.restype=ctypes.c_int64
    rows=[];spy.kernel_install(table)
    try:
        for name,A,B in fixtures():
            old=baseline[name];assert layout(A)==old['A'] and layout(B)==old['B'];spy.kernel_reset();C=A@B
            assert hashlib.sha256(C.tobytes()).hexdigest()==old['output_sha256'];assert [float(x).hex() for x in C.ravel()]==old['output_hex']
            buf=(ctypes.c_int64*(4096*10))();n=spy.kernel_events(buf);assert 0<n<=4096
            events=[dict(zip(FIELDS,map(int,buf[i*10:(i+1)*10]))) for i in range(n)]
            for e in events:e['kernel']=NAMES[e['kind']]
            rows.append(dict(name=name,output_sha256=old['output_sha256'],bitwise_baseline_parity=True,events=events))
    finally:spy.kernel_restore()
    out=dict(rows=rows,complete=True,cpu_seconds=time.process_time()-start,binary_sha256=manifest['binary_sha256'],new_matrix_calls=12,new_mass_model_MC_calls=0,scope='Only named table children captured; static direct functions remain covered by separate source inventory, no complete count asserted.')
    (HERE/'kernel_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(complete=True,cpu_seconds=out['cpu_seconds'],kernels={r['name']:[e['kernel'] for e in r['events']] for r in rows})))
if __name__=='__main__':main()
