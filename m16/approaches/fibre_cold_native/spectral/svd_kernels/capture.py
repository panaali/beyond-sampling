"""Actual table children for six saved full SVDs, no source substitution."""
import ctypes,hashlib,json,resource,subprocess,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'capture.json';assert not dest.exists()
    pins=json.loads((HERE/'pins.json').read_text())
    for path,sha in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    path=next(p for p in pins if 'libscipy_openblas' in p);lib=ctypes.CDLL(path);lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    nm=subprocess.check_output(['nm','-n',path],text=True);syms={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])}
    base=ctypes.cast(lib.scipy_dgemm_64_,ctypes.c_void_p).value-syms['scipy_dgemm_64_'];table=ctypes.c_void_p.from_address(base+syms['gotoblas']).value
    expected=json.loads((HERE/'runtime_table.json').read_text());assert table-base==int(expected['table_relative_offset'],16)
    for e in expected['entries']:assert ctypes.c_void_p.from_address(table+int(e['offset'],16)).value-base==int(e['target'],16)
    names=json.loads((HERE/'names.json').read_text());saved=json.loads((HERE.parent.parent/'capture.json').read_text());assert saved['complete']
    build=['gcc','-shared','-fPIC','-O2',str(HERE/'spy.c'),'-o',str(HERE/'spy.so')];subprocess.run(build,check=True)
    spy=ctypes.CDLL(str(HERE/'spy.so'));spy.kernel_install.argtypes=[ctypes.c_size_t];spy.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.kernel_events.restype=ctypes.c_int64
    out={'complete':False,'rows':[],'pins_sha256':hashlib.sha256((HERE/'pins.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'spy_binary_sha256':hashlib.sha256((HERE/'spy.so').read_bytes()).hexdigest(),'build_command':build,'runtime_table_guard':True};start=time.process_time();spy.kernel_install(table)
    try:
        for i,row in enumerate(saved['rows']):
            if row['name']!='svd':continue
            assert row['kwargs']=={'dict':[]};a=row['args']['items'][0];A=np.array([float.fromhex(x) for x in a['hex']]).reshape(a['shape']);assert A.shape==(1,4)
            spy.kernel_reset();new={'saved_row':i};out['rows'].append(new);value=np.linalg.svd(A)
            new['output_hex']=[[float(x).hex() for x in z.ravel()] for z in value];new['parity']=new['output_hex']==[z['hex'] for z in row['output']['items']]
            buf=(ctypes.c_int64*(4096*10))();n=spy.kernel_events(buf);assert 0<n<=4096
            fields=('kind','m','n','k','lda','ldb','ldc','alpha_bits','beta_bits','result');new['events']=[dict(zip(fields,map(int,buf[j*10:(j+1)*10]))) for j in range(n)]
            for e in new['events']:e['kernel']=names[e['kind']]
            assert new['parity']
        assert len(out['rows'])==6;out['complete']=True
    except Exception as exc:out['failure']=repr(exc);raise
    finally:
        spy.kernel_restore();out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,dispatch_restored=True)
        dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
