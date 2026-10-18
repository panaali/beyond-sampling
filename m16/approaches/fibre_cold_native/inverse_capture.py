"""Transparent installed kernel events for saved cold3x3 inverse inputs."""
import ctypes,hashlib,json,resource,subprocess,time
from pathlib import Path
import numpy as np
from capture import HERE,PINS,encode

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'inverse_capture.json';assert not dest.exists()
    for path,sha in PINS['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    capture=HERE/'capture.json';data=json.loads(capture.read_text());assert data['complete']
    rows=[r for r in data['rows'] if r['name']=='inv'];assert len(rows)==48
    fixtures=[]
    for i,row in enumerate(rows):
        a=row['args']['items'][0];A=np.array([float.fromhex(x) for x in a['hex']]).reshape(a['shape'])
        fixtures.append((f'cold_{i}',A,row['output']))
    D=np.array([[4.,1,2],[1,5,0],[2,0,6]])
    for name,A in [('identity',np.eye(3)),('dense',D),('pivot',D[[2,1,0]]),('near',np.array([[1.,1,0],[1,1+2.**-40,0],[0,0,1]])),('zero',np.zeros((3,3))),('duplicate',D[[0,1,1]])]:
        try:value=encode(np.linalg.inv(A))
        except np.linalg.LinAlgError:value={'failure':'LinAlgError'}
        fixtures.append((name,A,value))
    libpath=Path(next(p for p in PINS['files'] if 'libscipy_openblas' in p));lib=ctypes.CDLL(str(libpath))
    lib.scipy_openblas_get_num_threads64_.restype=ctypes.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    nm=subprocess.check_output(['nm','-n',str(libpath)],text=True)
    symbols={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])}
    base=ctypes.cast(lib.scipy_dgesv_64_,ctypes.c_void_p).value-symbols['scipy_dgesv_64_']
    table=ctypes.c_void_p.from_address(base+symbols['gotoblas']).value
    runtime=json.loads((HERE/'inverse_runtime_expected.json').read_text())
    assert table-base==int(runtime['table_relative_elf_offset'],16)
    for row in runtime['functions']:
        assert ctypes.c_void_p.from_address(table+int(row['table_offset'],16)).value-base==int(row['target_relative_elf_offset'],16)
    for offset,value in runtime['block_parameter_offsets'].items():
        assert ctypes.c_int.from_address(table+int(offset,16)).value==value
    build=['gcc','-shared','-fPIC','-O2',str(HERE/'inverse_spy.c'),'-o',str(HERE/'inverse_spy.so')]
    subprocess.run(build,check=True)
    spy=ctypes.CDLL(str(HERE/'inverse_spy.so'));spy.kernel_install.argtypes=[ctypes.c_size_t];spy.kernel_events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.kernel_events.restype=ctypes.c_int64
    names=['iamax','dot','scal','swap','gemv','gemm','copy','upper','lower','upack','lpack']
    start=time.process_time();out={'complete':False,'capture_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'spy_source_sha256':hashlib.sha256((HERE/'inverse_spy.c').read_bytes()).hexdigest(),'spy_binary_sha256':hashlib.sha256((HERE/'inverse_spy.so').read_bytes()).hexdigest(),'build_command':build,'runtime_targets_and_parameters_match':True,'runtime_expected_sha256':hashlib.sha256((HERE/'inverse_runtime_expected.json').read_bytes()).hexdigest(),'rows':[]}
    spy.kernel_install(table)
    try:
        for name,A,expected in fixtures:
            spy.kernel_reset();row={'name':name,'input':encode(A)};out['rows'].append(row)
            try:value=encode(np.linalg.inv(A))
            except np.linalg.LinAlgError:value={'failure':'LinAlgError'}
            row['output']=value
            buf=(ctypes.c_int64*(256*5))();count=spy.kernel_events(buf);assert 0<=count<=256
            row['events']=[{'kernel':names[buf[5*i]],'args':list(buf[5*i+1:5*i+5])} for i in range(count)]
            row['parity']=value==expected;assert row['parity'],name
        out['complete']=True
    except Exception as exc:out['failure']=repr(exc);raise
    finally:
        spy.kernel_restore();out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,dispatch_restored=True)
        dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
