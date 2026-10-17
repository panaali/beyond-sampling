"""LD_PRELOAD transparent saved cold SVD/Legendre inputs only."""
import ctypes,hashlib,json,os,resource,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def encode(a):
    if isinstance(a,tuple):return [encode(x) for x in a]
    return [float(x).hex() for x in np.asarray(a).ravel()]
def expected(row):
    return [x['hex'] for x in row['output']['items']]
def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'capture.json';assert not dest.exists()
    manifest=json.loads((HERE/'capture_manifest.json').read_text())
    for p,h in manifest['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    assert os.environ['LD_PRELOAD']==str(HERE/'spy.so')
    spy=ctypes.CDLL(str(HERE/'spy.so'));spy.events.argtypes=[ctypes.POINTER(ctypes.c_int64)];spy.events.restype=ctypes.c_int64
    names=json.loads((HERE/'event_names.json').read_text());saved=json.loads((HERE.parent.parent/'capture.json').read_text());assert saved['complete']
    out={'complete':False,'rows':[],'manifest_sha256':hashlib.sha256((HERE/'capture_manifest.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'spy_binary_sha256':hashlib.sha256((HERE/'spy.so').read_bytes()).hexdigest()};start=time.process_time()
    try:
        for i,row in enumerate(saved['rows']):
            if row['name'] not in ('svd','leggauss'):continue
            assert row['kwargs']=={'dict':[]},'Only frozen default keyword calls supported'
            spy.reset();new={'saved_row':i,'name':row['name']};out['rows'].append(new)
            if row['name']=='svd':
                a=row['args']['items'][0];A=np.array([float.fromhex(x) for x in a['hex']]).reshape(a['shape']);value=np.linalg.svd(A)
            else:value=np.polynomial.legendre.leggauss(row['args']['items'][0])
            new['output_hex']=encode(value);new['parity']=new['output_hex']==expected(row)
            buf=(ctypes.c_int64*(512*5))();n=spy.events(buf);assert 0<n<=512
            new['events']=[{'function':names[buf[5*j]],'args':list(buf[5*j+1:5*j+5])} for j in range(n)]
            assert new['parity']
            if row['name']=='leggauss':
                assert sum(e['function']=='dsterf' for e in new['events'])==1
                spy.sterf_input.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double)];spy.sterf_input.restype=ctypes.c_int64
                dd=(ctypes.c_double*16)();ee=(ctypes.c_double*16)();nn=spy.sterf_input(dd,ee);assert nn==row['args']['items'][0]
                new['sterf_input_D_hex']=[float(dd[j]).hex() for j in range(nn)];new['sterf_input_E_hex']=[float(ee[j]).hex() for j in range(nn-1)]
        assert len(out['rows'])==12;out['complete']=True
    except Exception as exc:out['failure']=repr(exc);raise
    finally:
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
if __name__=='__main__':main()
