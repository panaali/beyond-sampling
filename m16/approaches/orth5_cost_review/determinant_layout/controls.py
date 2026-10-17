"""Fixed exact layouts; baseline once and transparent GETRF observer once."""
import argparse
import ctypes as C
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
class Event(C.Structure):
    _fields_=[('m',C.c_int64),('n',C.c_int64),('lda',C.c_int64),('info',C.c_int64),('piv',C.c_int64*4),('before',C.c_double*16)]

def fixtures():
    bases={3:np.array([[4.,1,2],[1,5,1],[2,1,6]]),4:np.array([[4.,1,2,1],[1,5,1,2],[2,1,6,1],[1,2,1,7]])}
    for d,B in bases.items():
        singular=B.copy();singular[-1]=0
        for pattern,M in [('identity',np.eye(d)),('permuted_dense',np.roll(B,1,axis=0)),('zero_row',singular)]:
            yield f'd{d}_{pattern}_scalar',pattern,M.copy()
            storage=np.empty((d,d,128),dtype=np.float64);A=storage.transpose(2,1,0);A[:]=M
            for j in range(128):A[j,0,0]+=j/128
            assert A.strides==(8,1024,128*d*8)
            yield f'd{d}_{pattern}_batch',pattern,A

def exact_det(M):
    a=[[F(float(x)) for x in r] for r in M];n=len(a);det=F(1)
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        if pivot is None:return F(0)
        if pivot!=j:a[j],a[pivot]=a[pivot],a[j];det=-det
        v=a[j][j];det*=v
        for i in range(j+1,n):
            q=a[i][j]/v
            for k in range(j+1,n):a[i][k]-=q*a[j][k]
            a[i][j]=F(0)
    return det

def runtime_guard(env):
    for k,v in env['binary_hashes'].items():assert hashlib.sha256(Path(env['binary_paths'][k]).read_bytes()).hexdigest()==v,k
    from numpy.linalg import _umath_linalg
    assert Path(_umath_linalg.__file__).resolve()==Path(env['binary_paths']['numpy_linalg']).resolve()
    path=Path(env['binary_paths']['openblas']).resolve();lib=C.CDLL(str(path))
    lib.scipy_openblas_get_num_threads64_.restype=C.c_int;assert lib.scipy_openblas_get_num_threads64_()==1
    bases=[]
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if len(fields)>=6 and fields[2]=='00000000' and Path(fields[-1]).resolve()==path:bases.append(int(fields[0].split('-')[0],16))
    assert len(bases)==1,bases
    base=bases[0]
    nm=subprocess.check_output(['nm','-n',str(path)],text=True)
    syms={p[2]:int(p[0],16) for line in nm.splitlines() if len(p:=line.split())==3 and all(c in '0123456789abcdef' for c in p[0])}
    actual=C.cast(lib.scipy_dgetrf_64_,C.c_void_p).value
    assert actual==base+syms['scipy_dgetrf_64_']
    table=C.c_void_p.from_address(base+syms['gotoblas']).value
    expected=json.loads((HERE/'runtime_dispatch.json').read_text());checks=[]
    for e in expected['functions']:
        offset=int(e['table_offset'],16);target=C.c_void_p.from_address(table+offset).value-base
        assert target==int(e['target_relative_elf_offset'],16),(offset,target)
        checks.append({'table_offset':offset,'target_relative':target})
    blocks={}
    for offset,value in expected['block_parameter_offsets'].items():
        observed=C.c_int.from_address(table+int(offset,16)).value;assert observed==value
        blocks[offset]=observed
    return actual,{'getrf_relative_offset':syms['scipy_dgetrf_64_'],'unique_zero_offset_mapping':True,'functions':checks,'blocks':blocks,'threads':1,'binary_hashes':env['binary_hashes']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['baseline','observed']);mode=ap.parse_args().mode
    dest=HERE/(mode+'.json');assert not dest.exists(),'Preserve previous attempt'
    start=time.process_time();out={'complete':False,'mode':mode,'rows':[],'python':sys.version,'numpy':np.__version__}
    try:
        pins=json.loads((HERE/'pins.json').read_text())
        for name,sha in pins.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
        build=json.loads((HERE/'build.json').read_text());assert build['complete']
        for name,sha in build['source_sha256'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha,name
        assert hashlib.sha256((HERE/'spy.so').read_bytes()).hexdigest()==build['library_sha256']
        assert C.sizeof(Event)==192 and {name:getattr(Event,name).offset for name,_ in Event._fields_}=={'m':0,'n':8,'lda':16,'info':24,'piv':32,'before':64}
        actual,out['runtime']=runtime_guard(json.loads((HERE/'det_environment.json').read_text()))
        out['source_pins']={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['controls.py','PROTOCOL.md','pins.json','build.json']}
        spy=None;old=None
        if mode=='observed':
            old=json.loads((HERE/'baseline.json').read_text());assert old['complete'] and old['source_pins']==out['source_pins']
            out['baseline_sha256']=hashlib.sha256((HERE/'baseline.json').read_bytes()).hexdigest()
            spy=C.CDLL(str(HERE/'spy.so'));spy.audit_real_address.restype=C.c_size_t
            assert spy.audit_real_address()==actual and C.cast(spy.scipy_dgetrf_64_,C.c_void_p).value!=actual
            global_scope=C.CDLL(None)
            assert C.cast(global_scope.audit_real_address,C.c_void_p).value==C.cast(spy.audit_real_address,C.c_void_p).value,'Observer must be preloaded before NumPy'
            spy.audit_copy.argtypes=[C.POINTER(Event),C.c_int64];spy.audit_copy.restype=C.c_int64
        else:assert not hasattr(C.CDLL(None),'audit_real_address'),'Baseline must not preload observer'
        for ix,(name,pattern,A) in enumerate(fixtures()):
            mats=A[None] if A.ndim==2 else A;d=A.shape[-1]
            row={'name':name,'pattern':pattern,'shape':list(A.shape),'strides':list(A.strides),'dtype':str(A.dtype),
                 'input_sha256':hashlib.sha256(A.tobytes()).hexdigest(),'matrix_hex':[[[float(x).hex() for x in r] for r in M] for M in mats]}
            out['rows'].append(row)
            if spy:spy.audit_start()
            try:value=np.asarray(np.linalg.det(A))
            finally:
                if spy:spy.audit_stop()
            assert hashlib.sha256(A.tobytes()).hexdigest()==row['input_sha256']
            row['output_hex']=[float(x).hex() for x in value.ravel()]
            if spy:
                events=(Event*256)();count=int(spy.audit_copy(events,256));assert count==len(mats),(name,count)
                row['events']=[]
                for j,e in enumerate(events[:count]):
                    copied=[[float(e.before[4*i+k]).hex() for k in range(d)] for i in range(d)]
                    record={'m':int(e.m),'n':int(e.n),'lda':int(e.lda),'info':int(e.info),'pivots':list(e.piv)[:d],'matrix_hex':copied};row['events'].append(record)
                    assert (e.m,e.n,e.lda)==(d,d,d) and copied==row['matrix_hex'][j]
                    assert (e.info>0 if pattern=='zero_row' else e.info==0)
                    assert all(1<=p<=d for p in record['pivots'])
                prior=old['rows'][ix]
                for key in ['name','pattern','shape','strides','dtype','input_sha256','matrix_hex','output_hex']:assert row[key]==prior[key],(name,key)
                row['exact_determinants']=prior['exact_determinants']
                row['reference_checks']=prior['reference_checks']
            else:
                qs=[exact_det(M) for M in mats];row['exact_determinants']=list(map(str,qs))
                row['reference_checks']=[bool(float(v)==0 if not q else abs(float(v)-float(q))<=1e-12*abs(float(q))) for v,q in zip(value.ravel(),qs)]
                assert all(row['reference_checks']),name
        assert len(out['rows'])==12 and sum(len(r['matrix_hex']) for r in out['rows'])==774
        if spy:assert sum(len(r['events']) for r in out['rows'])==774
        out['complete']=True
    except BaseException as e:out['failure']=repr(e);raise
    finally:
        out.update(cpu_seconds=time.process_time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   mass_model_MC_calls=0,scope='Twelve exact finite layouts; copy and API/runtime binding only. No complete native price or corpus transfer.')
        dest.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'complete':out['complete'],'mode':mode,'arrays':len(out['rows']),'scalar_factorizations':774,'cpu_seconds':out['cpu_seconds']}))

if __name__=='__main__':main()
