"""Cold-only original-source import with transparent numerical call capture."""
import hashlib, importlib.util, json, math, resource, sys, time, types
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PINS=json.loads((HERE/'pins.json').read_text())
NAMES=('fiber3','fibre','fibre4','fibre4b')

def encode(value):
    if isinstance(value,np.ndarray):
        return {'array':True,'dtype':value.dtype.str,'shape':list(value.shape),'hex':[float(x).hex() for x in value.ravel()],'bytes_sha256':hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value,(float,np.floating)):return {'float':float(value).hex()}
    if isinstance(value,(int,np.integer)):return int(value)
    if isinstance(value,(tuple,list)):return {'kind':type(value).__name__,'items':[encode(x) for x in value]}
    if isinstance(value,dict):return {'dict':[[encode(k),encode(v)] for k,v in value.items()]}
    if value is None or isinstance(value,str):return value
    raise TypeError(type(value))

def prohibited(*args,**kwargs):
    raise AssertionError('Forbidden model I/O or clipping during cold-only capture')

def run(capture,targets):
    rows=[];old={n:sys.modules.get(n) for n in (*NAMES,'m43')}
    saved_load=np.load;saved=[np.linalg.svd,np.linalg.inv,np.polynomial.legendre.leggauss]
    stub=types.ModuleType('m43');stub.clip=prohibited;sys.modules['m43']=stub
    def wrapper(name,fn):
        def call(*args,**kwargs):
            row={'name':name,'args':encode(args),'kwargs':encode(kwargs)}
            rows.append(row)
            result=fn(*args,**kwargs)
            row['output']=encode(result)
            return result
        return call
    old_path=list(sys.path)
    try:
        np.load=prohibited
        if capture:
            np.linalg.svd=wrapper('svd',saved[0]);np.linalg.inv=wrapper('inv',saved[1]);np.polynomial.legendre.leggauss=wrapper('leggauss',saved[2])
        outputs={}
        for name in NAMES:
            path=Path(PINS['source_dir'])/(name+'.py')
            spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
            spec.loader.exec_module(mod)
            outputs[name]={key:encode(getattr(mod,key)) for key in targets[name]}
        return outputs,rows
    finally:
        np.load=saved_load;np.linalg.svd,np.linalg.inv,np.polynomial.legendre.leggauss=saved
        sys.path[:]=old_path
        for name,value in old.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(20,22));dest=HERE/'capture.json';assert not dest.exists(),'Preserve prior attempt'
    for path,sha in PINS['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    prior=json.loads((Path(PINS['prior'])/'cold.json').read_text());targets={n:list(prior['modules'][n]['targets']) for n in NAMES}
    out={'complete':False,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pins_sha256':hashlib.sha256((HERE/'pins.json').read_bytes()).hexdigest()};start=time.process_time()
    try:
        before=time.process_time();baseline,_=run(False,targets);out['baseline_cpu']=time.process_time()-before
        before=time.process_time();observed,rows=run(True,targets);out.update(capture_cpu=time.process_time()-before,rows=rows,objects=observed)
        assert baseline==observed,'Cold object parity failure'
        assert sum(r['name']=='svd' for r in rows)==6
        assert sum(r['name']=='inv' for r in rows)==48
        assert [r['args']['items'][0] for r in rows if r['name']=='leggauss']==PINS['expected']['leggauss']
        assert all(r['args']['items'][0]['shape']==[1,4] for r in rows if r['name']=='svd')
        assert all(r['args']['items'][0]['shape']==[3,3] for r in rows if r['name']=='inv')
        out.update(complete=True,all_assigned_objects_bitwise_equal=True,forbidden_model_io_or_clipping_calls=0)
    except Exception as exc:out['failure']=repr(exc);raise
    finally:
        out.update(cpu=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','objects')}))

if __name__=='__main__':main()
