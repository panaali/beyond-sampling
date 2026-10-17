"""Complete remaining ordered side components once; reuse preflight component0."""
import hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
from observer import HERE,load,install
FX,F3,Ops=load();obs,guard=install(FX,F3)
assert hashlib.sha256((HERE/'observer.py').read_bytes()).hexdigest()=='1760f68e486fab78da14f25ada058bbbc9df2d9d84466ade5d9278da4d84ed1c'
assert len(FX.FAMS4)==24 and all(len(f['sides'])==2 for f in FX.FAMS4)
index=int(sys.argv[1]);assert 1<=index<48
identity=json.loads((HERE/'noise_identity.json').read_text());model=tuple(np.array(x) for x in identity['model'])
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==identity['model_sha256']
folder=HERE/'noise_components';folder.mkdir(exist_ok=True);path=folder/f'{index:02d}.json';assert not path.exists(),'Preserve previous attempts'
fi,si=divmod(index,2);start=time.process_time();out={'model_sha256':identity['model_sha256'],'index':index,'family':fi,'side':si,'complete':False,'observer_sha256':hashlib.sha256((HERE/'observer.py').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
# No estimator code is copied.
path.write_text(json.dumps(out,indent=2)+'\n');ops=Ops();stats={'traces':0,'pieces':0}
try:
 fam=FX.FAMS4[fi];v=F3.face_err(model,fam,fam['sides'][si],ops,stats)
 out.update(complete=True,raw_error_component=v,normalized_error_component=v/FX.S3)
except Exception as exc:
 out['failure']=repr(exc);raise
finally:
 out.update(legacy_ops=ops.n,stats=stats,observer=obs.counts,guard=guard,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('index','complete','cpu_seconds','legacy_ops','stats')}),flush=True)
