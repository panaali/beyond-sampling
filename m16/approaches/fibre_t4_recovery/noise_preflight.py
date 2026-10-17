"""Frozen family0/side0 first missing noisy case; additive error component only."""
import hashlib,json,resource,time
import numpy as np
from observer import HERE,load,install
FX,F3,Ops=load();obs,guard=install(FX,F3)
identity=json.loads((HERE/'noise_identity.json').read_text());model=tuple(np.array(x) for x in identity['model'])
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==identity['model_sha256']
path=HERE/'noise_preflight.json';assert not path.exists(), 'Preserve previous attempts'
start=time.process_time();out={'model_sha256':identity['model_sha256'],'family':0,'side':0,'complete':False,'observer_sha256':hashlib.sha256((HERE/'observer.py').read_bytes()).hexdigest(),'scope':'One fixed face-side error contribution, not full accuracy or convergence certificate'}
path.write_text(json.dumps(out,indent=2)+'\n');ops=Ops();stats={'traces':0,'pieces':0}
try:
 fam=FX.FAMS4[0];v=F3.face_err(model,fam,fam['sides'][0],ops,stats)
 out.update(complete=True,raw_error_component=v,normalized_error_component=v/FX.S3)
finally:
 out.update(legacy_ops=ops.n,stats=stats,observer=obs.counts,guard=guard,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
