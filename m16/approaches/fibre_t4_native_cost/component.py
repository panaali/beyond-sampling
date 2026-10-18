"""Frozen original noisy family0/side0; no other trained numerical evaluation."""
import hashlib,json,resource,time
import numpy as np
from tracer import HERE,load
PRIOR=HERE.parent/'fibre_t4_recovery';sha=hashlib.sha256((HERE/'tracer.py').read_bytes()).hexdigest()
for case in ('primitive','zero','constant0','plus3_minus0','plus0_minus3'):
 c=json.loads((HERE/('controls_'+case+'.json')).read_text());assert c['complete'] and c['tracer_sha256']==sha
identity=json.loads((PRIOR/'noise_identity.json').read_text());expected=json.loads((PRIOR/'noise_preflight.json').read_text());model=tuple(np.array(w) for w in identity['model'])
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==expected['model_sha256']
path=HERE/'component.json';assert not path.exists(),'Preserve previous attempts'
start=time.process_time();out={'model_sha256':expected['model_sha256'],'family':0,'side':0,'tracer_sha256':sha,'complete':False};path.write_text(json.dumps(out,indent=2)+'\n')
FX,F3,Ops,ledger=load(True);ops=Ops();stats={'traces':0,'pieces':0}
try:
 fam=FX.FAMS4[0];value=F3.face_err(model,fam,fam['sides'][0],ops,stats)
 out.update(raw_error_component=value,legacy_ops=ops.n,stats=stats)
 assert value==expected['raw_error_component'] and ops.n==expected['legacy_ops'] and stats==expected['stats'],out
 out['complete']=True
except Exception as exc:out['failure']=repr(exc);raise
finally:
 out.update(ledger=ledger.report(),cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='ledger'}),flush=True)
