"""One original family/side; unchanged reviewed tracer, no new reference."""
import hashlib,importlib.util,json,resource,signal,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text())
for path,sha in inv['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
idx=int(sys.argv[1]);assert 1<=idx<48
native=Path(inv['native']);prior=Path(inv['prior'])
spec=importlib.util.spec_from_file_location('t180_unchanged_tracer',native/'tracer.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
expected=json.loads((prior/'noise_components'/f'{idx:02d}.json').read_text());assert expected['complete']
model=tuple(np.array(w) for w in json.loads((prior/'noise_identity.json').read_text())['model'])
assert hashlib.sha256(b''.join(w.astype('<f8').tobytes() for w in model)).hexdigest()==inv['model_sha256']==expected['model_sha256']
folder=HERE/'components';folder.mkdir(exist_ok=True);path=folder/f'{idx:02d}.json';assert not path.exists(),'Preserve previous attempts'
out={'index':idx,'family':expected['family'],'side':expected['side'],'model_sha256':inv['model_sha256'],'complete':False,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inventory_sha256':hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest()}
path.write_text(json.dumps(out,indent=2)+'\n');start=time.process_time();ledger=None

def timeout(*_):raise TimeoutError('Frozen 110 CPU-second component cap')
signal.signal(signal.SIGPROF,timeout);signal.setitimer(signal.ITIMER_PROF,109)
try:
 FX,F3,Ops,ledger=T.load(True);ops=Ops();stats={'traces':0,'pieces':0}
 fam=FX.FAMS4[expected['family']];value=F3.face_err(model,fam,fam['sides'][expected['side']],ops,stats)
 out.update(raw_error_component=value,legacy_ops=ops.n,stats=stats)
 assert value.hex()==expected['raw_error_component'].hex() and ops.n==expected['legacy_ops'] and stats==expected['stats'],out
 out['complete']=True
except Exception as exc:out['failure']=repr(exc);raise
finally:
 signal.setitimer(signal.ITIMER_PROF,0)
 out.update(ledger=ledger.report() if ledger else None,cpu_seconds=time.process_time()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='ledger'}),flush=True)
