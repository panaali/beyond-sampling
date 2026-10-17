"""Charge only original estimator assembly using saved component return values."""
import hashlib,importlib.util,json,resource,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;inv=json.loads((HERE/'inventory.json').read_text());native=Path(inv['native']);prior=Path(inv['prior'])
for path,sha in inv['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
spec=importlib.util.spec_from_file_location('assembly_t180_tracer',native/'tracer.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
rows=[json.loads((prior/'noise_preflight.json').read_text())]+[json.loads((prior/'noise_components'/f'{i:02d}.json').read_text()) for i in range(1,48)]
FX,F3,Ops,ledger=T.load(True);visits=[]

def saved(model,fam,side,ops,stats):
 i=len(visits);r=rows[i]
 assert fam is FX.FAMS4[r['family']] and side is fam['sides'][r['side']]
 visits.append(i);return r['raw_error_component']
F3.face_err=saved
path=HERE/'assembly.json';assert not path.exists();start=time.process_time();value=F3.est_F2L2_4(None)
assert visits==list(range(48));assert value.hex()==json.loads((prior/'noise_summary.json').read_text())['estimate'].hex()
out={'complete':True,'indices':visits,'estimate':value,'ledger':ledger.report(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cpu_seconds':time.process_time()-start,'scope':'Original outer estimator only, saved component results injected by uninstrumented callback. Component work accounted separately; no model evaluation or reusable-output advice assumed.'}
assert out['ledger']['named_numeric_total']==50 and out['ledger']['integer_control_total']==0
path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('ledger','indices')}))
